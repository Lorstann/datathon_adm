"""A/B/C modelleri, sifir kapisi, ensemble. 3 fold validasyon.

Kullanim: python scripts/06_train.py
Hava cache yoksa takvim+panel ile devam eder.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import build_panel, entity_static, load_test, load_train
from src.external.weather import CACHE_PATH, load_weather, weather_features
from src.models.boosting import fit_catboost, fit_lightgbm, fit_xgboost
from src.models.ensemble import apply_log_shift, cold_warm_blend, fit_log_shift, shift_is_stable
from src.models.level_shape import entity_level, fit_shape_model, predict_level_shape
from src.models.pipeline import fold_frames, mask_history_entities, xy
from src.models.zero_gate import apply_zero_gate, choose_threshold, zero_probability
from src.tracking import log_experiment
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold


def _weather():
    if not CACHE_PATH.exists():
        print("hava cache yok, hava ozellikleri atlandi")
        return None
    return weather_features(load_weather())


def _log(exp_id, model, fold, metrics, notes, target="log1p", feature_set="full"):
    log_experiment(
        exp_id=exp_id,
        model=model,
        target=target,
        feature_set=feature_set,
        fold=fold,
        metrics=metrics,
        notes=notes,
    )


def main() -> None:
    C.ensure_dirs()
    train = load_train()
    test = load_test()
    panel = build_panel(train, test)
    static = entity_static(panel)
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = _weather()

    records = []
    fold_shifts: list[dict] = []
    gate_thresholds = []
    lines = ["# Model validasyonu\n", f"Uretim: {pd.Timestamp.now()}\n"]

    for fold in C.FOLDS:
        split = make_fold(train, fold, profile, seed=C.SEED)
        train_f, valid_f, hist = fold_frames(split, static, weather)
        Xtr, ytr, cat, logg_tr, tr_use = xy(train_f)
        Xva, yva, _, logg_va, va_use = xy(valid_f, dropna_target=True)
        Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
        Xva = Xva[Xtr.columns]

        Xtr_masked = mask_history_entities(Xtr, tr_use[C.ENTITY], 0.30, C.SEED)

        # --- A: global LightGBM log1p ---
        m_a = fit_lightgbm(Xtr_masked, ytr, cat, seed=C.SEED, target="log1p", mask_history_rate=0.0, n_trees=250)
        pred_a = m_a.predict(Xva, logg_va)

        # --- A2: z hedefi ---
        yz = ytr - logg_tr
        m_z = fit_lightgbm(Xtr_masked, yz, cat, seed=C.SEED + 1, target="z", mask_history_rate=0.0, n_trees=250)
        pred_z = m_z.predict(Xva, logg_va)

        # --- sifir kapisi ---
        p0 = zero_probability(hist, va_use[C.ENTITY]).to_numpy()
        thr, _ = choose_threshold(va_use[C.TARGET].to_numpy(), pred_a, p0)
        gate_thresholds.append(thr)
        pred_a_gate = apply_zero_gate(pred_a, p0, threshold=thr)

        # --- B: cold kolu = z modeli (gecmis yok) ---
        hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]
        Xtr_cold = Xtr.copy()
        Xtr_cold[hist_cols] = np.nan
        Xva_cold = Xva.copy()
        Xva_cold[hist_cols] = np.nan
        m_cold = fit_lightgbm(Xtr_cold, yz, cat, seed=C.SEED + 2, target="z", mask_history_rate=0.0, n_trees=250)
        pred_cold = m_cold.predict(Xva_cold, logg_va)
        pred_b = cold_warm_blend(pred_a_gate, apply_zero_gate(pred_cold, p0, threshold=thr), split.is_cold)

        # --- C: seviye + sekil ---
        level_tr = entity_level(hist, tr_use[C.ENTITY], logg_tr)
        level_va = entity_level(hist, va_use[C.ENTITY], logg_va)
        m_shape = fit_shape_model(Xtr_masked, ytr, level_tr, cat, hist_cols, seed=C.SEED + 3)
        pred_c = predict_level_shape(m_shape, Xva, level_va)

        # --- CatBoost + XGBoost (cesitlilik) ---
        m_cat = fit_catboost(Xtr_masked, ytr, cat, seed=C.SEED, n_trees=150)
        pred_cat = m_cat.predict(Xva, logg_va)
        m_xgb = fit_xgboost(Xtr_masked, ytr, seed=C.SEED, n_trees=150)
        pred_xgb = m_xgb.predict(Xva, logg_va)

        pred_ens = 0.45 * pred_b + 0.25 * pred_z + 0.15 * pred_cat + 0.15 * pred_xgb
        pred_ens = apply_zero_gate(pred_ens, p0, threshold=thr)

        groups = np.where(split.is_cold, "cold", "warm") + "_" + va_use["il"].astype(str)
        shifts = fit_log_shift(va_use[C.TARGET].to_numpy(), pred_ens, pd.Series(groups))
        fold_shifts.append(shifts)
        pred_pp = apply_log_shift(pred_ens, pd.Series(groups), shifts)

        bundle = {
            "lgbm_A": pred_a,
            "lgbm_z": pred_z,
            "lgbm_A_gate": pred_a_gate,
            "lgbm_Bcoldwarm": pred_b,
            "level_shape_C": pred_c,
            "catboost": pred_cat,
            "xgboost": pred_xgb,
            "ensemble": pred_ens,
            "ensemble_shift": pred_pp,
        }
        lines.append(f"\n## Fold {fold.name}  gate_thr={thr:.2f}  checks={split.checks}\n")
        lines.append("| Model | all | warm | cold | blend |\n| --- | --- | --- | --- | --- |\n")
        for name, pred in bundle.items():
            m = segment_report(va_use[C.TARGET], pred, split.segment, is_cold=split.is_cold)
            _log(f"model_{name}", name, fold.name, m, notes=f"thr={thr:.2f}")
            records.append({"model": name, "fold": fold.name, **m})
            lines.append(
                f"| {name} | {m['rmsle_all']:.4f} | {m.get('rmsle_warm', float('nan')):.4f} | "
                f"{m.get('rmsle_cold', float('nan')):.4f} | {m.get('rmsle_blend', float('nan')):.4f} |\n"
            )

    df = pd.DataFrame(records)
    lines.append("\n## Fold ozeti (rmsle_blend mean / std)\n")
    lines.append("| Model | mean | std |\n| --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append(f"| {model} | {s['mean']:.4f} | {s['std']:.4f} |\n")

    stable = shift_is_stable(fold_shifts)
    n_stable = sum(stable.values())
    lines.append(
        f"\nPost-process kaydirma: {n_stable}/{len(stable)} grup isareti fold'lar arasi sabit. "
        "Karar kurali geregi yalnizca sabit isaretli gruplara uygulanir; "
        "tamami sabit degilse gonderimde kaydirma KAPALI kalir.\n"
    )
    med_thr = float(np.median(gate_thresholds)) if gate_thresholds else 0.85
    lines.append(f"Sifir kapisi esik medyani: {med_thr:.2f}\n")

    path = C.REPORTS_DIR / "05_models.md"
    path.write_text("".join(lines), encoding="utf-8")
    print(f"yazildi: {path}")
    print(df.groupby("model")["rmsle_blend"].mean().sort_values().to_string())


if __name__ == "__main__":
    main()
