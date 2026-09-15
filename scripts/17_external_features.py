"""TÜİK + EPİAŞ özellikli level_shape_C CV ve (esik tutarsa) submission.

Kullanim: python scripts/17_external_features.py
Kabul: blend ref C'den >=0.01 iyi VE cold kotulesmesin.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import (
    build_panel,
    entity_static,
    load_sample_submission,
    load_test,
    load_train,
)
from src.external.weather import CACHE_PATH, load_weather, weather_features
from src.features.build import assemble, model_matrix
from src.features.history import add_lagged_history, entity_history_features
from src.features.peer import peer_tables
from src.models.level_shape import entity_level, fit_shape_model, predict_level_shape
from src.models.pipeline import fold_frames, mask_history_entities, xy
from src.tracking import log_experiment
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

C_BLEND_REF = 1.3359
C_COLD_REF = 1.8746


def eval_c(split, static, weather, *, n_trees=300):
    train_f, valid_f, hist = fold_frames(split, static, weather)
    Xtr, ytr, cat, logg_tr, tr_use = xy(train_f)
    Xva, _, _, logg_va, va_use = xy(valid_f, dropna_target=True)
    Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
    Xva = Xva[Xtr.columns]
    Xtr_m = mask_history_entities(Xtr, tr_use[C.ENTITY], 0.30, C.SEED)
    level_tr = entity_level(hist, tr_use[C.ENTITY], logg_tr)
    level_va = entity_level(hist, va_use[C.ENTITY], logg_va)
    hist_cols = [c for c in Xtr_m.columns if c.startswith("hist_")]
    model = fit_shape_model(
        Xtr_m, ytr, level_tr, cat, hist_cols, seed=C.SEED + 3, n_trees=n_trees
    )
    pred = predict_level_shape(model, Xva, level_va)
    metrics = segment_report(va_use[C.TARGET], pred, split.segment, is_cold=split.is_cold)
    # coverage of new features
    cov = {}
    for col in ("log_nufus", "log_epias_total_lag1", "epias_yoy_ratio"):
        if col in va_use.columns:
            cov[col] = float(va_use[col].notna().mean())
    return metrics, cov, model


def build_submission(weather, train, test, sample, static) -> Path:
    origin = C.TRAIN_END
    hist = entity_history_features(train, origin)
    peer = peer_tables(train, origin)
    tr = add_lagged_history(train.sort_values([C.ENTITY, C.DATE], ignore_index=True))
    train_f = assemble(
        tr, origin=origin, history=hist, weather=weather, static=static, peer=peer, include_history=False
    )
    test_f = assemble(
        test, origin=origin, history=hist, weather=weather, static=static, peer=peer, include_history=True
    )
    y = np.log1p(train_f[C.TARGET].clip(lower=0).to_numpy())
    Xtr, cat = model_matrix(train_f)
    Xte, _ = model_matrix(test_f)
    Xtr, Xte = Xtr.align(Xte, join="outer", axis=1, fill_value=np.nan)
    Xte = Xte[Xtr.columns]
    Xtr_m = mask_history_entities(Xtr, train_f[C.ENTITY], 0.30, C.SEED)
    level_tr = entity_level(hist, train_f[C.ENTITY], train_f["log_guc"].to_numpy())
    level_te = entity_level(hist, test_f[C.ENTITY], test_f["log_guc"].to_numpy())
    hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]
    print("egitim C + external...")
    m = fit_shape_model(Xtr_m, y, level_tr, cat, hist_cols, seed=C.SEED + 3, n_trees=400)
    pred = predict_level_shape(m, Xte, level_te)
    out = sample.copy()
    out[C.TARGET] = np.clip(pred, 0.0, None)
    path = C.SUBMISSIONS_DIR / "submission_c_external.csv"
    out.to_csv(path, index=False)
    print(f"yazildi {path} mean={out[C.TARGET].mean():.1f}")
    return path


def main() -> None:
    C.ensure_dirs()
    assert CACHE_PATH.exists()
    weather = weather_features(load_weather())
    train = load_train()
    test = load_test()
    sample = load_sample_submission()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))

    rows = []
    lines = [
        "# C + TÜİK nüfus + EPİAŞ il-ay\n",
        f"Uretim: {pd.Timestamp.now()}\n",
        f"Ref C (weather-only era): blend={C_BLEND_REF:.4f} cold={C_COLD_REF:.4f}\n",
        "EPİAŞ: lag-1 ay + YoY (aynı ay gerçekleşen yok).\n",
    ]

    for fold in C.FOLDS:
        print(f"=== {fold.name} ===")
        split = make_fold(train, fold, profile, seed=C.SEED)
        metrics, cov, _ = eval_c(split, static, weather)
        rows.append({"model": "C_external", "fold": fold.name, **metrics})
        log_experiment(
            exp_id="C_external",
            model="level_shape_C_external",
            target="log1p",
            feature_set="weather+tuik+epias",
            fold=fold.name,
            metrics=metrics,
            notes=str(cov),
        )
        lines.append(f"\n## Fold {fold.name}\n")
        lines.append(
            f"| all | warm | cold | blend | cov_nufus | cov_epias |\n"
            f"| --- | --- | --- | --- | --- | --- |\n"
            f"| {metrics['rmsle_all']:.4f} | {metrics.get('rmsle_warm', float('nan')):.4f} | "
            f"{metrics.get('rmsle_cold', float('nan')):.4f} | {metrics.get('rmsle_blend', float('nan')):.4f} | "
            f"{cov.get('log_nufus', float('nan')):.3f} | {cov.get('log_epias_total_lag1', float('nan')):.3f} |\n"
        )
        print(
            f"  blend={metrics['rmsle_blend']:.4f} warm={metrics['rmsle_warm']:.4f} "
            f"cold={metrics['rmsle_cold']:.4f} cov={cov}"
        )

    df = pd.DataFrame(rows)
    b = summarize_folds(df, "rmsle_blend")["mean"]
    c = summarize_folds(df, "rmsle_cold")["mean"]
    submit_ok = (b <= C_BLEND_REF - 0.01) and (c <= C_COLD_REF + 0.01)
    # softer: improve blend 0.01 vs ref OR improve cold 0.02 without blend worsen
    submit_ok = submit_ok or (
        (c <= C_COLD_REF - 0.02) and (b <= C_BLEND_REF + 0.005)
    )
    decision = {
        "blend": float(b),
        "cold": float(c),
        "delta_blend": float(b - C_BLEND_REF),
        "delta_cold": float(c - C_COLD_REF),
        "submit_ok": bool(submit_ok),
    }
    lines.append(
        f"\n## Ozet\nblend={b:.4f} ({b - C_BLEND_REF:+.4f}) cold={c:.4f} ({c - C_COLD_REF:+.4f})\n"
        f"submit_ok: **{submit_ok}**\n"
    )
    (C.REPORTS_DIR / "18_external_features.md").write_text("".join(lines), encoding="utf-8")
    (C.REPORTS_DIR / "18_external_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print("DECISION", decision)

    if submit_ok:
        path = build_submission(weather, train, test, sample, static)
        decision["submission"] = str(path)
        (C.REPORTS_DIR / "18_external_decision.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )
    else:
        print("Esik tutulmadi — submission yok.")


if __name__ == "__main__":
    main()
