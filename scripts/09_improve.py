"""Havayla CV + aile ablasyonu + cold yama karsilastirmasi + gain.

Kullanim: python scripts/09_improve.py
Gerektirir: data/external/weather/era5_daily.parquet
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
from src.models.boosting import fit_lightgbm
from src.models.level_shape import (
    entity_level,
    fit_shape_model,
    gain_table,
    predict_level_shape,
)
from src.models.pipeline import fold_frames, mask_history_entities, xy
from src.tracking import log_experiment
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

# Ablasyon: dusurulecek kolon onekleri / adlari
WEATHER_PREFIXES = (
    "temperature_",
    "apparent_",
    "relative_humidity",
    "precipitation",
    "wind_speed",
    "shortwave",
    "sunshine",
    "cdd",
    "hdd",
    "temp_",
)
CALENDAR_COLS = {
    "is_holiday",
    "is_arife",
    "is_kurban",
    "is_ramazan_bayram",
    "is_ramadan",
    "bayram_gun_indeksi",
    "bayrama_kalan_gun",
    "is_kopru",
    "is_weekend",
    "holiday_x_logguc",
    "weekend_x_logguc",
}
# takvim zaman kolonlarinin bir kismi (ay/hafta) no_calendar'da KALIR — sadece tatil/haftasonu
HISTORY_PREFIX = "hist_"


def _require_weather() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"hava cache yok: {CACHE_PATH}")
    w = weather_features(load_weather())
    print(f"hava: {len(w):,} satir, {w['lokasyon'].nunique()} lokasyon")
    return w


def _drop_family(X: pd.DataFrame, family: str) -> pd.DataFrame:
    if family == "full":
        return X
    out = X.copy()
    if family == "no_weather":
        cols = [
            c
            for c in out.columns
            if any(c.startswith(p) or c == p for p in WEATHER_PREFIXES)
            or c in ("cdd_x_logguc", "hdd_x_logguc")
        ]
        return out.drop(columns=cols, errors="ignore")
    if family == "no_calendar":
        cols = [c for c in out.columns if c in CALENDAR_COLS]
        return out.drop(columns=cols, errors="ignore")
    if family == "no_history":
        cols = [c for c in out.columns if c.startswith(HISTORY_PREFIX)]
        return out.drop(columns=cols, errors="ignore")
    raise ValueError(family)


def _level_for(
    hist: pd.DataFrame,
    frame: pd.DataFrame,
    log_guc: np.ndarray,
    *,
    cold_patch: bool,
) -> np.ndarray:
    if cold_patch:
        return entity_level(
            hist,
            frame[C.ENTITY],
            log_guc,
            peer_guc_z=frame["peer_guc_z"] if "peer_guc_z" in frame.columns else None,
        )
    return entity_level(hist, frame[C.ENTITY], log_guc)


def eval_c(
    split,
    static,
    weather,
    *,
    family: str = "full",
    cold_patch: bool = False,
    n_trees: int = 300,
    seed: int = C.SEED,
    return_model: bool = False,
):
    train_f, valid_f, hist = fold_frames(split, static, weather)
    Xtr, ytr, cat, logg_tr, tr_use = xy(train_f)
    Xva, yva, _, logg_va, va_use = xy(valid_f, dropna_target=True)
    Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
    Xva = Xva[Xtr.columns]
    Xtr = _drop_family(Xtr, family)
    Xva = _drop_family(Xva, family)
    Xtr_m = mask_history_entities(Xtr, tr_use[C.ENTITY], 0.30, seed)

    level_tr = _level_for(hist, tr_use, logg_tr, cold_patch=cold_patch)
    level_va = _level_for(hist, va_use, logg_va, cold_patch=cold_patch)
    hist_cols = [c for c in Xtr_m.columns if c.startswith("hist_")]
    model = fit_shape_model(
        Xtr_m, ytr, level_tr, cat, hist_cols, seed=seed + 3, n_trees=n_trees
    )
    pred = predict_level_shape(model, Xva, level_va)
    metrics = segment_report(
        va_use[C.TARGET], pred, split.segment, is_cold=split.is_cold
    )
    if return_model:
        return metrics, model, va_use, pred, split
    return metrics


def eval_a(split, static, weather, *, n_trees: int = 250, seed: int = C.SEED):
    train_f, valid_f, hist = fold_frames(split, static, weather)
    Xtr, ytr, cat, logg_tr, tr_use = xy(train_f)
    Xva, _, _, logg_va, va_use = xy(valid_f, dropna_target=True)
    Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
    Xva = Xva[Xtr.columns]
    Xtr_m = mask_history_entities(Xtr, tr_use[C.ENTITY], 0.30, seed)
    m = fit_lightgbm(Xtr_m, ytr, cat, seed=seed, n_trees=n_trees, mask_history_rate=0.0)
    pred = m.predict(Xva, logg_va)
    return segment_report(va_use[C.TARGET], pred, split.segment, is_cold=split.is_cold)


def main() -> None:
    C.ensure_dirs()
    weather = _require_weather()
    train = load_train()
    test = load_test()
    panel = build_panel(train, test)
    static = entity_static(panel)
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))

    # ---------- Adim 1: havayla CV ----------
    lines6 = [
        "# CV with weather (confirmed cache)\n",
        f"Uretim: {pd.Timestamp.now()}\n",
        f"Hava cache: `{CACHE_PATH}`\n",
    ]
    rows_cv = []
    last_model = None
    last_va = None
    last_pred = None
    last_split = None

    for fold in C.FOLDS:
        print(f"=== CV fold {fold.name} ===")
        split = make_fold(train, fold, profile, seed=C.SEED)
        m_a = eval_a(split, static, weather)
        m_c = eval_c(split, static, weather, family="full", cold_patch=False)
        m_c2, model, va_use, pred, _ = eval_c(
            split,
            static,
            weather,
            family="full",
            cold_patch=True,
            return_model=True,
        )
        if fold.name == "A_mevsim":
            last_model, last_va, last_pred, last_split = model, va_use, pred, split

        for name, m in (("lgbm_A", m_a), ("level_shape_C", m_c), ("level_shape_C_coldpatch", m_c2)):
            log_experiment(
                exp_id=f"wx_{name}",
                model=name,
                target="log1p",
                feature_set="full_weather",
                fold=fold.name,
                metrics=m,
                notes="havayla CV",
            )
            rows_cv.append({"model": name, "fold": fold.name, **m})
        lines6.append(f"\n## Fold {fold.name}\n")
        lines6.append("| Model | all | warm | cold | blend |\n| --- | --- | --- | --- | --- |\n")
        for name, m in (("lgbm_A", m_a), ("level_shape_C", m_c), ("level_shape_C_coldpatch", m_c2)):
            lines6.append(
                f"| {name} | {m['rmsle_all']:.4f} | {m.get('rmsle_warm', float('nan')):.4f} | "
                f"{m.get('rmsle_cold', float('nan')):.4f} | {m.get('rmsle_blend', float('nan')):.4f} |\n"
            )

    df_cv = pd.DataFrame(rows_cv)
    lines6.append("\n## Ozet (rmsle_blend)\n| Model | mean | std |\n| --- | --- | --- |\n")
    for model, g in df_cv.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines6.append(f"| {model} | {s['mean']:.4f} | {s['std']:.4f} |\n")

    base_c = summarize_folds(df_cv[df_cv.model == "level_shape_C"], "rmsle_blend")["mean"]
    patch_c = summarize_folds(df_cv[df_cv.model == "level_shape_C_coldpatch"], "rmsle_blend")[
        "mean"
    ]
    base_cold = summarize_folds(df_cv[df_cv.model == "level_shape_C"], "rmsle_cold")["mean"]
    patch_cold = summarize_folds(
        df_cv[df_cv.model == "level_shape_C_coldpatch"], "rmsle_cold"
    )["mean"]
    lines6.append(
        f"\nCold patch: blend {base_c:.4f} -> {patch_c:.4f} (delta {patch_c - base_c:+.4f}); "
        f"cold {base_cold:.4f} -> {patch_cold:.4f} (delta {patch_cold - base_cold:+.4f})\n"
    )
    # Kabul: blend en az ~0.01 iyilesir VEYA cold belirgin iyilesir ve blend kotulesmez
    accept_patch = (patch_c <= base_c - 0.01) or (
        patch_cold < base_cold - 0.02 and patch_c <= base_c + 0.005
    )
    lines6.append(f"Cold patch kabul: **{accept_patch}**\n")
    (C.REPORTS_DIR / "06_cv_with_weather.md").write_text("".join(lines6), encoding="utf-8")
    print("yazildi reports/06_cv_with_weather.md")

    # ---------- Adim 2: ablasyon Fold A + B ----------
    lines7 = [
        "# Feature family ablation (level_shape_C, with weather baseline)\n",
        f"Uretim: {pd.Timestamp.now()}\n",
    ]
    abl_rows = []
    for fold in (C.FOLDS[0], C.FOLDS[1]):  # A and B
        print(f"=== Ablation fold {fold.name} ===")
        split = make_fold(train, fold, profile, seed=C.SEED)
        lines7.append(f"\n## Fold {fold.name}\n")
        lines7.append("| Family | blend | warm | cold | delta_vs_full |\n| --- | --- | --- | --- | --- |\n")
        full_blend = None
        for fam in ("full", "no_weather", "no_calendar", "no_history"):
            m = eval_c(split, static, weather, family=fam, cold_patch=False)
            if fam == "full":
                full_blend = m["rmsle_blend"]
            delta = m["rmsle_blend"] - full_blend
            abl_rows.append({"fold": fold.name, "family": fam, **m, "delta": delta})
            log_experiment(
                exp_id=f"abl_{fam}",
                model="level_shape_C",
                target="log1p",
                feature_set=fam,
                fold=fold.name,
                metrics=m,
                notes=f"ablation delta={delta:+.4f}",
            )
            lines7.append(
                f"| {fam} | {m['rmsle_blend']:.4f} | {m.get('rmsle_warm', float('nan')):.4f} | "
                f"{m.get('rmsle_cold', float('nan')):.4f} | {delta:+.4f} |\n"
            )
            print(f"  {fam}: blend={m['rmsle_blend']:.4f} delta={delta:+.4f}")

    lines7.append(
        "\nPozitif delta = aileyi cikarmak skoru kotulestirir (aile faydali). "
        "Negatif delta = aile zararli veya gurultu.\n"
    )
    (C.REPORTS_DIR / "07_ablation.md").write_text("".join(lines7), encoding="utf-8")
    print("yazildi reports/07_ablation.md")

    # ---------- Adim 4: gain + cold hata dilimi ----------
    lines8 = ["# Feature gain and cold error slices\n", f"Uretim: {pd.Timestamp.now()}\n"]
    if last_model is not None:
        # coldpatch model on fold A
        split = make_fold(train, C.FOLDS[0], profile, seed=C.SEED)
        m, model, va_use, pred, split = eval_c(
            split, static, weather, cold_patch=accept_patch, return_model=True
        )
        gain = gain_table(model, top=30)
        lines8.append("\n## Top-30 LightGBM gain (Fold A, C model)\n")
        lines8.append("| feature | gain |\n| --- | --- |\n")
        for _, r in gain.iterrows():
            lines8.append(f"| {r['feature']} | {r['gain']:.1f} |\n")
        gain.to_csv(C.REPORTS_DIR / "08_feature_gain.csv", index=False)

        resid = np.log1p(va_use[C.TARGET].clip(lower=0).to_numpy()) - np.log1p(
            np.clip(pred, 0, None)
        )
        cold = split.is_cold.to_numpy()
        va = va_use.copy()
        va["_resid"] = resid
        va["_cold"] = cold
        va["_sq"] = resid**2
        lines8.append("\n## Cold residual by month (Fold A)\n")
        lines8.append("| month | n_cold | rmsle_contrib_sqrt_mean_sq |\n| --- | --- | --- |\n")
        for month, g in va.loc[va["_cold"]].groupby("month"):
            lines8.append(
                f"| {int(month)} | {len(g)} | {np.sqrt(g['_sq'].mean()):.4f} |\n"
            )
        if "is_holiday" in va.columns:
            lines8.append("\n## Cold residual holiday vs not\n")
            for flag, g in va.loc[va["_cold"]].groupby("is_holiday"):
                lines8.append(
                    f"- holiday={int(flag)}: n={len(g)} sqrtMSLE={np.sqrt(g['_sq'].mean()):.4f}\n"
                )
        if "cdd" in va.columns:
            lines8.append("\n## Cold residual by CDD tercile\n")
            cold_df = va.loc[va["_cold"]].copy()
            cold_df["cdd_bin"] = pd.qcut(cold_df["cdd"], 3, duplicates="drop")
            for b, g in cold_df.groupby("cdd_bin", observed=True):
                lines8.append(f"- CDD {b}: n={len(g)} sqrtMSLE={np.sqrt(g['_sq'].mean()):.4f}\n")

    (C.REPORTS_DIR / "08_importance.md").write_text("".join(lines8), encoding="utf-8")
    print("yazildi reports/08_importance.md")

    # karar ozeti
    decision = {
        "base_c_blend": float(base_c),
        "patch_c_blend": float(patch_c),
        "base_c_cold": float(base_cold),
        "patch_c_cold": float(patch_cold),
        "accept_patch": bool(accept_patch),
        "submit_ok": bool(accept_patch and (base_c - patch_c) >= 0.01),
    }
    # plan: submit if improved C blend is >=0.01 better than weather CV baseline
    decision["submit_ok"] = bool(
        accept_patch and (base_c - patch_c) >= 0.01 and patch_cold <= base_cold + 0.01
    )
    # if patch not accepted, no v2 submit
    if not accept_patch:
        decision["submit_ok"] = False
        decision["reason"] = "cold patch did not improve enough"
    else:
        decision["reason"] = "use coldpatch C" if decision["submit_ok"] else "patch ok but <0.01 blend gain"

    import json

    (C.REPORTS_DIR / "09_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print("DECISION", decision)


if __name__ == "__main__":
    main()
