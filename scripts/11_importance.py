"""Fold A baseline C icin gain + cold residual dilimleri."""

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
from src.external.weather import load_weather, weather_features
from src.models.level_shape import entity_level, fit_shape_model, gain_table, predict_level_shape
from src.models.pipeline import fold_frames, mask_history_entities, xy
from src.validation.folds import build_cold_profile
from src.validation.split import make_fold


def main() -> None:
    weather = weather_features(load_weather())
    train = load_train()
    test = load_test()
    panel = build_panel(train, test)
    static = entity_static(panel)
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    fold = C.FOLDS[0]
    split = make_fold(train, fold, profile, seed=C.SEED)
    train_f, valid_f, hist = fold_frames(split, static, weather)
    Xtr, ytr, cat, logg_tr, tr_use = xy(train_f)
    Xva, _, _, logg_va, va_use = xy(valid_f, dropna_target=True)
    Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
    Xva = Xva[Xtr.columns]
    Xtr_m = mask_history_entities(Xtr, tr_use[C.ENTITY], 0.30, C.SEED)
    level_tr = entity_level(hist, tr_use[C.ENTITY], logg_tr)
    level_va = entity_level(hist, va_use[C.ENTITY], logg_va)
    hist_cols = [c for c in Xtr_m.columns if c.startswith("hist_")]
    model = fit_shape_model(Xtr_m, ytr, level_tr, cat, hist_cols, seed=C.SEED + 3, n_trees=300)
    pred = predict_level_shape(model, Xva, level_va)

    gain = gain_table(model, top=30)
    gain.to_csv(C.REPORTS_DIR / "08_feature_gain.csv", index=False)

    resid = np.log1p(va_use[C.TARGET].clip(lower=0).to_numpy()) - np.log1p(np.clip(pred, 0, None))
    cold = split.is_cold.to_numpy()
    va = va_use.copy()
    va["_sq"] = resid**2
    va["_cold"] = cold

    lines = [
        "# Feature gain and cold error slices (baseline C, with weather)\n",
        f"Uretim: {pd.Timestamp.now()}\n",
        "\n## Top-30 LightGBM gain (Fold A)\n",
        "| feature | gain |\n| --- | --- |\n",
    ]
    for _, r in gain.iterrows():
        lines.append(f"| {r['feature']} | {r['gain']:.1f} |\n")

    lines.append("\n## Cold residual by month (Fold A)\n")
    lines.append("| month | n_cold | sqrtMSLE |\n| --- | --- | --- |\n")
    for month, g in va.loc[va["_cold"]].groupby("month"):
        lines.append(f"| {int(month)} | {len(g)} | {np.sqrt(g['_sq'].mean()):.4f} |\n")

    if "is_holiday" in va.columns:
        lines.append("\n## Cold residual holiday vs not\n")
        for flag, g in va.loc[va["_cold"]].groupby("is_holiday"):
            lines.append(
                f"- holiday={int(flag)}: n={len(g)} sqrtMSLE={np.sqrt(g['_sq'].mean()):.4f}\n"
            )
    if "cdd" in va.columns:
        lines.append("\n## Cold residual by CDD tercile\n")
        cold_df = va.loc[va["_cold"]].copy()
        cold_df["cdd_bin"] = pd.qcut(cold_df["cdd"], 3, duplicates="drop")
        for b, g in cold_df.groupby("cdd_bin", observed=True):
            lines.append(f"- CDD {b}: n={len(g)} sqrtMSLE={np.sqrt(g['_sq'].mean()):.4f}\n")

    (C.REPORTS_DIR / "08_importance.md").write_text("".join(lines), encoding="utf-8")
    print("yazildi reports/08_importance.md")
    print(gain.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
