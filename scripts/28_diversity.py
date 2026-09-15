"""Model cesitliligi: LightGBM + CatBoost + XGBoost, ayni seviye+sekil cercevesinde.

Harman log1p uzayinda alinir: metrik orada L2, ham olcekte ortalamak yanlis
model birlestirir.

`reports/05_models.md`'deki eski ensemble sayilari tasinmaz; orada hem metrik
kirliydi hem egitim cercevesi tek-adimdi.

Seviye zinciri kullanilir, harman DEGIL: harman fold C'de isareti degistiriyor
(reports/27_features_v2.md).

Kullanim: python scripts/28_diversity.py
"""

from __future__ import annotations

import sys
import time
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import build_panel, entity_static, load_test, load_train
from src.external.weather import CACHE_PATH, load_weather, weather_features
from src.features.build import model_matrix
from src.models.boosting import fit_catboost, fit_xgboost
from src.models.level_shape import fit_shape_model, predict_level_shape
from src.models.multiorigin import LEVEL_COL, build_training_frame, mask_cold
from src.models.zero_gate import dead_probability, shrink_level
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

_cv = import_module("20_multiorigin_cv")

GATE_K = 0.75
MAX_ROWS = 900_000  # CatBoost/XGBoost bellek icin LightGBM'den dusuk


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    records = []
    lines = ["# Model cesitliligi (cok-origin cercevesi)\n", f"Uretim: {pd.Timestamp.now()}\n"]

    for fold in C.FOLDS:
        split = make_fold(train, fold, profile, seed=C.SEED)
        tr_f = build_training_frame(
            split.train,
            end_cap=fold.origin,
            static=static,
            weather=weather,
            max_rows=MAX_ROWS,
        )
        tr_f = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED)
        va_f = _cv.valid_frame(split, static, weather)
        va_f = va_f.loc[va_f[C.TARGET].notna()]

        y = va_f[C.TARGET].to_numpy()
        seg = split.segment.reindex(va_f.index)
        cold_s = split.is_cold.reindex(va_f.index)
        cold = cold_s.to_numpy()
        p_table = dead_probability(split.train, fold.origin)
        p_row = va_f[C.LOCATION].map(p_table["by_lokasyon"]).fillna(p_table["global"])
        p_row = np.where(cold, p_row.to_numpy(), 0.0)

        ytr = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
        lvl_tr = tr_f[LEVEL_COL].to_numpy()
        lvl_va = va_f[LEVEL_COL].to_numpy()
        resid_tr = ytr - lvl_tr

        Xtr, cat = model_matrix(tr_f)
        Xva, _ = model_matrix(va_f)
        Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
        Xva = Xva[Xtr.columns]
        hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]

        preds = {}
        t0 = time.time()
        m = fit_shape_model(Xtr, ytr, lvl_tr, cat, hist_cols, seed=C.SEED + 3, n_trees=600)
        preds["lgbm"] = predict_level_shape(m, Xva, lvl_va)
        print(fold.name, "lgbm", f"{time.time()-t0:.0f}s", flush=True)

        t0 = time.time()
        mc = fit_catboost(Xtr, resid_tr, cat, seed=C.SEED, n_trees=600)
        preds["catboost"] = np.clip(
            np.expm1(_resid_pred_cat(mc, Xva, lvl_va)), 0.0, None
        )
        print(fold.name, "catboost", f"{time.time()-t0:.0f}s", flush=True)

        t0 = time.time()
        mx = fit_xgboost(Xtr, resid_tr, cat, seed=C.SEED, n_trees=600)
        preds["xgboost"] = np.clip(np.expm1(_resid_pred_xgb(mx, Xva, lvl_va)), 0.0, None)
        print(fold.name, "xgboost", f"{time.time()-t0:.0f}s", flush=True)

        lp = {k: np.log1p(v) for k, v in preds.items()}
        preds["lgb_cat"] = np.expm1(0.6 * lp["lgbm"] + 0.4 * lp["catboost"])
        preds["lgb_cat_xgb"] = np.expm1(
            0.5 * lp["lgbm"] + 0.3 * lp["catboost"] + 0.2 * lp["xgboost"]
        )

        lines.append(f"\n## Fold {fold.name}  (egitim satiri {len(tr_f):,})\n\n")
        lines.append("| model | warm | cold | blend |\n| --- | --- | --- | --- |\n")
        for name, p in preds.items():
            p = shrink_level(np.clip(p, 0, None), p_row, strength=GATE_K)
            met = segment_report(y, p, seg, is_cold=cold_s)
            records.append({"model": name, "fold": fold.name, **met})
            lines.append(
                "| {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                    name, met["rmsle_warm"], met["rmsle_cold"], met["rmsle_blend"]
                )
            )
            print(" ", name, round(met["rmsle_blend"], 4), flush=True)
        del tr_f, va_f, Xtr, Xva

    df = pd.DataFrame(records)
    lines.append("\n## Ozet (rmsle_blend)\n\n| model | mean | std | warm | cold |\n")
    lines.append("| --- | --- | --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                model, s["mean"], s["std"], g["rmsle_warm"].mean(), g["rmsle_cold"].mean()
            )
        )
    piv = df.pivot(index="fold", columns="model", values="rmsle_blend")
    lines.append("\n### Fold bazinda lgbm farki (negatif = iyi)\n\n")
    for col in piv.columns:
        lines.append(f"- {col}: {(piv[col]-piv['lgbm']).round(4).to_dict()}\n")
    lines.append(f"\nEn iyi: **{df.groupby('model')['rmsle_blend'].mean().idxmin()}**\n")

    (C.REPORTS_DIR / "28_diversity.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().sort_values().to_string())


def _resid_pred_cat(model, X, level):
    Xc = X.reindex(columns=model.feature_names).copy()
    for c in model.cat_cols:
        Xc[c] = Xc[c].astype(str).fillna("__NA__")
    return np.clip(level + np.asarray(model.booster.predict(Xc), dtype="float64"), 0.0, None)


def _resid_pred_xgb(model, X, level):
    import xgboost as xgb

    Xn = X.reindex(columns=model.feature_names)
    r = model.booster.predict(xgb.DMatrix(Xn))
    return np.clip(level + np.asarray(r, dtype="float64"), 0.0, None)


if __name__ == "__main__":
    main()
