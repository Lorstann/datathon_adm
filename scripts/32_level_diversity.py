"""Cesitliligi ogrenicide degil SEVIYEDE aramak + XGBoost rovansi.

`reports/28_diversity.md`: CatBoost/XGBoost harmani LightGBM'i gecemedi. Ama
bu projede hassas olan bilesen ogrenici degil **seviye**: zincir ablasyonunda
m7 / m14 / m21 / genisleyen arasinda 0.04 blend farki vardi, harmanlanmis
seviye ise fold C'de isaret degistirdi. Yani seviye secimi hem etkili hem
kararsiz - tam olarak ortalamanin yardim ettigi durum.

Burada ayni ogrenici iki farkli seviye capasiyla egitilip log1p uzayinda
ortalaniyor. Ayrica XGBoost, kategorikleri dusurmeyen surumuyle
(`fit_xgboost(..., cat_cols)`) tekrar yarisiyor.

Kullanim: python scripts/32_level_diversity.py
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
from src.features.history import entity_history_features
from src.models.boosting import fit_xgboost
from src.models.level_shape import entity_level, fit_shape_model, predict_level_shape
from src.models.multiorigin import LEVEL_COL, build_training_frame, mask_cold
from src.models.zero_gate import dead_probability, shrink_level
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

_cv = import_module("20_multiorigin_cv")

GATE_K = 0.75
CHAINS = {
    "m7": ("hist_mean_7", "hist_mean_28", "hist_mean_91", "hist_mean"),
    "m28": ("hist_mean_28", "hist_mean_91", "hist_mean_7", "hist_mean"),
}


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    records = []
    lines = ["# Seviye cesitliligi + XGBoost rovansi\n", f"Uretim: {pd.Timestamp.now()}\n"]

    for fold in C.FOLDS:
        split = make_fold(train, fold, profile, seed=C.SEED)
        p_table = dead_probability(split.train, fold.origin)

        tr_f = build_training_frame(
            split.train,
            end_cap=fold.origin,
            static=static,
            weather=weather,
            max_rows=_cv.MAX_ROWS,
        )
        tr_masked = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED)
        is_masked = (
            tr_masked["hist_n"].isna().to_numpy() & tr_f["hist_n"].notna().to_numpy()
        )
        cold_lvl_tr = tr_f["_cold_level"].to_numpy()
        va_f = _cv.valid_frame(split, static, weather)
        va_f = va_f.loc[va_f[C.TARGET].notna()]
        hist_va = entity_history_features(split.train, fold.origin)
        hist_by_origin = {
            o: entity_history_features(split.train, o) for o in tr_f["_origin"].unique()
        }

        y = va_f[C.TARGET].to_numpy()
        seg = split.segment.reindex(va_f.index)
        cold_s = split.is_cold.reindex(va_f.index)
        cold = cold_s.to_numpy()
        p_row = va_f[C.LOCATION].map(p_table["by_lokasyon"]).fillna(p_table["global"])
        p_row = np.where(cold, p_row.to_numpy(), 0.0)

        ytr = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
        Xtr, cat = model_matrix(tr_masked)
        Xva, _ = model_matrix(va_f)
        Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
        Xva = Xva[Xtr.columns]
        hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]

        preds = {}
        for cname, chain in CHAINS.items():
            parts = [
                pd.Series(
                    entity_level(
                        hist_by_origin[o],
                        g[C.ENTITY],
                        g["log_guc"].to_numpy(),
                        warm_col=chain,
                    ),
                    index=g.index,
                )
                for o, g in tr_f.groupby("_origin", sort=False)
            ]
            lvl_tr = pd.concat(parts).reindex(tr_f.index).to_numpy()
            lvl_tr = np.where(is_masked, cold_lvl_tr, lvl_tr)
            lvl_va = entity_level(
                hist_va, va_f[C.ENTITY], va_f["log_guc"].to_numpy(), warm_col=chain
            )
            t0 = time.time()
            m = fit_shape_model(
                Xtr, ytr, lvl_tr, cat, hist_cols, seed=C.SEED + 3, n_trees=600
            )
            preds[f"lgbm_{cname}"] = predict_level_shape(m, Xva, lvl_va)
            print(fold.name, f"lgbm_{cname}", f"{time.time()-t0:.0f}s", flush=True)
            if cname == "m7":
                base_lvl_tr, base_lvl_va = lvl_tr, lvl_va

        t0 = time.time()
        mx = fit_xgboost(Xtr, ytr - base_lvl_tr, cat, seed=C.SEED, n_trees=600)
        import xgboost as xgb

        Xq = Xva.reindex(columns=mx.feature_names).copy()
        for c in mx.cat_cols:
            Xq[c] = Xq[c].astype("category")
        r = mx.booster.predict(xgb.DMatrix(Xq, enable_categorical=True))
        preds["xgb_cat"] = np.clip(np.expm1(base_lvl_va + r), 0.0, None)
        print(fold.name, "xgb_cat", f"{time.time()-t0:.0f}s", flush=True)

        lp = {k: np.log1p(np.clip(v, 0, None)) for k, v in preds.items()}
        preds["avg_m7_m28"] = np.expm1(0.5 * lp["lgbm_m7"] + 0.5 * lp["lgbm_m28"])
        preds["avg_m7_xgb"] = np.expm1(0.7 * lp["lgbm_m7"] + 0.3 * lp["xgb_cat"])
        preds["avg_all3"] = np.expm1(
            0.45 * lp["lgbm_m7"] + 0.3 * lp["lgbm_m28"] + 0.25 * lp["xgb_cat"]
        )

        lines.append(f"\n## Fold {fold.name}\n\n| model | warm | cold | blend |\n")
        lines.append("| --- | --- | --- | --- |\n")
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
        del tr_f, tr_masked, va_f, Xtr, Xva

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
    lines.append("\n### Fold bazinda lgbm_m7 farki (negatif = iyi)\n\n")
    for col in piv.columns:
        lines.append(f"- {col}: {(piv[col]-piv['lgbm_m7']).round(4).to_dict()}\n")
    lines.append(f"\nEn iyi: **{df.groupby('model')['rmsle_blend'].mean().idxmin()}**\n")

    (C.REPORTS_DIR / "32_level_diversity.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().sort_values().to_string())


if __name__ == "__main__":
    main()
