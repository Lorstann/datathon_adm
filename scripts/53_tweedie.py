"""Tweedie hedef, ham olcekte: log1p+L2 seviye/sekil modeline cesitlilik uyesi.

`docs/prior-work.md` 8.3/9.2: Tweedie RMSLE'yi degil ham olcek deviance'ini
optimize eder, yani ana model DEGIL -- yalniz RMSLE'de degerlendirilen bir
ensemble uyesi olarak test edilmeli. Onceki tur (round4) CatBoost/XGBoost
cesitliligini denedi ve agac-agac korelasyonu 0.86-0.93 cikip kazanc
getirmedi (`reports/44_round4_summary.md`); Tweedie farkli bir kayip
fonksiyonu oldugu icin hata yapisi gercekten farkli olabilir.

Kullanim: python scripts/53_tweedie.py
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
from src.models.boosting import fit_lightgbm
from src.models.level_shape import fit_shape_model, predict_level_shape
from src.models.multiorigin import LEVEL_COL, build_training_frame, mask_cold
from src.models.zero_gate import dead_probability, shrink_level
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

_cv = import_module("20_multiorigin_cv")
GATE_K = 0.75
SEED = C.SEED + 3
N_TREES_SHAPE = 600
N_TREES_TW = 600
TWEEDIE_POWER = 1.2
BLEND_WEIGHTS = [0.0, 0.15, 0.3, 0.5, 1.0]  # tweedie agirligi


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    records = []
    lines = [
        "# Tweedie ham-olcek cesitlilik uyesi\n",
        f"Uretim: {pd.Timestamp.now()}\n",
        f"tweedie_variance_power={TWEEDIE_POWER}, n_trees={N_TREES_TW}\n",
    ]

    for fold in C.FOLDS:
        t0 = time.time()
        split = make_fold(train, fold, profile, seed=C.SEED)
        p_table = dead_probability(split.train, fold.origin)
        tr_f = build_training_frame(
            split.train, end_cap=fold.origin, static=static, weather=weather,
            max_rows=_cv.MAX_ROWS,
        )
        tr_f = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED,
                         entry_offsets=profile.entry_offsets)
        va_f = _cv.valid_frame(split, static, weather)
        va_f = va_f.loc[va_f[C.TARGET].notna()]

        y = va_f[C.TARGET].to_numpy()
        seg = split.segment.reindex(va_f.index)
        cold_s = split.is_cold.reindex(va_f.index)
        p_row = va_f[C.LOCATION].map(p_table["by_lokasyon"]).fillna(p_table["global"])
        p_row = np.where(cold_s.to_numpy(), p_row.to_numpy(), 0.0)

        ytr_log = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
        ytr_raw = tr_f[C.TARGET].clip(lower=0).to_numpy()
        lvl_tr, lvl_va = tr_f[LEVEL_COL].to_numpy(), va_f[LEVEL_COL].to_numpy()
        Xtr, cat = model_matrix(tr_f)
        Xva, _ = model_matrix(va_f)
        Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
        Xva = Xva[Xtr.columns]
        hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]

        # 1. mevcut: log1p+L2 seviye/sekil modeli (baseline)
        shape_m = fit_shape_model(Xtr, ytr_log, lvl_tr, cat, hist_cols,
                                   seed=SEED, n_trees=N_TREES_SHAPE)
        pred_shape = predict_level_shape(shape_m, Xva, lvl_va)

        # 2. tweedie: ham hedef, ham olcekte tahmin
        tw_m = fit_lightgbm(
            Xtr, ytr_raw, cat, seed=SEED, n_trees=N_TREES_TW,
            mask_history_rate=0.0, history_cols=hist_cols,
            params_override={
                "objective": "tweedie",
                "tweedie_variance_power": TWEEDIE_POWER,
                "metric": "rmse",
            },
        )
        pred_tw = np.clip(tw_m.booster.predict(Xva[tw_m.feature_names]), 0.0, None)

        lines.append(f"\n## Fold {fold.name} ({time.time()-t0:.0f}s)\n\n")
        lines.append("| tweedie agirligi | warm | cold | blend |\n| --- | --- | --- | --- |\n")
        for w in BLEND_WEIGHTS:
            pred = (1 - w) * pred_shape + w * pred_tw
            pred = shrink_level(pred, p_row, strength=GATE_K)
            met = segment_report(y, pred, seg, is_cold=cold_s)
            records.append({"model": f"tw_w{w}", "fold": fold.name, **met})
            lines.append("| {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                w, met["rmsle_warm"], met["rmsle_cold"], met["rmsle_blend"]))
            print(fold.name, f"w={w}", round(met["rmsle_blend"], 4), flush=True)
        del tr_f, va_f, Xtr, Xva

    df = pd.DataFrame(records)
    lines.append("\n## Ozet (rmsle_blend)\n\n| tweedie agirligi | mean | std |\n| --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append(f"| {model} | {s['mean']:.4f} | {s['std']:.4f} |\n")
    piv = df.pivot(index="fold", columns="model", values="rmsle_blend")
    base = piv["tw_w0.0"]
    for c in piv.columns:
        lines.append(f"\n- {c}: {(piv[c] - base).round(4).to_dict()}")
    (C.REPORTS_DIR / "53_tweedie.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().to_string())


if __name__ == "__main__":
    main()
