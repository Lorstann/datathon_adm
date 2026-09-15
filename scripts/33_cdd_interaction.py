"""`cdd_x_slope` ve `hdd_x_slope` ayri ayri.

`reports/31_features_v3.md` ikisini birlikte olcup reddetmisti (fold A iyi,
B/C kotu). Ama doz-tepki taramasi mekanizmayi gosterdi: fold A'da kazanc ayla
ve CDD ile MONOTON artiyor (Nis +1003, May -1524, Haz -2595, Tem -3253;
CDD dortte birlerine gore -866 / -2375 / -3128). Kis fold'larinda CDD ~0,
yani `cdd_x_slope` orada sabit-sifir bir kolon: sinyal yok, yalnizca seyreltme.

Bu deney iki carpanin hangisinin ise yaradigini ayirir. Onemli, cunku HDD kis
fold'larinda AKTIF ve yine de zarar verdi -- yani "surucu aktifse yardim eder"
aciklamasi HDD icin gecerli degil ve ikisi ayni sepette tutulamaz.

Gonderim ufku Nis-Tem 2026, yani CDD rejimi.

Kullanim: python scripts/33_cdd_interaction.py
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
from src.models.level_shape import fit_shape_model, predict_level_shape
from src.models.multiorigin import LEVEL_COL, build_training_frame, mask_cold
from src.models.zero_gate import dead_probability, shrink_level
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

_cv = import_module("20_multiorigin_cv")

GATE_K = 0.75
V3_EXTRA = [
    "holiday_x_sens",
    "weekend_x_drop",
    "hist_holiday_sens",
    "hist_days_since_zero",
    "hist_max_zero_run",
    "hist_month_std",
    "hist_lok_mom",
    "hist_mom_vs_lok",
]
# v3'un geri kalani (takvim/sifir/akran) her varyantta DUSURULUR; burada
# yalnizca hava carpanlari sorgulaniyor.
VARIANTS = [
    ("v2_base", V3_EXTRA + ["cdd_x_slope", "hdd_x_slope"]),
    ("cdd_only", V3_EXTRA + ["hdd_x_slope"]),
    ("hdd_only", V3_EXTRA + ["cdd_x_slope"]),
    ("cdd_hdd", V3_EXTRA),
]


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    records = []
    lines = ["# CDD / HDD carpanlari ayri ayri\n", f"Uretim: {pd.Timestamp.now()}\n"]

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
        tr_f = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED)
        va_f = _cv.valid_frame(split, static, weather)
        va_f = va_f.loc[va_f[C.TARGET].notna()]

        y = va_f[C.TARGET].to_numpy()
        seg = split.segment.reindex(va_f.index)
        cold_s = split.is_cold.reindex(va_f.index)
        cold = cold_s.to_numpy()
        p_row = va_f[C.LOCATION].map(p_table["by_lokasyon"]).fillna(p_table["global"])
        p_row = np.where(cold, p_row.to_numpy(), 0.0)

        ytr = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
        lvl_tr = tr_f[LEVEL_COL].to_numpy()
        lvl_va = va_f[LEVEL_COL].to_numpy()
        Xtr_full, cat = model_matrix(tr_f)
        Xva_full, _ = model_matrix(va_f)

        cdd_mean = float(va_f["cdd"].mean()) if "cdd" in va_f.columns else float("nan")
        lines.append(f"\n## Fold {fold.name}  (ufuk ortalama CDD {cdd_mean:.2f})\n\n")
        lines.append("| variant | warm | cold | blend |\n| --- | --- | --- | --- |\n")

        for name, drop_cols in VARIANTS:
            t0 = time.time()
            Xtr = Xtr_full.drop(columns=[c for c in drop_cols if c in Xtr_full.columns])
            Xva = Xva_full.drop(columns=[c for c in drop_cols if c in Xva_full.columns])
            Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
            Xva = Xva[Xtr.columns]
            hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]
            m = fit_shape_model(
                Xtr, ytr, lvl_tr, cat, hist_cols, seed=C.SEED + 3, n_trees=600
            )
            pred = shrink_level(
                predict_level_shape(m, Xva, lvl_va), p_row, strength=GATE_K
            )
            met = segment_report(y, pred, seg, is_cold=cold_s)
            records.append({"model": name, "fold": fold.name, **met})
            lines.append(
                "| {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                    name, met["rmsle_warm"], met["rmsle_cold"], met["rmsle_blend"]
                )
            )
            print(fold.name, name, round(met["rmsle_blend"], 4), f"{time.time()-t0:.0f}s", flush=True)
        del tr_f, va_f, Xtr_full, Xva_full

    df = pd.DataFrame(records)
    lines.append("\n## Ozet (rmsle_blend)\n\n| variant | mean | std | warm |\n")
    lines.append("| --- | --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                model, s["mean"], s["std"], g["rmsle_warm"].mean()
            )
        )
    piv = df.pivot(index="fold", columns="model", values="rmsle_blend")
    lines.append("\n### Fold bazinda v2_base farki (negatif = iyi)\n\n")
    for col in piv.columns:
        lines.append(f"- {col}: {(piv[col]-piv['v2_base']).round(4).to_dict()}\n")

    (C.REPORTS_DIR / "33_cdd_interaction.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().sort_values().to_string())


if __name__ == "__main__":
    main()
