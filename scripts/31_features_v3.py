"""Ozellik paketi v3 ablasyonu.

v2 gonderimde (LB 1.06899). v3'un fikri: statik katsayilari ufuk boyunca
degisen degiskenlerle CARPARAK satir duzeyine indirmek.

  - `cdd_x_slope` / `hdd_x_slope`: trafoya ozgu sicaklik katsayisi x o gunun
    derece-gunu. Ufuk Nis-Tem, yani sogutma rampasinin ustu.
  - `holiday_x_sens` / `weekend_x_drop`: trafonun kendi tatil / hafta sonu
    tepkisi x gunun takvim durumu. Ufukta Kurban (2026-05-27) var.
  - sifir blogu geometrisi (`hist_days_since_zero`, `hist_max_zero_run`):
    warm + sifir dilimi kare hatanin %5-18'i.
  - `hist_month_std`: seviyenin guvenilirligi.
  - `hist_lok_mom` / `hist_mom_vs_lok`: akran momentumu.

Kullanim: python scripts/31_features_v3.py
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

WEATHER_INT = ["cdd_x_slope", "hdd_x_slope"]
CAL_INT = ["holiday_x_sens", "weekend_x_drop", "hist_holiday_sens"]
ZERO_GEO = ["hist_days_since_zero", "hist_max_zero_run"]
STABILITY = ["hist_month_std"]
PEER_MOM = ["hist_lok_mom", "hist_mom_vs_lok"]
V3_ALL = WEATHER_INT + CAL_INT + ZERO_GEO + STABILITY + PEER_MOM

VARIANTS = [
    ("v2_base", V3_ALL),                      # v3 kolonlarinin tamami dusurulur
    ("v3_weather_only", CAL_INT + ZERO_GEO + STABILITY + PEER_MOM),
    ("v3_all", []),
]


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    records = []
    lines = ["# Ozellik paketi v3\n", f"Uretim: {pd.Timestamp.now()}\n"]
    lines.append(f"\nHepsine cold olu-trafo kapisi k={GATE_K}. v2_base = gonderimdeki yapilandirma.\n")

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

        lines.append(f"\n## Fold {fold.name}\n\n")
        lines.append("| variant | ozellik | warm | cold | blend |\n")
        lines.append("| --- | --- | --- | --- | --- |\n")

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
                "| {} | {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                    name, len(Xtr.columns), met["rmsle_warm"], met["rmsle_cold"],
                    met["rmsle_blend"],
                )
            )
            print(fold.name, name, round(met["rmsle_blend"], 4), f"{time.time()-t0:.0f}s", flush=True)
        del tr_f, va_f, Xtr_full, Xva_full

    df = pd.DataFrame(records)
    lines.append("\n## Ozet (rmsle_blend)\n\n| variant | mean | std | warm | cold |\n")
    lines.append("| --- | --- | --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                model, s["mean"], s["std"], g["rmsle_warm"].mean(), g["rmsle_cold"].mean()
            )
        )
    piv = df.pivot(index="fold", columns="model", values="rmsle_blend")
    lines.append("\n### Fold bazinda v2_base farki (negatif = iyi)\n\n")
    for col in piv.columns:
        lines.append(f"- {col}: {(piv[col]-piv['v2_base']).round(4).to_dict()}\n")
    lines.append(f"\nEn iyi: **{df.groupby('model')['rmsle_blend'].mean().idxmin()}**\n")

    (C.REPORTS_DIR / "31_features_v3.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().sort_values().to_string())


if __name__ == "__main__":
    main()
