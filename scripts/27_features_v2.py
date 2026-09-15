"""Ozellik paketi v2 + harmanlanmis seviye ablasyonu.

Hedef `reports/23_error_decomp.md`'nin en buyuk ele alinabilir dilimi:
warm + sifir-olmayan satirlar (kare hatanin %24-31'i).

Yeni ozellikler ve gerekceleri:
  - `hist_mom_*`  : ortalamaya donus. corr(seviye kaymasi, m7-m91) = -0.31,
                    18.198 trafo-ufuk cifti uzerinde olculdu.
  - `hist_yoy_gap`: ayni takvim penceresinin gecen yilki hali, varligin genel
                    seviyesine gore. Gonderim origin'inde trafolarin %54'unde
                    dolu ve ufuk (Nis-Tem) tam gecen yilin karsiligi.
  - `hist_q*_91`  : son ceyregin ceyreklikleri; ortalamanin tasiyamadigi
                    saglam olcek.
  - `hist_dow_rel_*`: hafta gunu profili varligin kendi seviyesine gore
                    (sanayi/konut imzasi).
  - `hist_*_slope`: trafo basina sicaklik duyarliligi. Ufuk sogutma rampasi;
                    ayni lokasyondaki iki trafo ayni havayi gorur, tepkileri
                    farklidir.
  - `hist_rel_peer`: akranlara gore konum.

Seviye: tek pencere yerine 0.7*m7 + 0.3*m91 harmani (w taramasi 0.3'te dip).

Kullanim: python scripts/27_features_v2.py
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
from src.models.level_shape import WARM_LEVEL_BLEND, fit_shape_model, predict_level_shape
from src.models.multiorigin import LEVEL_COL, build_training_frame, mask_cold
from src.models.zero_gate import dead_probability, shrink_level
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

_cv = import_module("20_multiorigin_cv")

GATE_K = 0.75

NEW_FEATURES = (
    [f"hist_mom_{a}_{b}" for a, b in (("7", "28"), ("7", "91"), ("28", "91"), ("7", "365"))]
    + ["hist_yoy_gap", "hist_q25_91", "hist_q75_91", "hist_iqr_91"]
    + [f"hist_dow_rel_{k}" for k in range(7)]
    + ["hist_cdd_slope", "hist_hdd_slope", "hist_rel_peer"]
)

# (ad, yeni ozellikler acik mi, seviye harmani)
VARIANTS = [
    ("base", False, None),
    ("blend_level", False, WARM_LEVEL_BLEND),
    ("feat_v2", True, None),
    ("feat_v2_blend", True, WARM_LEVEL_BLEND),
]


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    records = []
    lines = ["# Ozellik paketi v2 + harmanlanmis seviye\n", f"Uretim: {pd.Timestamp.now()}\n"]
    lines.append(f"\nHepsine cold olu-trafo kapisi k={GATE_K} uygulanmistir.\n")

    for fold in C.FOLDS:
        split = make_fold(train, fold, profile, seed=C.SEED)
        p_table = dead_probability(split.train, fold.origin)

        lines.append(f"\n## Fold {fold.name}\n\n")
        lines.append("| variant | warm | cold | blend | s |\n")
        lines.append("| --- | --- | --- | --- | --- |\n")

        for name, use_new, blend in VARIANTS:
            t0 = time.time()
            tr_f = build_training_frame(
                split.train,
                end_cap=fold.origin,
                static=static,
                weather=weather,
                max_rows=_cv.MAX_ROWS,
                blend_weights=blend,
            )
            tr_f = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED)
            va_f = _cv.valid_frame(split, static, weather, blend_weights=blend)
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

            Xtr, cat = model_matrix(tr_f)
            Xva, _ = model_matrix(va_f)
            if not use_new:
                drop = [c for c in NEW_FEATURES if c in Xtr.columns]
                Xtr = Xtr.drop(columns=drop)
                Xva = Xva.drop(columns=[c for c in drop if c in Xva.columns])
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
                "| {} | {:.4f} | {:.4f} | {:.4f} | {:.0f} |\n".format(
                    name,
                    met["rmsle_warm"],
                    met["rmsle_cold"],
                    met["rmsle_blend"],
                    time.time() - t0,
                )
            )
            print(fold.name, name, round(met["rmsle_blend"], 4), flush=True)
            del tr_f, va_f, Xtr, Xva

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
    lines.append("\n### Fold bazinda base farki (negatif = iyi)\n\n")
    for col in piv.columns:
        lines.append(f"- {col}: {(piv[col]-piv['base']).round(4).to_dict()}\n")
    lines.append(f"\nEn iyi: **{df.groupby('model')['rmsle_blend'].mean().idxmin()}**\n")

    (C.REPORTS_DIR / "27_features_v2.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().sort_values().to_string())


if __name__ == "__main__":
    main()
