"""Origin yogunlugu + LightGBM kapasitesi, duzeltilmis metrik uzerinde.

`reports/19_c_tune.md` hiperparametre taramasi eski (kirli) metrikle ve eski
egitim cercevesiyle yapilmisti; sonuclari tasinmaz. Burada cok-origin
cercevesinde yeniden olculuyor. Cold olu-trafo kapisi (k=0.75) her varyanta
uygulanir, cunku karar kuralinin gonderim yapilandirmasiyla ayni olmasi lazim.

Fold A disarida: tek kullanilabilir origin, egitim cercevesi 88k satir.

Kullanim: python scripts/25_capacity.py
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
from src.models.multiorigin import (
    LEVEL_COL,
    build_training_frame,
    mask_cold,
    training_origins,
)
from src.models.zero_gate import dead_probability, shrink_level
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

_cv = import_module("20_multiorigin_cv")

GATE_K = 0.75

# (ad, origin araligi, max origin, max satir, agac, lgbm override)
VARIANTS = [
    ("base_45d_l63_t600", 45, 8, 1_200_000, 600, {}),
    ("dense_30d_l63_t600", 30, 12, 1_800_000, 600, {}),
    ("dense_30d_l127_t900", 30, 12, 1_800_000, 900, {"num_leaves": 127, "min_child_samples": 40}),
    ("dense_30d_l63_t1200_lr03", 30, 12, 1_800_000, 1200, {"learning_rate": 0.03}),
    ("dense_21d_l63_t600", 21, 16, 2_400_000, 600, {}),
]


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    records = []
    lines = ["# Origin yogunlugu + kapasite\n", f"Uretim: {pd.Timestamp.now()}\n"]
    lines.append(f"\nHer varyanta cold olu-trafo kapisi k={GATE_K} uygulanmistir.\n")

    for fold in [f for f in C.FOLDS if f.name != "A_mevsim"]:
        split = make_fold(train, fold, profile, seed=C.SEED)
        va_f = _cv.valid_frame(split, static, weather)
        va_f = va_f.loc[va_f[C.TARGET].notna()]
        y = va_f[C.TARGET].to_numpy()
        seg = split.segment.reindex(va_f.index)
        cold_s = split.is_cold.reindex(va_f.index)
        cold = cold_s.to_numpy()

        p_table = dead_probability(split.train, fold.origin)
        p_row = va_f[C.LOCATION].map(p_table["by_lokasyon"]).fillna(p_table["global"])
        p_row = np.where(cold, p_row.to_numpy(), 0.0)

        Xva_full, _ = model_matrix(va_f)
        lvl_va = va_f[LEVEL_COL].to_numpy()

        lines.append(f"\n## Fold {fold.name}\n\n")
        lines.append("| variant | egitim satiri | warm | cold | blend | s |\n")
        lines.append("| --- | --- | --- | --- | --- | --- |\n")

        for name, spacing, max_o, max_rows, n_trees, override in VARIANTS:
            t0 = time.time()
            origins = training_origins(
                fold.origin, spacing_days=spacing, max_origins=max_o
            )
            tr_f = build_training_frame(
                split.train,
                end_cap=fold.origin,
                static=static,
                weather=weather,
                origins=origins,
                max_rows=max_rows,
            )
            tr_f = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED)
            ytr = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
            lvl_tr = tr_f[LEVEL_COL].to_numpy()

            Xtr, cat = model_matrix(tr_f)
            Xtr, Xva = Xtr.align(Xva_full, join="outer", axis=1, fill_value=np.nan)
            Xva = Xva[Xtr.columns]
            hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]

            m = fit_shape_model(
                Xtr,
                ytr,
                lvl_tr,
                cat,
                hist_cols,
                seed=C.SEED + 3,
                n_trees=n_trees,
                params_override=override or None,
            )
            pred = predict_level_shape(m, Xva, lvl_va)
            pred = shrink_level(pred, p_row, strength=GATE_K)
            met = segment_report(y, pred, seg, is_cold=cold_s)
            records.append({"model": name, "fold": fold.name, **met})
            lines.append(
                "| {} | {:,} | {:.4f} | {:.4f} | {:.4f} | {:.0f} |\n".format(
                    name,
                    len(tr_f),
                    met["rmsle_warm"],
                    met["rmsle_cold"],
                    met["rmsle_blend"],
                    time.time() - t0,
                )
            )
            print(fold.name, name, round(met["rmsle_blend"], 4), flush=True)
            del tr_f, Xtr, Xva

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
    base = piv["base_45d_l63_t600"]
    lines.append("\n### Fold bazinda base farki (negatif = iyi)\n\n")
    for col in piv.columns:
        lines.append(f"- {col}: {(piv[col]-base).round(4).to_dict()}\n")
    best = df.groupby("model")["rmsle_blend"].mean().idxmin()
    lines.append(f"\nEn iyi: **{best}**\n")

    (C.REPORTS_DIR / "25_capacity.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().sort_values().to_string())


if __name__ == "__main__":
    main()
