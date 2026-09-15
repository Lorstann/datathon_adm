"""Guc elastikiyeti: z = log1p(y) - b*log(guc), b veriden kestirilir.

Kodun her yerinde b=1 varsayiliyordu. Veri reddediyor: b = 1.1596 +- 0.0176
(t=9.1, n=5.046 trafo). Sonucu cold seviyesinde monoton bir sapma ve bu sapma
sekil modelinden SONRA da duruyor -- cold artiklari (gercek - tahmin):

    <100      -0.332
    250-400   +0.161
    400-630   +0.207
    1000-1600 +0.233

Yani kucuk trafolar fazla, buyukler eksik tahmin ediliyor. b origin'ler arasi
kararli: 1.2165 / 1.2030 / 1.1695.

Kullanim: python scripts/45_elasticity.py
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
from src.validation.folds import build_cold_profile, guc_band
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

_cv = import_module("20_multiorigin_cv")
GATE_K = 0.75
VARIANTS = (("b1_mevcut", False), ("b_tahminli", True))


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    records = []
    lines = ["# Guc elastikiyeti duzeltmesi\n", f"Uretim: {pd.Timestamp.now()}\n"]

    for fold in C.FOLDS:
        split = make_fold(train, fold, profile, seed=C.SEED)
        p_table = dead_probability(split.train, fold.origin)
        lines.append(f"\n## Fold {fold.name}\n\n| variant | warm | cold | blend |\n")
        lines.append("| --- | --- | --- | --- |\n")

        for name, use_e in VARIANTS:
            t0 = time.time()
            tr_f = build_training_frame(
                split.train, end_cap=fold.origin, static=static, weather=weather,
                max_rows=_cv.MAX_ROWS, use_elasticity=use_e,
            )
            tr_f = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED,
                             entry_offsets=profile.entry_offsets)
            va_f = _cv.valid_frame(split, static, weather, use_elasticity=use_e)
            va_f = va_f.loc[va_f[C.TARGET].notna()]

            y = va_f[C.TARGET].to_numpy()
            seg = split.segment.reindex(va_f.index)
            cold_s = split.is_cold.reindex(va_f.index)
            cold = cold_s.to_numpy()
            p_row = va_f[C.LOCATION].map(p_table["by_lokasyon"]).fillna(p_table["global"])
            p_row = np.where(cold, p_row.to_numpy(), 0.0)

            ytr = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
            Xtr, cat = model_matrix(tr_f)
            Xva, _ = model_matrix(va_f)
            Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
            Xva = Xva[Xtr.columns]
            hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]
            m = fit_shape_model(Xtr, ytr, tr_f[LEVEL_COL].to_numpy(), cat, hist_cols,
                                seed=C.SEED + 3, n_trees=600)
            pred = shrink_level(
                predict_level_shape(m, Xva, va_f[LEVEL_COL].to_numpy()), p_row,
                strength=GATE_K,
            )
            met = segment_report(y, pred, seg, is_cold=cold_s)
            records.append({"model": name, "fold": fold.name, **met})
            lines.append("| {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                name, met["rmsle_warm"], met["rmsle_cold"], met["rmsle_blend"]))
            print(fold.name, name, round(met["rmsle_blend"], 4),
                  "cold", round(met["rmsle_cold"], 4), f"{time.time()-t0:.0f}s", flush=True)

            # Sapma gercekten kapandi mi: cold artigi guc bandina gore
            r = np.log1p(y) - np.log1p(np.clip(pred, 0, None))
            gb = guc_band(va_f[C.POWER]).astype(str).to_numpy()
            sel = cold & (y > 0)
            if sel.any():
                bias = pd.Series(r[sel]).groupby(gb[sel]).mean().round(3)
                lines.append(f"\ncold artik ({name}): {bias.to_dict()}\n\n")
            del tr_f, va_f, Xtr, Xva

    df = pd.DataFrame(records)
    lines.append("\n## Ozet\n\n| variant | mean | std | warm | cold |\n")
    lines.append("| --- | --- | --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            model, s["mean"], s["std"], g["rmsle_warm"].mean(), g["rmsle_cold"].mean()))
    piv = df.pivot(index="fold", columns="model", values="rmsle_blend")
    pc = df.pivot(index="fold", columns="model", values="rmsle_cold")
    lines.append(f"\nblend farki: {(piv['b_tahminli'] - piv['b1_mevcut']).round(4).to_dict()}\n")
    lines.append(f"\ncold farki: {(pc['b_tahminli'] - pc['b1_mevcut']).round(4).to_dict()}\n")
    (C.REPORTS_DIR / "45_elasticity.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().to_string())


if __name__ == "__main__":
    main()
