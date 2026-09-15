"""TÜİK nufus + EPİAŞ il-ay ozellikleri, DUZELTILMIS metrik uzerinde.

`reports/18_external_features.md` bunlari reddetmisti (+0.0020) ama o olcum
hem kirli `is_cold` tanimiyla hem tek-adim egitim cercevesiyle yapilmisti.
Ikisi de degisti, dolayisiyla karar tasinmaz ve yeniden olculmeli.

EPİAŞ sizinti acisindan guvenli: tahmin ayi icin lag-1 ay + YoY kullaniliyor,
ayni ayin gerceklesen degeri ozellik degil (bkz. src/external/epias.py).

Kullanim: python scripts/35_external_recheck.py
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


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    records = []
    lines = ["# Dis veri (TÜİK + EPİAŞ) yeniden olcum\n", f"Uretim: {pd.Timestamp.now()}\n"]

    for fold in C.FOLDS:
        split = make_fold(train, fold, profile, seed=C.SEED)
        p_table = dead_probability(split.train, fold.origin)
        lines.append(f"\n## Fold {fold.name}\n\n| variant | ozellik | warm | cold | blend |\n")
        lines.append("| --- | --- | --- | --- | --- |\n")

        for use_ext in (False, True):
            t0 = time.time()
            tr_f = build_training_frame(
                split.train, end_cap=fold.origin, static=static, weather=weather,
                max_rows=_cv.MAX_ROWS, use_external=use_ext,
            )
            tr_f = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED)
            va_f = _cv.valid_frame(split, static, weather, use_external=use_ext)
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
                predict_level_shape(m, Xva, va_f[LEVEL_COL].to_numpy()), p_row, strength=GATE_K
            )
            met = segment_report(y, pred, seg, is_cold=cold_s)
            name = "with_external" if use_ext else "no_external"
            records.append({"model": name, "fold": fold.name, **met})
            lines.append(
                "| {} | {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                    name, len(Xtr.columns), met["rmsle_warm"], met["rmsle_cold"], met["rmsle_blend"]
                )
            )
            print(fold.name, name, round(met["rmsle_blend"], 4), f"{time.time()-t0:.0f}s", flush=True)
            del tr_f, va_f, Xtr, Xva

    df = pd.DataFrame(records)
    lines.append("\n## Ozet\n\n| variant | mean | std |\n| --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append(f"| {model} | {s['mean']:.4f} | {s['std']:.4f} |\n")
    piv = df.pivot(index="fold", columns="model", values="rmsle_blend")
    lines.append(
        f"\nFold bazinda fark (with - no): {(piv['with_external']-piv['no_external']).round(4).to_dict()}\n"
    )
    (C.REPORTS_DIR / "35_external_recheck.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().to_string())


if __name__ == "__main__":
    main()
