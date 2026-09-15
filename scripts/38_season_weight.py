"""Mevsim-esli ornek agirliklandirma.

Gonderim ufkunun tamami Nis-Tem. Egitim cercevesindeki 8 origin ise yila
yayilmis, yani ufku Nis-Tem'e denk gelen satirlar toplamin ancak ~%25'i.
Model mevsimsel sekli ogrenirken yazi azinlikta gorup ortalamaya cekiyor
olabilir (olculdu: gonderimde Nis->Tem sekil +0.33, gecen yil gercek +0.64;
bir kismi 2026'nin serin gecmesiyle aciklaniyor ama tamami degil).

Deney: ufuk ayi hedef pencerenin aylarina denk gelen egitim satirlarina
`w` agirligi ver. Fold'da hedef pencere = o fold'un dogrulama aylari,
gonderimde Nis-Tem olacak.

Kullanim: python scripts/38_season_weight.py
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
WEIGHTS = (1.0, 2.0, 4.0)


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    records, lines = [], ["# Mevsim-esli agirliklandirma\n", f"Uretim: {pd.Timestamp.now()}\n"]

    for fold in C.FOLDS:
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

        target_months = set(pd.date_range(fold.valid_start, fold.valid_end).month)
        in_season = tr_f["month"].isin(target_months).to_numpy()
        y = va_f[C.TARGET].to_numpy()
        seg = split.segment.reindex(va_f.index)
        cold_s = split.is_cold.reindex(va_f.index)
        p_row = va_f[C.LOCATION].map(p_table["by_lokasyon"]).fillna(p_table["global"])
        p_row = np.where(cold_s.to_numpy(), p_row.to_numpy(), 0.0)
        ytr = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
        lvl_tr, lvl_va = tr_f[LEVEL_COL].to_numpy(), va_f[LEVEL_COL].to_numpy()
        Xtr, cat = model_matrix(tr_f)
        Xva, _ = model_matrix(va_f)
        Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
        Xva = Xva[Xtr.columns]
        hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]

        lines.append(f"\n## Fold {fold.name}  (mevsim-ici egitim satiri {in_season.mean():.1%})\n\n")
        lines.append("| w | warm | cold | blend |\n| --- | --- | --- | --- |\n")
        for w in WEIGHTS:
            t0 = time.time()
            sw = np.where(in_season, w, 1.0) if w != 1.0 else None
            m = fit_shape_model(Xtr, ytr, lvl_tr, cat, hist_cols,
                                seed=C.SEED + 3, n_trees=600, weight=sw)
            pred = shrink_level(predict_level_shape(m, Xva, lvl_va), p_row, strength=GATE_K)
            met = segment_report(y, pred, seg, is_cold=cold_s)
            records.append({"model": f"w{w}", "fold": fold.name, **met})
            lines.append("| {:.0f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                w, met["rmsle_warm"], met["rmsle_cold"], met["rmsle_blend"]))
            print(fold.name, w, round(met["rmsle_blend"], 4), f"{time.time()-t0:.0f}s", flush=True)
        del tr_f, va_f, Xtr, Xva

    df = pd.DataFrame(records)
    lines.append("\n## Ozet\n\n| w | mean | std | warm |\n| --- | --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append(f"| {model} | {s['mean']:.4f} | {s['std']:.4f} | {g['rmsle_warm'].mean():.4f} |\n")
    piv = df.pivot(index="fold", columns="model", values="rmsle_blend")
    for c in piv.columns:
        lines.append(f"\n- {c} vs w1.0: {(piv[c]-piv['w1.0']).round(4).to_dict()}")
    (C.REPORTS_DIR / "38_season_weight.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().to_string())


if __name__ == "__main__":
    main()
