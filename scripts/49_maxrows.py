"""Ayni origin'ler, daha cok satir. Kapasite deneyi (reports/25) origin
YOGUNLUGUNU test etmisti; satir kirpmasini hic test etmedik. Gonderim
cercevesi 2.4M satir uretip 1.8M'de kirpiliyor.
"""
from __future__ import annotations
import sys, time
from importlib import import_module
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd
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
CAPS = [("cap_1.2M", 1_200_000), ("kirpmasiz", 10**9)]

def main():
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None
    records, lines = [], ["# Satir kirpmasi\n", f"Uretim: {pd.Timestamp.now()}\n"]
    for fold in C.FOLDS:
        split = make_fold(train, fold, profile, seed=C.SEED)
        p_table = dead_probability(split.train, fold.origin)
        va_f = _cv.valid_frame(split, static, weather)
        va_f = va_f.loc[va_f[C.TARGET].notna()]
        y = va_f[C.TARGET].to_numpy()
        seg = split.segment.reindex(va_f.index)
        cold_s = split.is_cold.reindex(va_f.index)
        p_row = va_f[C.LOCATION].map(p_table["by_lokasyon"]).fillna(p_table["global"])
        p_row = np.where(cold_s.to_numpy(), p_row.to_numpy(), 0.0)
        Xva_full, _ = model_matrix(va_f)
        lvl_va = va_f[LEVEL_COL].to_numpy()
        lines.append(f"\n## Fold {fold.name}\n\n| variant | satir | warm | cold | blend |\n")
        lines.append("| --- | --- | --- | --- | --- |\n")
        for name, cap in CAPS:
            t0 = time.time()
            tr_f = build_training_frame(split.train, end_cap=fold.origin, static=static,
                                        weather=weather, max_rows=cap)
            tr_f = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED,
                             entry_offsets=profile.entry_offsets)
            ytr = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
            Xtr, cat = model_matrix(tr_f)
            Xtr, Xva = Xtr.align(Xva_full, join="outer", axis=1, fill_value=np.nan)
            Xva = Xva[Xtr.columns]
            hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]
            m = fit_shape_model(Xtr, ytr, tr_f[LEVEL_COL].to_numpy(), cat, hist_cols,
                                seed=C.SEED + 3, n_trees=600)
            pred = shrink_level(predict_level_shape(m, Xva, lvl_va), p_row, strength=GATE_K)
            met = segment_report(y, pred, seg, is_cold=cold_s)
            records.append({"model": name, "fold": fold.name, **met})
            lines.append("| {} | {:,} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                name, len(tr_f), met["rmsle_warm"], met["rmsle_cold"], met["rmsle_blend"]))
            print(fold.name, name, len(tr_f), round(met["rmsle_blend"], 4),
                  f"{time.time()-t0:.0f}s", flush=True)
            del tr_f, Xtr, Xva
        del va_f, Xva_full
    df = pd.DataFrame(records)
    lines.append("\n## Ozet\n\n| variant | mean | std | warm | cold |\n| --- | --- | --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            model, s["mean"], s["std"], g["rmsle_warm"].mean(), g["rmsle_cold"].mean()))
    piv = df.pivot(index="fold", columns="model", values="rmsle_blend")
    lines.append(f"\nfark (kirpmasiz - cap): {(piv['kirpmasiz'] - piv['cap_1.2M']).round(4).to_dict()}\n")
    (C.REPORTS_DIR / "49_maxrows.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().to_string())

if __name__ == "__main__":
    main()
