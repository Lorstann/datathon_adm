"""Cok-origin C: seviye zinciri / agac sayisi / seviye ofseti ablasyonu.

Fold A veri acligindan (tek kullanilabilir origin, ~88k satir) disarida:
gonderim origin'inde 8 origin var, A'nin rejimi orayi temsil etmiyor.

Kullanim: python scripts/21_mo_tune.py
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
from src.models.boosting import fit_lightgbm
from src.models.level_shape import entity_level, fit_shape_model, predict_level_shape
from src.models.multiorigin import LEVEL_COL, build_training_frame, mask_cold
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

_cv = import_module("20_multiorigin_cv")

CHAINS = {
    "chain_14": ("hist_mean_14", "hist_mean_28", "hist_mean_91", "hist_mean"),
    "chain_7": ("hist_mean_7", "hist_mean_28", "hist_mean_91", "hist_mean"),
    "chain_21": ("hist_mean_21", "hist_mean_28", "hist_mean_91", "hist_mean"),
    "chain_all": ("hist_mean",),
}


def relevel(frame: pd.DataFrame, hist: pd.DataFrame, chain) -> np.ndarray:
    return entity_level(
        hist, frame[C.ENTITY], frame["log_guc"].to_numpy(), warm_col=chain
    )


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    folds = [f for f in C.FOLDS if f.name != "A_mevsim"]
    records: list[dict] = []
    lines = ["# Cok-origin C ablasyonu\n", f"Uretim: {pd.Timestamp.now()}\n"]

    for fold in folds:
        split = make_fold(train, fold, profile, seed=C.SEED)
        tr_f = build_training_frame(
            split.train,
            end_cap=fold.origin,
            static=static,
            weather=weather,
            max_rows=_cv.MAX_ROWS,
        )
        va_f = _cv.valid_frame(split, static, weather)
        va_f = va_f.loc[va_f[C.TARGET].notna()]
        hist_va = entity_history_features(split.train, fold.origin)

        ytr = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
        yva = va_f[C.TARGET].to_numpy()
        seg = split.segment.reindex(va_f.index)
        cold = split.is_cold.reindex(va_f.index)

        # Cold maskesi bir kez secilir; butun zincir varyantlari ayni maskeyi
        # paylasir, yoksa varyantlar arasi fark maskeden gelir.
        tr_masked = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED)
        is_masked = tr_masked["hist_n"].isna().to_numpy() & tr_f["hist_n"].notna().to_numpy()
        cold_lvl_tr = tr_f["_cold_level"].to_numpy()

        Xtr, cat = model_matrix(tr_masked)
        Xva, _ = model_matrix(va_f)
        Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
        Xva = Xva[Xtr.columns]
        hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]
        logg_va = va_f["log_guc"].to_numpy()

        hist_by_origin = {
            o: entity_history_features(split.train, o) for o in tr_f["_origin"].unique()
        }

        header = f"\n## Fold {fold.name}  egitim satiri={len(tr_f):,}\n\n"
        lines.append(header)
        lines.append("| variant | warm | cold | blend | s |\n")
        lines.append("| --- | --- | --- | --- | --- |\n")

        def run(name: str, pred: np.ndarray, t0: float) -> None:
            m = segment_report(yva, pred, seg, is_cold=cold)
            records.append({"model": name, "fold": fold.name, **m})
            lines.append(
                "| {} | {:.4f} | {:.4f} | {:.4f} | {:.0f} |\n".format(
                    name,
                    m["rmsle_warm"],
                    m["rmsle_cold"],
                    m["rmsle_blend"],
                    time.time() - t0,
                )
            )
            print(fold.name, name, round(m["rmsle_blend"], 4), flush=True)

        for cname, chain in CHAINS.items():
            parts = [
                pd.Series(relevel(g, hist_by_origin[o], chain), index=g.index)
                for o, g in tr_f.groupby("_origin", sort=False)
            ]
            lvl_tr = pd.concat(parts).reindex(tr_f.index).to_numpy()
            lvl_tr = np.where(is_masked, cold_lvl_tr, lvl_tr)
            lvl_va = relevel(va_f, hist_va, chain)

            tree_grid = (600, 1500) if cname == "chain_14" else (600,)
            for n_trees in tree_grid:
                t0 = time.time()
                m = fit_shape_model(
                    Xtr, ytr, lvl_tr, cat, hist_cols, seed=C.SEED + 3, n_trees=n_trees
                )
                run("{}_t{}".format(cname, n_trees), predict_level_shape(m, Xva, lvl_va), t0)

            if cname == "chain_14":
                # Seviye ofseti olmadan dogrudan log1p regresyonu: ozellikler
                # artik egitim ve tahminde ayni anlami tasidigi icin ofsetin
                # hala gerekli olup olmadigini test eder.
                t0 = time.time()
                mm = fit_lightgbm(
                    Xtr,
                    ytr,
                    cat,
                    seed=C.SEED + 5,
                    target="log1p",
                    mask_history_rate=0.0,
                    n_trees=1500,
                )
                run("no_level_t1500", mm.predict(Xva, logg_va), t0)

    df = pd.DataFrame(records)
    lines.append("\n## Ozet (rmsle_blend)\n\n")
    lines.append("| variant | mean | std | warm | cold |\n")
    lines.append("| --- | --- | --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                model, s["mean"], s["std"], g["rmsle_warm"].mean(), g["rmsle_cold"].mean()
            )
        )
    best = df.groupby("model")["rmsle_blend"].mean().idxmin()
    lines.append("\nEn iyi: **{}**\n".format(best))
    (C.REPORTS_DIR / "21_mo_tune.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().sort_values().to_string())


if __name__ == "__main__":
    main()
