"""Cok-origin egitim cercevesi vs eski tek-adim cerceve. 3 fold, duzeltilmis is_cold.

Kullanim: python scripts/20_multiorigin_cv.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import build_panel, entity_static, load_test, load_train
from src.external.weather import CACHE_PATH, load_weather, weather_features
from src.features.build import assemble, model_matrix
from src.features.history import entity_history_features
from src.features.peer import peer_tables
from src.models.level_shape import (
    WARM_LEVEL_CHAIN,
    entity_level,
    fit_shape_model,
    power_elasticity,
    predict_level_shape,
)
from src.models.multiorigin import (
    LEVEL_COL,
    build_training_frame,
    frozen_history,
    mask_cold,
)
from src.features.history import seasonal_lag_columns
from src.models.pipeline import fold_frames, mask_history_entities, xy
from src.tracking import log_experiment
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

MAX_ROWS = 1_200_000
MAX_ORIGINS = 8


def _weather():
    if not CACHE_PATH.exists():
        print("hava cache yok, hava ozellikleri atlandi")
        return None
    return weather_features(load_weather())


def valid_frame(split, static, weather, blend_weights=None, seasonal_lag=False, use_external=False, use_elasticity=False):
    """Dogrulama cercevesi: gecmis fold origin'inde donmus, test ile ayni sekil."""
    origin = split.fold.origin
    hist = frozen_history(split.train, origin, weather)
    peer = peer_tables(split.train, origin)
    f = assemble(
        split.valid,
        origin=origin,
        history=hist,
        weather=weather,
        static=static,
        peer=peer,
        include_history=True,
        use_external=use_external,
    )
    # Maskelenen trafolar dogrulamada da gercek cold gibi gorunmeli:
    # `apply_entry_delay` satirlari kirpiyor ama transduktif kolonlar hala
    # trafonun GERCEK yasini tasiyor. Gercek test cold satirlarinda yas
    # ufka giristen itibaren sayiliyor.
    if len(split.masked_offsets):
        off = f[C.ENTITY].map(split.masked_offsets)
        sel = off.notna().to_numpy()
        if sel.any():
            hz = f["horizon_day"].to_numpy(dtype="float64")
            o = off.to_numpy(dtype="float64")
            if "devreye_alinma_yasi" in f.columns:
                f.loc[sel, "devreye_alinma_yasi"] = (hz - o)[sel]
            if "test_gun_sayisi" in f.columns:
                f.loc[sel, "test_gun_sayisi"] = (
                    split.fold.n_valid_days - o
                )[sel]

    if seasonal_lag:
        f = pd.concat(
            [f, seasonal_lag_columns(f, split.train, origin)], axis=1
        )
    f[LEVEL_COL] = entity_level(
        hist,
        f[C.ENTITY],
        f["log_guc"].to_numpy(),
        warm_col=WARM_LEVEL_CHAIN,
        blend_weights=blend_weights,
        elasticity=power_elasticity(hist) if use_elasticity else None,
    )
    return f


def old_path(split, static, weather):
    """Eski C: tek-adim lagged egitim satirlari, genisleyen ortalama seviye."""
    train_f, valid_f, hist = fold_frames(split, static, weather)
    Xtr, ytr, cat, logg_tr, tr_use = xy(train_f)
    Xva, yva, _, logg_va, va_use = xy(valid_f, dropna_target=True)
    Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
    Xva = Xva[Xtr.columns]
    Xtr_m = mask_history_entities(Xtr, tr_use[C.ENTITY], 0.30, C.SEED)
    hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]
    level_tr = entity_level(
        hist, tr_use[C.ENTITY], logg_tr, warm_col="hist_mean"
    )
    level_va = entity_level(
        hist, va_use[C.ENTITY], logg_va, warm_col="hist_mean"
    )
    m = fit_shape_model(Xtr_m, ytr, level_tr, cat, hist_cols, seed=C.SEED + 3)
    return predict_level_shape(m, Xva, level_va), va_use


def new_path(split, static, weather, n_trees=600):
    origin = split.fold.origin
    tr_f = build_training_frame(
        split.train,
        end_cap=origin,
        static=static,
        weather=weather,
        origins=None,
        max_rows=MAX_ROWS,
    )
    tr_f = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED)
    va_f = valid_frame(split, static, weather)
    va_f = va_f.loc[va_f[C.TARGET].notna()]

    ytr = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
    lvl_tr = tr_f[LEVEL_COL].to_numpy()
    lvl_va = va_f[LEVEL_COL].to_numpy()
    Xtr, cat = model_matrix(tr_f)
    Xva, _ = model_matrix(va_f)
    Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
    Xva = Xva[Xtr.columns]
    hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]
    m = fit_shape_model(
        Xtr, ytr, lvl_tr, cat, hist_cols, seed=C.SEED + 3, n_trees=n_trees
    )
    return predict_level_shape(m, Xva, lvl_va), va_f, m, len(tr_f)


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = _weather()

    records = []
    lines = [
        "# Cok-origin egitim cercevesi\n",
        f"Uretim: {pd.Timestamp.now()}\n\n",
        "`is_cold` duzeltmesi sonrasi (cold = origin'de gecmisi olmayan satir). "
        "Eski raporlardaki sayilarla dogrudan karsilastirilamaz.\n",
    ]

    for fold in C.FOLDS:
        t0 = time.time()
        split = make_fold(train, fold, profile, seed=C.SEED)
        cold_rate = float(split.is_cold.mean())

        pred_old, va_old = old_path(split, static, weather)
        seg_old = split.segment.reindex(va_old.index)
        cold_old = split.is_cold.reindex(va_old.index)
        m_old = segment_report(
            va_old[C.TARGET], pred_old, seg_old, is_cold=cold_old
        )

        pred_new, va_new, model, n_tr = new_path(split, static, weather)
        seg_new = split.segment.reindex(va_new.index)
        cold_new = split.is_cold.reindex(va_new.index)
        m_new = segment_report(
            va_new[C.TARGET], pred_new, seg_new, is_cold=cold_new
        )

        for name, m in (("level_shape_C_eski", m_old), ("multiorigin_C", m_new)):
            records.append({"model": name, "fold": fold.name, **m})
            log_experiment(
                exp_id=f"mo_{name}",
                model=name,
                target="log1p",
                feature_set="multiorigin" if "multi" in name else "onestep",
                fold=fold.name,
                metrics=m,
                notes=f"cold_rate={cold_rate:.3f}",
            )

        lines.append(
            f"\n## Fold {fold.name}  cold_row_rate={cold_rate:.3f}  "
            f"egitim satiri={n_tr:,}  ({time.time()-t0:.0f}s)\n\n"
        )
        lines.append("| Model | all | warm | cold | blend |\n| --- | --- | --- | --- | --- |\n")
        for name, m in (("level_shape_C_eski", m_old), ("multiorigin_C", m_new)):
            lines.append(
                f"| {name} | {m['rmsle_all']:.4f} | {m.get('rmsle_warm', float('nan')):.4f} | "
                f"{m.get('rmsle_cold', float('nan')):.4f} | {m.get('rmsle_blend', float('nan')):.4f} |\n"
            )
        print(lines[-3], lines[-1])

    df = pd.DataFrame(records)
    lines.append("\n## Ozet (rmsle_blend)\n\n| Model | mean | std |\n| --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append(f"| {model} | {s['mean']:.4f} | {s['std']:.4f} |\n")

    piv = df.pivot(index="fold", columns="model", values="rmsle_blend")
    delta = (piv["multiorigin_C"] - piv["level_shape_C_eski"]).mean()
    lines.append(f"\nDelta (multiorigin - eski), blend ortalamasi: **{delta:+.4f}**\n")

    path = C.REPORTS_DIR / "20_multiorigin.md"
    path.write_text("".join(lines), encoding="utf-8")
    print(f"yazildi: {path}")
    print(df.groupby("model")["rmsle_blend"].mean().to_string())


if __name__ == "__main__":
    main()
