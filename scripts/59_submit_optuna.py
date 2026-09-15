"""22_submit_multiorigin.py ile birebir ayni boru hatti, tek fark: Optuna
taramasinin kazanan hiperparametreleri (reports/57_optuna_tune.md).

CV: 3-fold mean 1.1940 -> 1.1917, std 0.1356 -> 0.1333 (fold A -0.0052,
C -0.0023, B +0.0006 -- gurultu). Ilk defa sistematik hiperparametre
aramasindan gelen kazanc; onceki tum ayarlar (num_leaves=63, lr=0.05,
depth sinirsiz) round1'den beri hic degismemisti.

python scripts/59_submit_optuna.py
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
from src.data.load import (
    build_panel,
    entity_static,
    load_sample_submission,
    load_test,
    load_train,
)
from src.external.weather import CACHE_PATH, load_weather, weather_features
from src.features.build import assemble, model_matrix
from src.features.peer import peer_tables
from src.models.level_shape import (
    WARM_LEVEL_CHAIN,
    entity_level,
    fit_shape_model,
    predict_level_shape,
)
from src.features.history import entity_history_features
from src.models.multiorigin import (
    LEVEL_COL,
    build_training_frame,
    frozen_history,
    mask_cold,
    training_origins,
)
from src.models.zero_gate import dead_probability, shrink_level
from src.validation.folds import build_cold_profile

N_TREES = 600
MAX_ROWS = 1_800_000
SEEDS = (C.SEED + 3, C.SEED + 13, C.SEED + 23)
GATE_K = 0.75
OUT = C.SUBMISSIONS_DIR / "submission_optuna.csv"

# reports/57_optuna_tune.md: optuna_top1, CV 3-fold mean 1.1940 -> 1.1917
OPTUNA_PARAMS = {
    "num_leaves": 145,
    "min_child_samples": 63,
    "learning_rate": 0.07686932565906376,
    "subsample": 0.8400055837408262,
    "colsample_bytree": 0.5402386648520059,
    "reg_lambda": 0.020901689794526224,
    "reg_alpha": 0.003402265507666337,
    "max_depth": 4,
}


def main() -> None:
    C.ensure_dirs()
    t0 = time.time()
    train, test, sample = load_train(), load_test(), load_sample_submission()
    static = entity_static(build_panel(train, test))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None
    origin = C.TRAIN_END

    origins = training_origins(origin, max_origins=8)
    print("origins:", [str(o.date()) for o in origins], flush=True)
    tr_f = build_training_frame(
        train, end_cap=origin, static=static, weather=weather,
        origins=origins, max_rows=MAX_ROWS,
    )
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    tr_f = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED, entry_offsets=profile.entry_offsets)
    print(f"egitim satiri: {len(tr_f):,}  ({time.time()-t0:.0f}s)", flush=True)

    hist = frozen_history(train, origin, weather)
    peer = peer_tables(train, origin)
    te_f = assemble(
        test, origin=origin, history=hist, weather=weather, static=static,
        peer=peer, include_history=True,
    )
    te_f[LEVEL_COL] = entity_level(
        hist, te_f[C.ENTITY], te_f["log_guc"].to_numpy(), warm_col=WARM_LEVEL_CHAIN
    )

    y = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
    lvl_tr = tr_f[LEVEL_COL].to_numpy()
    lvl_te = te_f[LEVEL_COL].to_numpy()
    print(f"model_matrix basliyor ({time.time()-t0:.0f}s)", flush=True)
    Xtr, cat = model_matrix(tr_f)
    print(f"Xtr hazir ({time.time()-t0:.0f}s)", flush=True)
    Xte, _ = model_matrix(te_f)
    print(f"Xte hazir ({time.time()-t0:.0f}s)", flush=True)
    Xtr, Xte = Xtr.align(Xte, join="outer", axis=1, fill_value=np.nan)
    Xte = Xte[Xtr.columns]
    hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]
    print(f"hazir, {len(Xtr.columns)} kolon ({time.time()-t0:.0f}s)", flush=True)

    acc = np.zeros(len(Xte))
    for s in SEEDS:
        print(f"  seed {s} basliyor ({time.time()-t0:.0f}s)", flush=True)
        m = fit_shape_model(
            Xtr, y, lvl_tr, cat, hist_cols, seed=s, n_trees=N_TREES,
            params_override=OPTUNA_PARAMS,
        )
        acc += np.log1p(predict_level_shape(m, Xte, lvl_te))
        print(f"  seed {s} bitti ({time.time()-t0:.0f}s)", flush=True)
    pred = np.clip(np.expm1(acc / len(SEEDS)), 0.0, None)

    is_cold = ~te_f[C.ENTITY].isin(set(train[C.ENTITY].unique())).to_numpy()
    p_table = dead_probability(train, origin)
    p_row = te_f[C.LOCATION].map(p_table["by_lokasyon"]).fillna(p_table["global"])
    p_row = np.where(is_cold, p_row.to_numpy(), 0.0)
    pred = shrink_level(pred, p_row, strength=GATE_K)

    out = sample.copy()
    out[C.TARGET] = pred
    assert out["id"].equals(test["id"]), "id sirasi sample_submission ile uyusmuyor"
    out.to_csv(OUT, index=False)
    print(f"yazildi {OUT}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
