"""CV kazananı seviye+sekil (C) icin tek gonderim. python scripts/08_submit_c.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

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
from src.features.history import add_lagged_history, entity_history_features
from src.features.peer import peer_tables
from src.models.level_shape import entity_level, fit_shape_model, predict_level_shape
from src.models.pipeline import mask_history_entities


def main() -> None:
    C.ensure_dirs()
    train, test, sample = load_train(), load_test(), load_sample_submission()
    static = entity_static(build_panel(train, test))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None
    origin = C.TRAIN_END
    hist = entity_history_features(train, origin)
    peer = peer_tables(train, origin)
    tr = add_lagged_history(train.sort_values([C.ENTITY, C.DATE], ignore_index=True))
    train_f = assemble(tr, origin=origin, history=hist, weather=weather, static=static, peer=peer, include_history=False)
    test_f = assemble(test, origin=origin, history=hist, weather=weather, static=static, peer=peer, include_history=True)
    y = np.log1p(train_f[C.TARGET].clip(lower=0).to_numpy())
    Xtr, cat = model_matrix(train_f)
    Xte, _ = model_matrix(test_f)
    Xtr, Xte = Xtr.align(Xte, join="outer", axis=1, fill_value=np.nan)
    Xte = Xte[Xtr.columns]
    Xtr_m = mask_history_entities(Xtr, train_f[C.ENTITY], 0.30, C.SEED)
    hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]
    level_tr = entity_level(hist, train_f[C.ENTITY], train_f["log_guc"].to_numpy())
    level_te = entity_level(hist, test_f[C.ENTITY], test_f["log_guc"].to_numpy())
    print("egitim C...")
    m = fit_shape_model(Xtr_m, y, level_tr, cat, hist_cols, seed=C.SEED + 3)
    pred = predict_level_shape(m, Xte, level_te)
    out = sample.copy()
    out[C.TARGET] = np.clip(pred, 0.0, None)
    path = C.SUBMISSIONS_DIR / "submission_level_shape.csv"
    out.to_csv(path, index=False)
    print(f"yazildi {path}  mean={out[C.TARGET].mean():.1f}")


if __name__ == "__main__":
    main()
