"""Tam train uzerinde model kurar, submission.csv yazar.

Gonderim Kaggle MCP ile yapilir; bu betik dosyayi hazirlar ve experiments.csv'ye
yerel not duser. `--submit` bayragi yoktur: her gonderim sozlu onay ister.

Kullanim: python scripts/07_submit.py
"""

from __future__ import annotations

import sys
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
from src.features.history import add_lagged_history, entity_history_features
from src.features.peer import peer_tables
from src.models.boosting import fit_catboost, fit_lightgbm, fit_xgboost
from src.models.level_shape import entity_level, fit_shape_model, predict_level_shape
from src.models.pipeline import mask_history_entities
from src.models.zero_gate import apply_zero_gate, zero_probability


def _weather():
    if CACHE_PATH.exists():
        return weather_features(load_weather())
    print("hava cache yok")
    return None


def main() -> None:
    C.ensure_dirs()
    train = load_train()
    test = load_test()
    sample = load_sample_submission()
    panel = build_panel(train, test)
    static = entity_static(panel)
    weather = _weather()
    origin = C.TRAIN_END

    hist = entity_history_features(train, origin)
    peer = peer_tables(train, origin)

    tr = add_lagged_history(train.sort_values([C.ENTITY, C.DATE], ignore_index=True))
    train_f = assemble(
        tr, origin=origin, history=hist, weather=weather, static=static, peer=peer, include_history=False
    )
    test_f = assemble(
        test, origin=origin, history=hist, weather=weather, static=static, peer=peer, include_history=True
    )

    y = np.log1p(train_f[C.TARGET].clip(lower=0).to_numpy())
    Xtr, cat = model_matrix(train_f)
    Xte, _ = model_matrix(test_f)
    Xtr, Xte = Xtr.align(Xte, join="outer", axis=1, fill_value=np.nan)
    Xte = Xte[Xtr.columns]
    logg_te = test_f["log_guc"].to_numpy()
    yz = y - train_f["log_guc"].to_numpy()

    Xtr_m = mask_history_entities(Xtr, train_f[C.ENTITY], 0.30, C.SEED)

    print("egitim LightGBM A...")
    m_a = fit_lightgbm(Xtr_m, y, cat, seed=C.SEED, n_trees=500)
    print("egitim LightGBM z...")
    m_z = fit_lightgbm(Xtr_m, yz, cat, seed=C.SEED + 1, n_trees=500, target="z")
    print("egitim CatBoost...")
    m_cat = fit_catboost(Xtr_m, y, cat, seed=C.SEED, n_trees=400)
    print("egitim XGBoost...")
    m_xgb = fit_xgboost(Xtr_m, y, seed=C.SEED, n_trees=400)

    hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]
    Xtr_cold = Xtr.copy()
    Xtr_cold[hist_cols] = np.nan
    Xte_cold = Xte.copy()
    Xte_cold[hist_cols] = np.nan
    print("egitim cold z...")
    m_cold = fit_lightgbm(Xtr_cold, yz, cat, seed=C.SEED + 2, n_trees=400, target="z")

    print("egitim seviye+sekil C...")
    level_tr = entity_level(hist, train_f[C.ENTITY], train_f["log_guc"].to_numpy())
    level_te = entity_level(hist, test_f[C.ENTITY], logg_te)
    m_shape = fit_shape_model(Xtr_m, y, level_tr, cat, hist_cols, seed=C.SEED + 3)
    pred_c = predict_level_shape(m_shape, Xte, level_te)

    pred_a = m_a.predict(Xte, logg_te)
    pred_z = m_z.predict(Xte, logg_te)
    pred_cat = m_cat.predict(Xte, logg_te)
    pred_xgb = m_xgb.predict(Xte, logg_te)
    pred_cold = m_cold.predict(Xte_cold, logg_te)

    train_ents = set(train[C.ENTITY].unique())
    is_cold = ~test[C.ENTITY].isin(train_ents)
    p0 = zero_probability(hist, test[C.ENTITY]).to_numpy()
    thr = 0.85

    pred_a = apply_zero_gate(pred_a, p0, threshold=thr)
    pred_cold = apply_zero_gate(pred_cold, p0, threshold=thr)
    pred_b = cold_warm_blend(pred_a, pred_cold, is_cold)
    ens = apply_zero_gate(
        0.45 * pred_b + 0.25 * pred_z + 0.15 * pred_cat + 0.15 * pred_xgb,
        p0,
        threshold=thr,
    )

    def write(pred: np.ndarray, name: str) -> Path:
        out = sample.copy()
        # sample sirasi test ile ayni dogrulandi (reports/01)
        out[C.TARGET] = np.clip(pred, 0.0, None)
        path = C.SUBMISSIONS_DIR / name
        out.to_csv(path, index=False)
        print(f"yazildi {path}  satir={len(out):,}  mean={out[C.TARGET].mean():.1f}")
        return path

    p1 = write(pred_b, "submission_lgbm_bc.csv")
    p2 = write(ens, "submission_ensemble.csv")
    p3 = write(pred_c, "submission_level_shape.csv")
    print("Gonderim icin hazir. Kaggle'a YUKLEME: sozlu onay sonrasi MCP ile.")
    print(p1)
    print(p2)
    print(p3)


if __name__ == "__main__":
    main()
