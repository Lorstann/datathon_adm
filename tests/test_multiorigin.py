"""Cok-origin egitim cercevesi ve seviye zincirinin degismezleri."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.features.history import entity_history_features
from src.models.level_shape import entity_level
from src.models.multiorigin import (
    COLD_LEVEL_COL,
    LEVEL_COL,
    build_training_frame,
    mask_cold,
    origin_frame,
    training_origins,
)


def _panel(n_entities: int = 12, days: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range(C.TRAIN_START, periods=days, freq="D")
    rows = []
    for i in range(n_entities):
        base = 5.0 + i * 0.3
        start = 0 if i % 4 else days // 2  # bir kismi sonradan devreye giriyor
        for d in dates[start:]:
            rows.append(
                {
                    C.ENTITY: f"E{i}",
                    C.POWER: int(100 * (1 + i % 5)),
                    C.DATE: d,
                    C.TARGET: float(np.expm1(base + rng.normal(0, 0.1))),
                    C.LOCATION: "İZMİR>METROPOL>BORNOVA" if i % 2 else "MANİSA>GÖRDES",
                }
            )
    return pd.DataFrame(rows).sort_values([C.ENTITY, C.DATE], ignore_index=True)


def _static(panel: pd.DataFrame) -> pd.DataFrame:
    from src.data.load import entity_static, split_location

    p = panel.copy()
    p["id"] = pd.NA
    p["is_test"] = False
    p = p.join(split_location(p[C.LOCATION]))
    return entity_static(p)


def test_training_origins_respect_spacing_and_min_history():
    end = pd.Timestamp("2026-03-31")
    origins = training_origins(end, spacing_days=45, min_history_days=30)
    assert origins == sorted(origins, reverse=True)
    assert all(o < end for o in origins)
    assert all((o - C.TRAIN_START).days >= 30 for o in origins)
    assert all(
        (origins[i] - origins[i + 1]).days == 45 for i in range(len(origins) - 1)
    )


def test_origin_frame_horizon_is_forward_and_capped():
    panel = _panel()
    static = _static(panel)
    origin = C.TRAIN_START + pd.Timedelta(days=150)
    cap = origin + pd.Timedelta(days=60)
    f = origin_frame(panel, origin, static=static, weather=None, end_cap=cap)

    assert f[C.DATE].min() > origin, "egitim satiri origin'i gecmeli"
    assert f[C.DATE].max() <= cap, "end_cap asilamaz"
    # Test satirlariyla ayni isaret ve aralik: 1..horizon
    assert f["horizon_day"].min() >= 1
    assert f["horizon_day"].max() <= (cap - origin).days
    assert f[LEVEL_COL].notna().all()
    assert f[COLD_LEVEL_COL].notna().all()


def test_history_features_are_frozen_within_a_horizon():
    """Ayni origin'de bir trafonun gecmis ozeti ufuk boyunca degismez."""
    panel = _panel()
    static = _static(panel)
    origin = C.TRAIN_START + pd.Timedelta(days=150)
    f = origin_frame(
        panel, origin, static=static, weather=None, end_cap=origin + pd.Timedelta(days=60)
    )
    spread = f.groupby(C.ENTITY)["hist_mean"].agg(lambda s: s.max() - s.min())
    assert float(np.nanmax(spread.to_numpy())) == 0.0


def test_mask_cold_drops_history_and_falls_back_to_cold_level():
    panel = _panel()
    static = _static(panel)
    origin = C.TRAIN_START + pd.Timedelta(days=150)
    f = origin_frame(
        panel, origin, static=static, weather=None, end_cap=origin + pd.Timedelta(days=60)
    )
    m = mask_cold(f, f[C.ENTITY], rate=0.5, seed=1)
    masked = m["hist_n"].isna() & f["hist_n"].notna()
    assert masked.any(), "maskeleme hicbir satiri etkilemedi"
    # Maskelenen satirda seviye cold koluna dusmus olmali
    assert np.allclose(
        m.loc[masked, LEVEL_COL].to_numpy(), f.loc[masked, COLD_LEVEL_COL].to_numpy()
    )
    # Maskeleme varlik duzeyinde: bir trafonun ya tum satirlari gider ya hicbiri
    per_entity = masked.groupby(f[C.ENTITY]).nunique()
    assert set(per_entity.unique()) <= {1}


def test_entity_level_chain_falls_back_column_by_column():
    hist = pd.DataFrame(
        {
            "hist_mean_14": [1.0, np.nan, np.nan],
            "hist_mean_28": [2.0, 2.5, np.nan],
            "hist_mean": [3.0, 3.5, np.nan],
            "hist_mean_z": [0.0, 0.0, 0.0],
        },
        index=pd.Index(["a", "b", "c"], name=C.ENTITY),
    )
    ents = pd.Series(["a", "b", "c"])
    log_guc = np.array([1.0, 1.0, 1.0])
    lvl = entity_level(
        hist, ents, log_guc, warm_col=("hist_mean_14", "hist_mean_28", "hist_mean")
    )
    assert lvl[0] == 1.0  # ilk pencere var
    assert lvl[1] == 2.5  # ilk pencere yok, ikinciye dustu
    assert lvl[2] == 1.0  # hicbiri yok: cold kolu = global z (0) + log_guc


def test_build_training_frame_stacks_origins_and_caps_rows():
    panel = _panel()
    static = _static(panel)
    cap = C.TRAIN_START + pd.Timedelta(days=280)
    origins = [
        C.TRAIN_START + pd.Timedelta(days=d) for d in (120, 165, 210)
    ]
    f = build_training_frame(
        panel, end_cap=cap, static=static, weather=None, origins=origins, max_rows=10**9
    )
    assert set(f["_origin"].unique()) <= set(origins)
    assert f[C.DATE].max() <= cap

    small = build_training_frame(
        panel, end_cap=cap, static=static, weather=None, origins=origins, max_rows=50
    )
    assert len(small) == 50


def test_fold_cold_flag_matches_zero_history():
    """`is_cold` maskelenmis olmayi degil, gecmissizligi isaretlemeli."""
    from src.data.load import build_panel, entity_static, load_test, load_train
    from src.validation.folds import build_cold_profile
    from src.validation.split import make_fold

    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    split = make_fold(train, C.FOLDS[1], profile, seed=C.SEED)
    assert split.is_cold.equals(split.history_days.eq(0).reset_index(drop=True))
    # Maskelenen trafo valid'de hala gorunuyorsa cold sayilmali. Bir kismi
    # `apply_entry_delay` ile tamamen dusebilir; onlar valid'de hic yok.
    still_present = set(split.masked_entities) & set(split.valid[C.ENTITY])
    assert still_present <= set(split.valid.loc[split.is_cold.to_numpy(), C.ENTITY])
