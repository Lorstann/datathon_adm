"""Takvim / tatil dogrulamasi: Diyanet 2025-2026 bayram tarihleri."""

from __future__ import annotations

import pandas as pd

from src.features.calendar import calendar_features


def test_kurban_2026_dates():
    d = pd.Series(pd.date_range("2026-05-26", "2026-05-30", freq="D"))
    cal = calendar_features(d)
    # 26 Mayis arife, 27-30 kurban
    assert cal.loc[d[d == "2026-05-26"].index[0], "is_arife"] == 1
    assert cal.loc[d[d == "2026-05-27"].index[0], "is_kurban"] == 1
    assert cal.loc[d[d == "2026-05-30"].index[0], "is_kurban"] == 1


def test_ramazan_bayram_2025_masks_april_first():
    d = pd.Series(pd.to_datetime(["2025-04-01"]))
    cal = calendar_features(d)
    assert cal["is_ramazan_bayram"].iloc[0] == 1


def test_harmonics_range():
    d = pd.Series(pd.date_range("2025-01-01", periods=30, freq="D"))
    cal = calendar_features(d)
    assert cal["sin_doy"].between(-1, 1).all()
    assert cal["is_weekend"].isin([0, 1]).all()
