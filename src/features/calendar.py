"""Takvim, tatil, bayram ve Fourier harmonikleri.

Bayram yil-uzeri kaymasi: Kurban 2025-06-06 -> 2026-05-27 (~10 gun erken).
`bayrama_kalan_gun` Gregoryen lag'in sessizce bozmasini engeller.
Ramazan aylari Diyanet takviminden sabitlendi (train icinde, test'te yok).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import holidays
from holidays.constants import HALF_DAY, PUBLIC

from src import config as C

# Diyanet: Ramazan 2025 = 1 Mar - 29 Mar; 2026 = 19 Sub - 19 Mar.
RAMADAN_RANGES = (
    (pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-29")),
    (pd.Timestamp("2026-02-19"), pd.Timestamp("2026-03-19")),
)

_TR_CACHE: dict[int, object] = {}


def _tr_holidays(years: list[int]):
    key = hash(tuple(sorted(years)))
    if key not in _TR_CACHE:
        _TR_CACHE[key] = holidays.country_holidays(
            "TR",
            years=years,
            categories=(PUBLIC, HALF_DAY),
            language="tr",
        )
    return _TR_CACHE[key]


def _nearest_signed(day: pd.Series, bayram_dates: pd.DatetimeIndex) -> pd.Series:
    if len(bayram_dates) == 0:
        return pd.Series(0, index=day.index, dtype="int16")
    b = bayram_dates.to_numpy(dtype="datetime64[D]").astype(np.int64)
    t = day.to_numpy(dtype="datetime64[D]").astype(np.int64)
    i = np.searchsorted(b, t)
    i0 = np.clip(i - 1, 0, len(b) - 1)
    i1 = np.clip(i, 0, len(b) - 1)
    d0 = b[i0] - t
    d1 = b[i1] - t
    nearest = np.where(np.abs(d0) <= np.abs(d1), d0, d1)
    return pd.Series(nearest, index=day.index, dtype="int16")


def _is_ramadan(dates: pd.Series) -> pd.Series:
    out = pd.Series(False, index=dates.index)
    for a, b in RAMADAN_RANGES:
        out = out | ((dates >= a) & (dates <= b))
    return out


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Satir hizasinda takvim + tatil ozellikleri. Girdi bir tarih serisi."""
    d = pd.to_datetime(dates)
    years = list(range(int(d.min().year), int(d.max().year) + 1))
    tr = _tr_holidays(years)

    day = d.dt.normalize()
    uniq_days = pd.DatetimeIndex(day.unique())
    name_by_day = {ts: tr.get(ts.date()) for ts in uniq_days}
    names = day.map(name_by_day)
    is_holiday = names.notna()
    is_arife = names.fillna("").str.contains("saat 13", case=False, regex=False)
    is_kurban = names.fillna("").str.contains("Kurban", case=False, regex=False)
    is_ramazan_bayram = names.fillna("").str.contains(
        "Ramazan Bayram", case=False, regex=False
    )

    doy = d.dt.dayofyear.astype("int16")
    dow = d.dt.dayofweek.astype("int8")
    month = d.dt.month.astype("int8")

    bayram_mask = is_kurban | is_ramazan_bayram
    bayram_dates = pd.DatetimeIndex(
        pd.Series(name_by_day)
        .loc[lambda s: s.fillna("").str.contains("Kurban|Ramazan Bayram", case=False, regex=True)]
        .index
    ).sort_values()

    # bayram_gun_indeksi: ayni bayram blogu icindeki 1..n gun. Arife 0.
    # Bloğu isim + yil ile gruplamak yerine takvimde ardışık tatil günleri.
    bayram_idx = pd.Series(0, index=d.index, dtype="int8")
    if bayram_mask.any():
        ordered = pd.DataFrame({"tarih": d, "flag": bayram_mask}).sort_values("tarih")
        grp = (ordered["flag"] != ordered["flag"].shift()).cumsum()
        within = ordered.groupby(grp).cumcount() + 1
        bayram_idx.loc[ordered.index] = np.where(ordered["flag"], within, 0).astype("int8")

    out = pd.DataFrame(
        {
            "year": d.dt.year.astype("int16"),
            "month": month,
            "day": d.dt.day.astype("int8"),
            "dow": dow,
            "doy": doy,
            "week": d.dt.isocalendar().week.astype("int8"),
            "is_weekend": (dow >= 5).astype("int8"),
            "is_month_start": d.dt.is_month_start.astype("int8"),
            "is_month_end": d.dt.is_month_end.astype("int8"),
            "sin_doy": np.sin(2 * np.pi * doy / 365.25),
            "cos_doy": np.cos(2 * np.pi * doy / 365.25),
            "sin_dow": np.sin(2 * np.pi * dow / 7.0),
            "cos_dow": np.cos(2 * np.pi * dow / 7.0),
            "sin_month": np.sin(2 * np.pi * month / 12.0),
            "cos_month": np.cos(2 * np.pi * month / 12.0),
            "is_holiday": is_holiday.astype("int8"),
            "is_arife": is_arife.astype("int8"),
            "is_kurban": is_kurban.astype("int8"),
            "is_ramazan_bayram": is_ramazan_bayram.astype("int8"),
            "is_ramadan": _is_ramadan(d).astype("int8"),
            "bayram_gun_indeksi": bayram_idx.astype("int8"),
            "bayrama_kalan_gun": _nearest_signed(day, bayram_dates),
        },
        index=dates.index,
    )
    out["is_kopru"] = (
        (~out["is_holiday"].astype(bool))
        & (dow < 5)
        & (out["bayrama_kalan_gun"].abs() == 1)
    ).astype("int8")
    return out
