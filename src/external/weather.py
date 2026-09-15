"""Open-Meteo ERA5-Land hava verisi: cekme, cache, ozellik turevleri.

HDD/CDD bazlari Ember'in Turkiye analizininden: CDD = max(0, T-22),
HDD = max(0, 18-T). V-sekli tepki dogrusal sicaklik teriminin yerine gecer.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests
from scipy.signal import savgol_filter

from src import config as C
from src.external.coords import LOCATION_COORDS

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = (
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "shortwave_radiation_sum",
    "sunshine_duration",
)
CDD_BASE = 22.0
HDD_BASE = 18.0
CACHE_PATH = C.WEATHER_DIR / "era5_daily.parquet"
NORMALS_PATH = C.WEATHER_DIR / "era5_normals.parquet"


def _fetch_one(
    lat: float,
    lon: float,
    start: str,
    end: str,
    retries: int = 8,
) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Europe/Istanbul",
    }
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=60)
            if r.status_code == 429:
                last_err = RuntimeError("429 Too Many Requests")
                time.sleep(20 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001 — ag gecici hatalari yutulsun
            last_err = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Open-Meteo basarisiz ({lat},{lon}): {last_err}")


def _grid_key(lat: float, lon: float, step: float = 0.1) -> tuple[float, float]:
    return (round(lat / step) * step, round(lon / step) * step)


def fetch_archive(
    start: str = "2025-01-01",
    end: str = "2026-07-31",
    coords: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Her ERA5 hucresi icin bir istek; lokasyonlar hucreye yapistirilir."""
    coords = coords or LOCATION_COORDS
    cell_to_locs: dict[tuple[float, float], list[str]] = {}
    for loc, (lat, lon) in coords.items():
        cell_to_locs.setdefault(_grid_key(lat, lon), []).append(loc)

    C.WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    cells = list(cell_to_locs.items())
    for i, ((lat, lon), locs) in enumerate(cells):
        part = C.WEATHER_DIR / f"_cell_{lat:.2f}_{lon:.2f}.parquet"
        if part.exists():
            daily_df = pd.read_parquet(part)
        else:
            payload = _fetch_one(lat, lon, start, end)
            daily = payload["daily"]
            daily_df = pd.DataFrame(daily)
            daily_df["tarih"] = pd.to_datetime(daily_df["time"])
            daily_df = daily_df.drop(columns=["time"])
            daily_df["lat_snap"] = payload.get("latitude")
            daily_df["lon_snap"] = payload.get("longitude")
            daily_df["elevation"] = payload.get("elevation")
            daily_df.to_parquet(part, index=False)
            time.sleep(2.5)
        for loc in locs:
            chunk = daily_df.copy()
            chunk[C.LOCATION] = loc
            frames.append(chunk)
        print(f"  hucre {i+1}/{len(cells)} {lat:.2f},{lon:.2f} -> {len(locs)} lokasyon")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values([C.LOCATION, C.DATE], ignore_index=True)


def fetch_and_cache(
    start: str = "2025-01-01",
    end: str = "2026-07-31",
    path=CACHE_PATH,
) -> pd.DataFrame:
    C.ensure_dirs()
    df = fetch_archive(start=start, end=end)
    df.to_parquet(path, index=False)
    return df


def load_normals(path=NORMALS_PATH) -> pd.DataFrame:
    """2016-2025 gercek cok-yillik gunun-normali tablosu (31 Mart kurali icin)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} yok. Once `python scripts/58_fetch_climate_normals.py` calistir."
        )
    return pd.read_parquet(path)


def load_weather(path=CACHE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} yok. Once `python scripts/04_fetch_weather.py` calistir."
        )
    df = pd.read_parquet(path)
    df[C.DATE] = pd.to_datetime(df[C.DATE])
    return df


def _savgol_safe(x: np.ndarray, window: int = 7, poly: int = 2) -> np.ndarray:
    if len(x) < window:
        return x.astype("float64")
    return savgol_filter(x.astype("float64"), window_length=window, polyorder=poly)


def weather_features(
    weather: pd.DataFrame,
    *,
    normals: bool = False,
    normals_cutoff: pd.Timestamp | None = None,
    normals_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ham ERA5 kolonlarindan HDD/CDD, hareketli toplam ve Savitzky-Golay.

    `normals=True` ise sicaklik kolonlarinin yilin gunu ortalamasi kullanilir
    (iklim normalleri ablasyonu). Hareketli toplamlar da o serinin uzerinden.

    `normals_cutoff` verilirse (31 Mart kurali uyumu icin) yilin-gunu ortalamasi
    YALNIZCA `tarih <= normals_cutoff` satirlarindan hesaplanir ve yalnizca
    `tarih > normals_cutoff` satirlarina uygulanir; cutoff'tan once gercek
    gozlenmis deger kalir. Aksi halde (parametresiz `normals=True`) ortalama
    test donemindeki GERCEKLESMIS 2026 degerini de icine katar -- bu da tam
    olarak yasaklanan "gelecek gercek hava verisi" sizintisinin kendisidir,
    yarim agirlikla.
    """
    df = weather.copy()
    t = df["temperature_2m_mean"].astype("float64")
    if normals:
        doy = df[C.DATE].dt.dayofyear
        cols = [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "apparent_temperature_mean",
            "relative_humidity_2m_mean",
            "shortwave_radiation_sum",
        ]
        cols = [c for c in cols if c in df.columns]
        if normals_cutoff is not None:
            if normals_table is not None:
                avg = normals_table.set_index([C.LOCATION, "doy"])[cols]
            else:
                src = df.loc[df[C.DATE] <= normals_cutoff]
                avg = src.groupby([src[C.LOCATION], src[C.DATE].dt.dayofyear])[cols].mean()
            future = df[C.DATE] > normals_cutoff
            key = pd.MultiIndex.from_arrays([df.loc[future, C.LOCATION], doy[future]])
            for col in cols:
                vals = avg[col].reindex(key).to_numpy(dtype="float64")
                df[col] = df[col].astype("float64")
                df.loc[future, col] = vals
            # eksik (lokasyon, doy) ciftleri icin (orn. kapsanmayan grid
            # hucresi) lokasyon medyanina, sonra genel medyana geriler
            for col in cols:
                df[col] = df.groupby(C.LOCATION)[col].transform(
                    lambda s: s.fillna(s.median())
                )
                df[col] = df[col].fillna(df[col].median())
            t = df["temperature_2m_mean"].astype("float64")
        else:
            for col in cols:
                df[col] = df[col].groupby([df[C.LOCATION], doy]).transform("mean")
            t = df["temperature_2m_mean"].astype("float64")

    df["cdd"] = np.maximum(0.0, t - CDD_BASE)
    df["hdd"] = np.maximum(0.0, HDD_BASE - t)
    df["temp_sq"] = t ** 2

    g = df.groupby(C.LOCATION, sort=False)
    for w in (3, 7, 14):
        df[f"cdd_sum_{w}"] = g["cdd"].transform(
            lambda s, w=w: s.rolling(w, min_periods=1).sum()
        )
        df[f"hdd_sum_{w}"] = g["hdd"].transform(
            lambda s, w=w: s.rolling(w, min_periods=1).sum()
        )
    df["temp_savgol"] = g["temperature_2m_mean"].transform(
        lambda s: pd.Series(_savgol_safe(s.to_numpy()), index=s.index)
    )
    df["temp_lag1"] = g["temperature_2m_mean"].shift(1)
    df["temp_lag2"] = g["temperature_2m_mean"].shift(2)
    df["temp_lag3"] = g["temperature_2m_mean"].shift(3)
    return df
