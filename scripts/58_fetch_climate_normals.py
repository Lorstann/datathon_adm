"""Cok-yillik (2011-2025) iklim normali cek: 31 Mart kurali icin gercek `climatology`.

Panel-ici tek-yillik "gecen yil ayni gun" ortalamasi fold A gibi panel
baslangicina yakin origin'lerde tanimsiz (oncesinde hic Nisan-Temmuz yok).
Gercek sistemde de tek yillik normal zayif bir tahmin olurdu. Open-Meteo
archive API'si 1940'a kadar gidiyor; 15 yillik pencere sağlam bir normal
verir ve panel tarihinden bagimsizdir -- bu yuzden HER origin icin (fold A
dahil, gercek submission dahil) ayni tabloyu kullanabiliriz.

Kullanim: python scripts/58_fetch_climate_normals.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src import config as C
from src.external.coords import LOCATION_COORDS
from src.external.weather import DAILY_VARS, NORMALS_PATH, _fetch_one, _grid_key

START, END = "2016-01-01", "2025-12-31"


def main() -> None:
    C.WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    cell_to_locs: dict[tuple[float, float], list[str]] = {}
    for loc, (lat, lon) in LOCATION_COORDS.items():
        cell_to_locs.setdefault(_grid_key(lat, lon), []).append(loc)

    frames = []
    cells = list(cell_to_locs.items())
    t0 = time.time()
    for i, ((lat, lon), locs) in enumerate(cells):
        part = C.WEATHER_DIR / f"_normals_cell_{lat:.2f}_{lon:.2f}.parquet"
        if part.exists():
            daily_df = pd.read_parquet(part)
        else:
            try:
                payload = _fetch_one(lat, lon, START, END, retries=2)
            except RuntimeError as exc:
                print(f"  hucre {i+1}/{len(cells)} basarisiz, atlandi: {exc}", flush=True)
                continue
            daily_df = pd.DataFrame(payload["daily"])
            daily_df["tarih"] = pd.to_datetime(daily_df["time"])
            daily_df = daily_df.drop(columns=["time"])
            daily_df.to_parquet(part, index=False)
            time.sleep(8.0)
        for loc in locs:
            chunk = daily_df.copy()
            chunk[C.LOCATION] = loc
            frames.append(chunk)
        print(f"  hucre {i+1}/{len(cells)} {lat:.2f},{lon:.2f} -> {len(locs)} lokasyon "
              f"({time.time()-t0:.0f}s)", flush=True)

    raw = pd.concat(frames, ignore_index=True)
    raw["doy"] = raw["tarih"].dt.dayofyear
    cols = [c for c in DAILY_VARS if c in raw.columns]
    normals = raw.groupby([C.LOCATION, "doy"])[cols].mean().reset_index()
    normals.to_parquet(NORMALS_PATH, index=False)
    print(f"yazildi: {NORMALS_PATH}  {len(normals):,} satir ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
