"""Open-Meteo ERA5-Land gunluk arsivini ceker ve cache'ler.

Kullanim: python scripts/04_fetch_weather.py
Lisans: CC-BY 4.0, atif docs/external-data.md icinde.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.external.coords import LOCATION_COORDS
from src.external.weather import CACHE_PATH, fetch_and_cache


def main() -> None:
    print(f"cekilecek lokasyon: {len(LOCATION_COORDS)}")
    df = fetch_and_cache()
    print(f"satir: {len(df):,}  kolon: {list(df.columns)}")
    print(f"tarih: {df['tarih'].min().date()} - {df['tarih'].max().date()}")
    print(f"cache: {CACHE_PATH}")
    n_grid = df[["lat_snap", "lon_snap"]].drop_duplicates().shape[0]
    print(f"ERA5-Land tekil hucre: {n_grid} / {df['lokasyon'].nunique()} lokasyon")


if __name__ == "__main__":
    main()
