"""Deney kayit defteri (workspace kurali 14).

Her satir tek bir (konfigurasyon, fold) cifti. Ortalama ve std ayri
hesaplanir, cunku model secim kurali ikisine birlikte bakiyor.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from src import config as C

COLUMNS = [
    "exp_id",
    "timestamp",
    "model",
    "target",
    "feature_set",
    "fold",
    "rmsle_all",
    "rmsle_warm",
    "rmsle_cold",
    "rmsle_blend",
    "seed",
    "lb_public",
    "notes",
]


def log_experiment(
    *,
    exp_id: str,
    model: str,
    target: str,
    feature_set: str,
    fold: str,
    metrics: dict[str, float],
    seed: int = C.SEED,
    lb_public: float | None = None,
    notes: str = "",
    path: Path = C.EXPERIMENTS_CSV,
) -> None:
    """experiments.csv'ye tek satir ekler. Dosya yoksa basligiyla olusturur."""
    row = {
        "exp_id": exp_id,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "target": target,
        "feature_set": feature_set,
        "fold": fold,
        "rmsle_all": metrics.get("rmsle_all"),
        "rmsle_warm": metrics.get("rmsle_warm"),
        "rmsle_cold": metrics.get("rmsle_cold"),
        "rmsle_blend": metrics.get("rmsle_blend"),
        "seed": seed,
        "lb_public": lb_public,
        "notes": notes,
    }
    df = pd.DataFrame([row], columns=COLUMNS)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")


def load_experiments(path: Path = C.EXPERIMENTS_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, encoding="utf-8")


def leaderboard(
    metric: str = "rmsle_blend", path: Path = C.EXPERIMENTS_CSV
) -> pd.DataFrame:
    """Konfigurasyon basina fold'lar arasi ortalama ve std tablosu.

    Siralama ortalamaya gore, ama std kolonu daima gosterilir: ortalama farki
    std'den kucukse fark anlamli sayilmaz.
    """
    df = load_experiments(path)
    if df.empty:
        return df
    g = df.groupby(["exp_id", "model", "target", "feature_set"], dropna=False)[metric]
    out = g.agg(mean="mean", std=lambda s: s.std(ddof=0), n_folds="size")
    return out.sort_values("mean").reset_index()
