"""Merkezi konfigurasyon: yollar, sabitler, seed, fold tanimlari.

Tum rastgelelik SEED'den turer. Yarisma yapisina ait sabitler (ufuk uzunlugu,
cold-start oranlari, fold pencereleri) burada tek yerde tutulur.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SEED = 42

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
EXTERNAL_DIR = ROOT / "data" / "external"
WEATHER_DIR = EXTERNAL_DIR / "weather"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SUBMISSIONS_DIR = ROOT / "submissions"
MODELS_DIR = ROOT / "models"

TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DIR / "sample_submission.csv"

EXPERIMENTS_CSV = ROOT / "experiments.csv"

ENTITY = "tanim"
DATE = "tarih"
TARGET = "tuketim"
POWER = "guc"
LOCATION = "lokasyon"

# Yarisma yapisi
TRAIN_START = pd.Timestamp("2025-01-01")
TRAIN_END = pd.Timestamp("2026-03-31")
TEST_START = pd.Timestamp("2026-04-01")
TEST_END = pd.Timestamp("2026-07-31")
HORIZON_DAYS = 122

# Gercek test setinde olculen cold-start oranlari. Validasyon maskelemesi
# bu oranlari hedefler (bkz. docs/prior-work.md 10.2).
TEST_COLD_ENTITY_RATE = 0.2877  # 2024 / 7036
TEST_COLD_ROW_RATE = 0.2216  # 158369 / 714688


@dataclass(frozen=True)
class Fold:
    """Bir sabit-origin validasyon fold'u.

    origin: egitim verisinin son gunu (dahil). Bu tarihten sonraki hicbir
    hedef bilgisi ozellik uretiminde kullanilamaz.
    """

    name: str
    origin: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    weight: float

    @property
    def n_valid_days(self) -> int:
        return (self.valid_end - self.valid_start).days + 1


FOLDS: tuple[Fold, ...] = (
    Fold(
        name="A_mevsim",
        origin=pd.Timestamp("2025-03-31"),
        valid_start=pd.Timestamp("2025-04-01"),
        valid_end=pd.Timestamp("2025-07-31"),
        weight=2.0,
    ),
    Fold(
        name="B_guncel",
        origin=pd.Timestamp("2025-11-30"),
        valid_start=pd.Timestamp("2025-12-01"),
        valid_end=pd.Timestamp("2026-03-31"),
        weight=1.0,
    ),
    Fold(
        name="C_ara",
        origin=pd.Timestamp("2025-09-30"),
        valid_start=pd.Timestamp("2025-10-01"),
        valid_end=pd.Timestamp("2026-01-31"),
        weight=1.0,
    ),
)

# Gercek tahmin icin origin: train'in son gunu.
SUBMISSION_FOLD = Fold(
    name="submission",
    origin=TRAIN_END,
    valid_start=TEST_START,
    valid_end=TEST_END,
    weight=1.0,
)

# Gecmis uzunlugu segment sinirlari (gun). Her deneyde bu kirilimda rapor.
HISTORY_SEGMENT_EDGES = (0, 1, 30, 90, 270)
HISTORY_SEGMENT_LABELS = (
    "0_cold",
    "1_1-29g",
    "2_30-89g",
    "3_90-269g",
    "4_270g+",
)


def ensure_dirs() -> None:
    for d in (
        INTERIM_DIR,
        PROCESSED_DIR,
        EXTERNAL_DIR,
        WEATHER_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        SUBMISSIONS_DIR,
        MODELS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
