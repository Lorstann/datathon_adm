"""RMSLE ve segment bazli raporlama.

RMSLE, log1p uzayinda RMSE'nin kendisidir:

    RMSLE(p, a) = sqrt(mean((log1p(p) - log1p(a))^2)) = RMSE(log1p(p), log1p(a))

Bu kimlik iki sonuc dogurur ve ikisi de projenin her yerinde gecerli:
  1. log1p hedefi uzerinde L2 ile egitmek dogrudan metrigi minimize eder.
  2. Geri donusumde smearing / Jensen duzeltmesi UYGULANMAMALIDIR. RMSLE'nin
     Bayes-optimal tahmini E[log1p(y)|x] oldugu icin, daima >= 1 olan smearing
     carpani tahmini optimumdan uzaklastirir. Gerekce: docs/prior-work.md 8.1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C


def _to_array(x) -> np.ndarray:
    return np.asarray(x, dtype="float64").ravel()


def rmsle(y_true, y_pred) -> float:
    """Ham olcekte RMSLE. Negatif tahminler 0 sayilir (yarisma kurali)."""
    a = _to_array(y_true)
    p = np.clip(_to_array(y_pred), 0.0, None)
    if a.shape != p.shape:
        raise ValueError(f"sekil uyusmazligi: {a.shape} vs {p.shape}")
    if np.any(a < 0):
        raise ValueError("gercek degerlerde negatif var; log1p tanimsiz")
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(a)) ** 2)))


def rmsle_from_log(z_true, z_pred) -> float:
    """Zaten log1p uzayindaki degerler icin RMSE.

    Model log uzayinda calistiginda geri donusum yapmadan skorlamak icin.
    Tahmin log1p(0) = 0 tabaninda kirpilir, cunku ham olcekte 0'a kirpmanin
    log uzayindaki karsiligi budur.
    """
    a = _to_array(z_true)
    p = np.clip(_to_array(z_pred), 0.0, None)
    if a.shape != p.shape:
        raise ValueError(f"sekil uyusmazligi: {a.shape} vs {p.shape}")
    return float(np.sqrt(np.mean((p - a) ** 2)))


def segment_report(
    y_true,
    y_pred,
    segment: pd.Series,
    *,
    is_cold: pd.Series | None = None,
    blend_cold_rate: float = C.TEST_COLD_ROW_RATE,
) -> dict[str, float]:
    """Toplam, segment bazli ve gercek test oranlariyla harmanlanmis RMSLE.

    `segment` gecmis uzunlugu kirilimi (bkz. history_segment).
    `is_cold` verilirse warm/cold ayrimi ve harman hesaplanir. Harman, gercek
    test setindeki satir oranlarini kullanir; boylece dogrulama setindeki cold
    orani hedeften sapsa bile raporlanan sayi test'i temsil eder.
    """
    a = _to_array(y_true)
    p = np.clip(_to_array(y_pred), 0.0, None)
    sq = (np.log1p(p) - np.log1p(a)) ** 2

    out: dict[str, float] = {"rmsle_all": float(np.sqrt(np.mean(sq)))}

    seg = pd.Series(np.asarray(segment).ravel(), index=range(len(sq)))
    for label, idx in seg.groupby(seg, observed=True).groups.items():
        pos = np.asarray(idx, dtype="int64")
        out[f"rmsle_seg_{label}"] = float(np.sqrt(np.mean(sq[pos])))
        out[f"n_seg_{label}"] = float(pos.size)

    if is_cold is not None:
        cold = np.asarray(is_cold).ravel().astype(bool)
        if cold.any():
            out["rmsle_cold"] = float(np.sqrt(np.mean(sq[cold])))
        if (~cold).any():
            out["rmsle_warm"] = float(np.sqrt(np.mean(sq[~cold])))
        if cold.any() and (~cold).any():
            # Harman kare-hata duzeyinde yapilir, RMSLE duzeyinde degil:
            # RMSLE'lerin agirlikli ortalamasi metrigi yanlis birlestirir.
            mse = blend_cold_rate * np.mean(sq[cold]) + (1 - blend_cold_rate) * np.mean(
                sq[~cold]
            )
            out["rmsle_blend"] = float(np.sqrt(mse))
    return out


def summarize_folds(rows: pd.DataFrame, metric: str = "rmsle_blend") -> pd.Series:
    """Fold'lar arasi ortalama ve standart sapma.

    Model secim kurali: ortalama farki fold'lar arasi std'den kucukse fark
    anlamli sayilmaz (M5 birincisinin kriteri, docs/prior-work.md 9.2).
    `ddof=0` kullanilir: uc fold'un tamami elimizde, populasyondan ornek degil.
    """
    vals = rows[metric].astype("float64")
    return pd.Series(
        {
            "mean": vals.mean(),
            "std": vals.std(ddof=0),
            "min": vals.min(),
            "max": vals.max(),
            "n_folds": float(len(vals)),
        }
    )
