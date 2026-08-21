"""Sizinti denetimi. Her ozellik seti egitime girmeden once bunlardan gecer.

Denetlenen kurallar (docs/science-superpowers/questions/... "Sizinti denetimi"):
  1. Hicbir ozellik origin sonrasi hedef bilgisi kullanamaz.
  2. Trafo gecmisi ozellikleri ufuk boyunca sabittir.
  3. shift/rolling daima groupby(tanim) sonrasi ve tarihe gore sirali.
  4. Maskelenen trafonun gecmisi egitim cercevesinde hic bulunmaz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C


class LeakageError(AssertionError):
    """Sizinti denetimi basarisiz."""


def assert_grouped_sorted(df: pd.DataFrame) -> None:
    """Cerceve (tanim, tarih) sirali mi.

    Siralanmamis bir cercevede shift/rolling bir trafonun gecmisini sessizce
    digerine sizdirir. Bu, denetimin en sik yakaladigi hata.
    """
    key = df[[C.ENTITY, C.DATE]]
    if not key.equals(key.sort_values([C.ENTITY, C.DATE])):
        raise LeakageError("cerceve (tanim, tarih) sirali degil")
    dup = df.duplicated([C.ENTITY, C.DATE]).sum()
    if dup:
        raise LeakageError(f"{dup} adet tekrarli (tanim, tarih) cifti var")


def assert_no_future_target(
    observed: pd.DataFrame, origin: pd.Timestamp, label: str = "egitim"
) -> None:
    """Egitim cercevesinde origin sonrasi hedef yok."""
    future = observed.loc[observed[C.DATE] > origin, C.TARGET].notna().sum()
    if future:
        raise LeakageError(
            f"{label} cercevesinde origin ({origin.date()}) sonrasi "
            f"{future} adet gozlenmis hedef var"
        )


def assert_history_features_constant(
    features: pd.DataFrame, columns: list[str], *, tol: float = 0.0
) -> None:
    """Trafo gecmisi ozellikleri ufuk boyunca sabit mi.

    Hedef geri beslemesi olmadigi icin bu ozellikler origin'de donar ve 122
    gun boyunca degismez. Ufuk icinde degisiyorlarsa ya hedef sizmis ya da
    rolling penceresi kaydirilmamis.
    """
    if not columns:
        return
    g = features.groupby(C.ENTITY, sort=False)[columns]
    spread = (g.max() - g.min()).abs().max()
    bad = spread[spread > tol]
    if len(bad):
        raise LeakageError(
            "ufuk boyunca sabit olmasi gereken ozellikler degisiyor: "
            + ", ".join(f"{c}={v:.6g}" for c, v in bad.items())
        )


def assert_masked_entities_absent(
    observed: pd.DataFrame, masked_entities: pd.Index, origin: pd.Timestamp
) -> None:
    """Maskelenen trafolarin origin oncesi hicbir satiri kalmamis."""
    if len(masked_entities) == 0:
        return
    left = observed.loc[
        observed[C.ENTITY].isin(set(masked_entities)) & (observed[C.DATE] <= origin)
    ]
    if len(left):
        raise LeakageError(
            f"{len(left)} satir: maskelenen trafolarin gecmisi silinmemis. "
            "Yalnizca lag NaN'lamak yetmez, satirlar dusurulmelidir."
        )


def assert_weather_not_future(
    weather: pd.DataFrame, date_col: str = C.DATE, lead_cols: list[str] | None = None
) -> None:
    """Hava ozellikleri t aninda veya oncesinde mi.

    Ileri bakan (t+1) hava ozelligi yasak. Bu kontrol, kolon adlarinda ileri
    kaydirma isareti ("lead", "fwd", "next") araniyor; isim disi bir ileri
    kaydirma otomatik yakalanamaz, o yuzden ozellik uretiminde disiplin sart.
    """
    suspicious = [
        c
        for c in (lead_cols or weather.columns)
        if any(tok in str(c).lower() for tok in ("lead", "fwd", "forward", "next"))
    ]
    if suspicious:
        raise LeakageError(f"ileri bakan hava ozelligi: {suspicious}")


def run_all_checks(
    *,
    observed: pd.DataFrame,
    origin: pd.Timestamp,
    masked_entities: pd.Index | None = None,
    features: pd.DataFrame | None = None,
    constant_columns: list[str] | None = None,
) -> dict[str, str]:
    """Tum denetimleri sirayla calistirir ve gecen kontrollerin ozetini doner."""
    results: dict[str, str] = {}

    assert_grouped_sorted(observed)
    results["grouped_sorted"] = "OK"

    assert_no_future_target(observed, origin)
    results["no_future_target"] = "OK"

    if masked_entities is not None:
        assert_masked_entities_absent(observed, masked_entities, origin)
        results["masked_absent"] = f"OK ({len(masked_entities)} trafo)"

    if features is not None and constant_columns:
        assert_history_features_constant(features, constant_columns)
        results["history_constant"] = f"OK ({len(constant_columns)} kolon)"

    return results


def target_correlation_probe(
    features: pd.DataFrame, target: pd.Series, threshold: float = 0.999
) -> pd.Series:
    """Hedefle neredeyse birebir korele ozellikleri isaretler.

    Sizintinin en kaba biciminin son savunma hatti: bir ozellik hedefle
    r > 0.999 korele ise ya hedefin kendisi ya da bire bir donusumu.
    """
    num = features.select_dtypes(include=[np.number])
    if num.empty:
        return pd.Series(dtype="float64")
    corr = num.corrwith(target.astype("float64")).abs()
    return corr[corr > threshold].sort_values(ascending=False)
