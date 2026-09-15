"""Akran / hiyerarsik hedef ozetleri.

Tavan olcumu (reports/02): lokasyon cold-start seviyesi icin bilgilendirici
degil, guc bandi ise oyle. Bu modul yine de her iki anahtari uretir; model
secimi ablasyonla yapilir. Hesap origin oncesi gozlemler uzerinden, yani
out-of-fold: dogrulama satirlari bu ortalamalara hic girmez.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.validation.folds import guc_band


def peer_tables(observed: pd.DataFrame, origin: pd.Timestamp) -> dict[str, pd.Series]:
    """Origin'e kadar grup medyanlari (log1p ve z uzayi)."""
    hist = observed.loc[observed[C.DATE] <= origin].copy()
    if hist.empty:
        return {}
    hist["log1p"] = np.log1p(hist[C.TARGET].clip(lower=0))
    hist["z"] = hist["log1p"] - np.log(hist[C.POWER].clip(lower=1).astype("float64"))
    hist["guc_band"] = guc_band(hist[C.POWER]).astype("string")
    hist["dow"] = hist[C.DATE].dt.dayofweek.astype("int8")
    hist["is_weekend"] = (hist["dow"] >= 5).astype("int8")
    pos = hist.loc[hist[C.TARGET] > 0]  # sifirlar ortalamayi asagi ceker

    tables: dict[str, pd.Series] = {
        "peer_global_z": pd.Series({"_": float(pos["z"].median())}),
        "peer_guc_z": pos.groupby("guc_band", observed=True)["z"].median(),
        "peer_guc_log": pos.groupby("guc_band", observed=True)["log1p"].median(),
        "peer_lok_z": pos.groupby(C.LOCATION, observed=True)["z"].median(),
        "peer_guc_wknd_z": pos.groupby(["guc_band", "is_weekend"], observed=True)["z"].median(),
        "peer_zero_guc": hist.groupby("guc_band", observed=True)["log1p"].agg(
            lambda s: float((np.expm1(s) <= 0).mean())
        ),
    }
    return tables


def attach_peer(df: pd.DataFrame, tables: dict[str, pd.Series]) -> pd.DataFrame:
    """Satir cercevesine akran kolonlarini yazar. `df` guc ve lokasyon icermeli."""
    out = df.copy()
    if "guc_band" not in out.columns:
        out["guc_band"] = guc_band(out[C.POWER]).astype("string")
    if "is_weekend" not in out.columns:
        out["is_weekend"] = (pd.to_datetime(out[C.DATE]).dt.dayofweek >= 5).astype("int8")

    glo = tables.get("peer_global_z")
    out["peer_global_z"] = float(glo.iloc[0]) if glo is not None and len(glo) else np.nan
    out["peer_guc_z"] = out["guc_band"].map(tables.get("peer_guc_z", pd.Series(dtype="float64")))
    out["peer_guc_log"] = out["guc_band"].map(tables.get("peer_guc_log", pd.Series(dtype="float64")))
    out["peer_lok_z"] = out[C.LOCATION].map(tables.get("peer_lok_z", pd.Series(dtype="float64")))
    out["peer_zero_guc"] = out["guc_band"].map(tables.get("peer_zero_guc", pd.Series(dtype="float64")))

    wk = tables.get("peer_guc_wknd_z")
    if wk is not None and not wk.empty:
        key = pd.MultiIndex.from_arrays([out["guc_band"], out["is_weekend"]])
        out["peer_guc_wknd_z"] = wk.reindex(key).to_numpy()
    else:
        out["peer_guc_wknd_z"] = np.nan
    return out
