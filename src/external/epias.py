"""EPİAŞ il-aylık tüketim özellikleri.

Kaynak: data/external/epias/consumption_quantity_izmir_manisa.csv
Sızıntı: tahmin ayı için lag-1 ay + YoY (aynı ay önceki yıl) kullanılır;
aynı ay gerçekleşeni özellik olarak alınmaz (yayımlanma gecikmesi).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C

EPIAS_PATH = C.EXTERNAL_DIR / "epias" / "consumption_quantity_izmir_manisa.csv"

PROFILES = (
    "Mesken",
    "Sanayi",
    "Ticarethane",
    "Tarımsal Sulama",
    "Aydınlatma",
    "Diğer",
)


def _fold_il(s: str) -> str:
    t = str(s).strip().upper().replace("İ", "I").replace("Ş", "S")
    for a, b in (("Ğ", "G"), ("Ü", "U"), ("Ö", "O"), ("Ç", "C")):
        t = t.replace(a, b)
    return t


def load_epias(path: Path = EPIAS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df["il_key"] = df["il"].map(_fold_il)
    df["period"] = pd.PeriodIndex(df["period"].astype(str), freq="M")
    df["miktar"] = pd.to_numeric(df["miktar"], errors="coerce").fillna(0.0)
    # wide: il x period x profile
    wide = (
        df.pivot_table(
            index=["il_key", "period"],
            columns="profil_grubu",
            values="miktar",
            aggfunc="sum",
        )
        .fillna(0.0)
        .reset_index()
    )
    for p in PROFILES:
        if p not in wide.columns:
            wide[p] = 0.0
    wide["epias_total"] = wide[list(PROFILES)].sum(axis=1)
    for p in PROFILES:
        wide[f"epias_share_{p}"] = np.where(
            wide["epias_total"] > 0, wide[p] / wide["epias_total"], 0.0
        )
    wide["log_epias_total"] = np.log1p(wide["epias_total"])
    return wide


def attach_epias(df: pd.DataFrame, epias: pd.DataFrame | None = None) -> pd.DataFrame:
    """Lag-1 ay ve YoY (12 ay önce) EPİAŞ toplam/pay özellikleri."""
    epias = load_epias() if epias is None else epias
    out = df.copy()
    if "il" not in out.columns:
        from src.data.load import split_location

        out = out.join(split_location(out[C.LOCATION]))

    out["_il_key"] = out["il"].map(_fold_il)
    per = pd.to_datetime(out[C.DATE]).dt.to_period("M")
    out["_period"] = per
    out["_lag1"] = per - 1
    out["_yoy"] = per - 12

    cols = ["log_epias_total", "epias_share_Mesken", "epias_share_Sanayi", "epias_share_Ticarethane", "epias_share_Tarımsal Sulama"]
    # rename for safe column names
    rename_lag = {
        "log_epias_total": "log_epias_total_lag1",
        "epias_share_Mesken": "epias_share_mesken_lag1",
        "epias_share_Sanayi": "epias_share_sanayi_lag1",
        "epias_share_Ticarethane": "epias_share_ticaret_lag1",
        "epias_share_Tarımsal Sulama": "epias_share_tarim_lag1",
    }
    lag = epias[["il_key", "period", *cols]].rename(columns={"il_key": "_il_key", "period": "_lag1", **rename_lag})
    yoy_map = {
        "log_epias_total": "log_epias_total_yoy",
        "epias_share_Mesken": "epias_share_mesken_yoy",
    }
    yoy = epias[["il_key", "period", "log_epias_total", "epias_share_Mesken"]].rename(
        columns={"il_key": "_il_key", "period": "_yoy", **yoy_map}
    )

    out = out.merge(lag, on=["_il_key", "_lag1"], how="left")
    out = out.merge(yoy, on=["_il_key", "_yoy"], how="left")
    # YoY oran: bu ayın lag1'i / 12 ay önceki lag1 yaklaşık; basit: total_lag1 / total_yoy
    out["epias_yoy_ratio"] = np.where(
        out["log_epias_total_yoy"].notna() & (out["log_epias_total_yoy"] > 0),
        np.expm1(out["log_epias_total_lag1"].fillna(0)) / np.expm1(out["log_epias_total_yoy"]),
        np.nan,
    )
    return out.drop(columns=["_il_key", "_period", "_lag1", "_yoy"])
