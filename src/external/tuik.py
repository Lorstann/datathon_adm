"""TÜİK ADNKS ilçe nüfus özellikleri.

Kaynak: data/external/tuik/ilce_nufus.csv
Join: panel `il` + ilçe anahtarı (İzmir: ilce, Manisa: bolge).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C

TUIK_PATH = C.EXTERNAL_DIR / "tuik" / "ilce_nufus.csv"

# TÜİK ad → panel UPPER anahtar
_NAME_ALIASES = {
    "IZMIR KEMALPASA": "KEMALPASA",
    "MANISA KOPRUBASI": "KOPRUBASI",
    "KOPRUBASI": "KOPRUBASI",
}


def _tr_upper_key(s: str) -> str:
    """Türkçe büyük harf + ASCII katlama (eşleme için)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    t = str(s).strip()
    # Türkçe I/İ
    t = t.replace("i", "İ").replace("ı", "I")
    t = t.upper()
    # fold diacritics to ASCII-ish matching panel strings stored as Unicode
    repl = {
        "İ": "I",
        "I": "I",
        "Ş": "S",
        "Ğ": "G",
        "Ü": "U",
        "Ö": "O",
        "Ç": "C",
        "Â": "A",
    }
    # Keep Turkish letters as in panel (which uses İ, Ş, etc.) — dual keys
    return t


def _fold(s: str) -> str:
    t = _tr_upper_key(s)
    for a, b in (
        ("İ", "I"),
        ("Ş", "S"),
        ("Ğ", "G"),
        ("Ü", "U"),
        ("Ö", "O"),
        ("Ç", "C"),
    ):
        t = t.replace(a, b)
    t = " ".join(t.split())
    return _NAME_ALIASES.get(t, t)


def load_nufus(path: Path = TUIK_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df["il_key"] = df["il"].map(_fold)
    df["ilce_key"] = df["ilce"].map(_fold)
    df["nufus"] = pd.to_numeric(df["nufus"], errors="coerce")
    df["yil"] = pd.to_numeric(df["yil"], errors="coerce").astype("Int64")
    return df.dropna(subset=["nufus", "yil"])


def district_key_from_row(il, bolge, ilce) -> str:
    """Panel satırından ilçe eşleme anahtarı (fold edilmiş)."""
    il_f = _fold(il)
    if il_f == "IZMIR":
        return _fold(ilce) if pd.notna(ilce) else ""
    # Manisa: bolge = ilçe
    return _fold(bolge) if pd.notna(bolge) else _fold(ilce)


def attach_nufus(
    df: pd.DataFrame,
    nufus: pd.DataFrame | None = None,
    *,
    asof_year: int | None = None,
) -> pd.DataFrame:
    """Satırlara log_nufus ve log_guc_per_nufus ekler.

    asof_year: verilmezse her satırın tarih yılına göre en güncel <= yıl nüfus.
    """
    nufus = load_nufus() if nufus is None else nufus
    out = df.copy()
    if "il" not in out.columns:
        from src.data.load import split_location

        out = out.join(split_location(out[C.LOCATION]))

    keys = [
        district_key_from_row(i, b, c)
        for i, b, c in zip(out["il"], out["bolge"], out["ilce"], strict=True)
    ]
    out["_dist_key"] = keys
    out["_il_key"] = out["il"].map(_fold)
    years = (
        pd.to_datetime(out[C.DATE]).dt.year.astype("int16")
        if asof_year is None
        else pd.Series(asof_year, index=out.index)
    )
    out["_y"] = years

    # her (il, ilce) için yıla göre asof merge
    nu = nufus.rename(columns={"il_key": "_il_key", "ilce_key": "_dist_key", "yil": "_ny"})
    nu = nu.sort_values("_ny")
    # merge_asof needs sorted by year within groups — do per-key lookup
    lookup = {}
    for (il_k, dist_k), g in nu.groupby(["_il_key", "_dist_key"], sort=False):
        g = g.sort_values("_ny")
        lookup[(il_k, dist_k)] = g[["_ny", "nufus"]].to_numpy()

    vals = np.full(len(out), np.nan)
    for i, (il_k, dist_k, y) in enumerate(
        zip(out["_il_key"], out["_dist_key"], out["_y"], strict=True)
    ):
        arr = lookup.get((il_k, dist_k))
        if arr is None or len(arr) == 0:
            continue
        # last year <= y
        ok = arr[arr[:, 0] <= y]
        if len(ok):
            vals[i] = ok[-1, 1]
        else:
            vals[i] = arr[0, 1]

    out["nufus"] = vals
    out["log_nufus"] = np.log(np.clip(out["nufus"], 1.0, None))
    if C.POWER in out.columns:
        out["log_guc_per_nufus"] = np.log(
            out[C.POWER].clip(lower=1).astype("float64")
        ) - out["log_nufus"]
    # turizm kıyısı bayrağı (plan P1 hafif)
    coastal = {"CESME", "SEFERIHISAR", "DIKILI", "FOCA", "URLA", "KARABURUN"}
    out["is_coastal_tourism"] = out["_dist_key"].isin(coastal).astype("int8")

    return out.drop(columns=["_dist_key", "_il_key", "_y"])
