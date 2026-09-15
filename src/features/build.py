"""Ozellik matrisini birlestirir."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.features.calendar import calendar_features
from src.features.history import entity_history_features
from src.features.peer import attach_peer, peer_tables
from src.validation.folds import guc_band

# opsiyonel dis veri (dosya yoksa sessizce atlanir)
try:
    from src.external.tuik import TUIK_PATH, attach_nufus, load_nufus
except Exception:  # pragma: no cover
    TUIK_PATH = None
    attach_nufus = None
    load_nufus = None
try:
    from src.external.epias import EPIAS_PATH, attach_epias, load_epias
except Exception:  # pragma: no cover
    EPIAS_PATH = None
    attach_epias = None
    load_epias = None

HISTORY_COLS_CONSTANT = [
    "hist_n",
    "hist_mean",
    "hist_median",
    "hist_std",
    "hist_mean_z",
    "hist_zero_rate",
    "hist_last_zero_streak",
    "hist_last_gap",
    "hist_coverage",
    "hist_mean_7",
    "hist_pos_mean_7",
    "hist_pos_mean_28",
    "hist_pos_mean_91",
    "hist_pos_mean",
    "hist_mean_14",
    "hist_mean_21",
    "hist_mean_28",
    "hist_mean_91",
    "hist_mean_365",
    "hist_yoy_mean",
    "hist_yoy_gap",
    "hist_hijri_mean",
    "hist_mom_7_28",
    "hist_mom_7_91",
    "hist_mom_28_91",
    "hist_q25_91",
    "hist_q75_91",
    "hist_iqr_91",
    "hist_cdd_slope",
    "hist_hdd_slope",
    "hist_trend",
    "hist_weekend_drop",
]


def transductive_features(rows: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """test.csv'den tureyen panel uyeligi. Mesru ama operasyonel degil."""
    s = static.reindex(rows[C.ENTITY].to_numpy())
    age = (rows[C.DATE].to_numpy() - s["panel_ilk_tarih"].to_numpy()).astype(
        "timedelta64[D]"
    ).astype("float64")
    out = pd.DataFrame(
        {
            "devreye_alinma_yasi": age,
            "test_gun_sayisi": s["test_gun_sayisi"].to_numpy(),
            "lokasyon_derinlik": s["lokasyon_derinlik"].to_numpy(),
            "il": s["il"].to_numpy(),
            "bolge": s["bolge"].to_numpy(),
            "ilce": s["ilce"].to_numpy(),
        },
        index=rows.index,
    )
    return out


def add_horizon_index(rows: pd.DataFrame, origin: pd.Timestamp) -> pd.Series:
    return (rows[C.DATE] - origin).dt.days.astype("int16")


def assemble(
    rows: pd.DataFrame,
    *,
    origin: pd.Timestamp,
    history: pd.DataFrame,
    weather: pd.DataFrame | None,
    static: pd.DataFrame,
    peer: dict[str, pd.Series],
    weather_cols: list[str] | None = None,
    include_history: bool = True,
    use_external: bool = False,
) -> pd.DataFrame:
    """Satir cercevesine (trafo-gun) tum ozellik ailelerini ekler.

    `rows` en az tanim, guc, tarih, lokasyon icermeli.
    use_external: TÜİK/EPİAŞ (LB kazanan C'de kapaliydi; varsayilan False).
    """
    out = rows.copy()
    cal = calendar_features(out[C.DATE])
    out = pd.concat([out, cal], axis=1)
    out["horizon_day"] = add_horizon_index(out, origin)
    out["guc_band"] = guc_band(out[C.POWER]).astype("string")
    out["log_guc"] = np.log(out[C.POWER].clip(lower=1).astype("float64"))

    hist_keep = [c for c in history.columns if c not in (C.POWER, C.LOCATION, "guc", "guc_band", "log_guc")]
    if include_history and hist_keep:
        mapped = history[hist_keep].reindex(out[C.ENTITY].to_numpy())
        mapped.index = out.index
        out = pd.concat([out, mapped], axis=1)

    # NOT: origin'den bagimsiz oranlar (`hist_n_orani`, `yas_orani`) denendi ve
    # REDDEDILDI (reports/46_origin_invariant.md). Test degerlerinin %27-33'u
    # egitim araliginin disinda kaliyor ve oranlar bunu 0.04'e indiriyordu, ama
    # CV kazandirmadi: agacin son esikte doymasi zarar vermiyormus.
    out = attach_peer(out, peer)
    # Varligin lokasyon akranlarina gore konumu. Seviyenin kendisi degil,
    # akranlardan sapmasi tasiniyor: lokasyon ortak kaymasi zaten peer
    # kolonunda, model ikisini ayirabiliyor.
    if "peer_guc_log" in out.columns and "hist_mean_7" in out.columns:
        out["hist_rel_peer"] = out["hist_mean_7"] - out["peer_guc_log"]
    trans = transductive_features(out, static)
    out = pd.concat([out, trans], axis=1)
    out["lokasyon_cat"] = out[C.LOCATION].astype("string")

    if use_external:
        if attach_nufus is not None and TUIK_PATH is not None and TUIK_PATH.exists():
            out = attach_nufus(out)
        if attach_epias is not None and EPIAS_PATH is not None and EPIAS_PATH.exists():
            out = attach_epias(out)

    if weather is not None:
        wcols = weather_cols or [
            c
            for c in weather.columns
            if c not in (C.LOCATION, C.DATE, "lat_snap", "lon_snap", "elevation")
        ]
        w = weather[[C.LOCATION, C.DATE, *wcols]]
        out = out.merge(w, on=[C.LOCATION, C.DATE], how="left")

    # etkilesimler (az sayida, agac zaten boluyor ama dogrusal iliskiyi kolaylastirir)
    if "cdd" in out.columns:
        out["cdd_x_logguc"] = out["cdd"] * out["log_guc"]
        out["hdd_x_logguc"] = out["hdd"] * out["log_guc"]
        # NOT: `cdd_x_slope` / `hdd_x_slope` carpimlari denendi ve reddedildi
        # (reports/33_cdd_interaction.md). Ham egimler ozellik olarak kaliyor.
    out["weekend_x_logguc"] = out["is_weekend"] * out["log_guc"]
    out["holiday_x_logguc"] = out["is_holiday"] * out["log_guc"]
    return out


FEATURE_EXCLUDE = {
    "id",
    C.ENTITY,
    C.DATE,
    C.TARGET,
    C.LOCATION,
    "is_test",
    "ilce",  # Manisa'da NA, lokasyon zaten var
    "_level",  # seviye+sekil hedefinde ofset, ozellik degil
    "_cold_level",
    "_origin",
}


def model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Ogreniciye gidecek X ve kategorik kolon adlari."""
    cols = [c for c in df.columns if c not in FEATURE_EXCLUDE]
    X = df[cols].copy()
    drop_dt = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
    if drop_dt:
        X = X.drop(columns=drop_dt)
    cat_cols = [
        c
        for c in ("il", "bolge", "guc_band", "lokasyon_cat")
        if c in X.columns
    ]
    # lokasyon ham kolonu FEATURE_EXCLUDE'da; kategori olarak guc_band/il/bolge
    for c in cat_cols:
        X[c] = X[c].astype("category")
    return X, cat_cols
