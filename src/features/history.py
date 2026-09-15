"""Origin'e kadar donmus trafo gecmisi ozetleri.

Hedef geri beslemesi olmadigi icin bu ozellikler ufuk boyunca SABITTIR.
Hesaplama daima `tarih <= origin` satirlari uzerinden, groupby(tanim) ile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.validation.folds import guc_band


def entity_history_features(
    observed: pd.DataFrame,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    """Index = tanim. Origin'e kadar (dahil) ozetler.

    Maskeleme uygulandiktan sonra cagrilirsa cold trafolar tabloda yoktur;
    birlestirmede NaN kalir ve agac modeli NaN'i dallandirir.
    """
    hist = observed.loc[observed[C.DATE] <= origin, :].copy()
    if hist.empty:
        return pd.DataFrame(index=pd.Index([], name=C.ENTITY))

    hist["log1p"] = np.log1p(hist[C.TARGET].clip(lower=0))
    hist["z"] = hist["log1p"] - np.log(hist[C.POWER].clip(lower=1).astype("float64"))
    hist["is_zero"] = (hist[C.TARGET] <= 0).astype("int8")
    hist["dow"] = hist[C.DATE].dt.dayofweek.astype("int8")

    g = hist.groupby(C.ENTITY, sort=False)
    out = g.agg(
        hist_n=(C.DATE, "nunique"),
        hist_first=(C.DATE, "min"),
        hist_last=(C.DATE, "max"),
        hist_mean=("log1p", "mean"),
        hist_median=("log1p", "median"),
        hist_std=("log1p", "std"),
        hist_mean_z=("z", "mean"),
        hist_zero_rate=("is_zero", "mean"),
        guc=(C.POWER, "first"),
        lokasyon=(C.LOCATION, "first"),
    )
    out["hist_std"] = out["hist_std"].fillna(0.0)
    out["hist_span"] = (out["hist_last"] - out["hist_first"]).dt.days + 1
    out["hist_coverage"] = out["hist_n"] / out["hist_span"].clip(lower=1)
    out["hist_last_gap"] = (origin - out["hist_last"]).dt.days
    out["log_guc"] = np.log(out["guc"].clip(lower=1).astype("float64"))
    out["guc_band"] = guc_band(out["guc"]).astype("string")

    last = g.tail(1)
    out["hist_last_zero_streak"] = 0
    # son gozlem sifir mi: streak icin kaba vekil (tam streak pahali).
    # Tam streak: sondan geriye tarama, vektorize etmek icin kumulatif.
    hist_sorted = hist.sort_values([C.ENTITY, C.DATE])
    flip = hist_sorted["is_zero"].eq(0)
    run_id = flip.groupby(hist_sorted[C.ENTITY]).cumsum()
    streak = hist_sorted["is_zero"].groupby([hist_sorted[C.ENTITY], run_id]).cumsum()
    last_streak = streak.groupby(hist_sorted[C.ENTITY]).tail(1)
    last_streak.index = hist_sorted.groupby(C.ENTITY).tail(1)[C.ENTITY]
    out["hist_last_zero_streak"] = last_streak.reindex(out.index).fillna(0).astype("int32")

    for days, name in ((7, "7"), (14, "14"), (21, "21"), (28, "28"), (91, "91"), (365, "365")):
        sl = hist.loc[hist[C.DATE] > origin - pd.Timedelta(days=days)]
        gg = sl.groupby(C.ENTITY, sort=False)
        out[f"hist_mean_{name}"] = gg["log1p"].mean()
        out[f"hist_med_{name}"] = gg["log1p"].median()
        out[f"hist_zero_{name}"] = gg["is_zero"].mean()

    # Yalniz SIFIR OLMAYAN gunlerin ortalamasi. `reports/02` karar 3: regresor
    # sifir olmayan satirlarda egitilmeli, sifir olasiligi ayri modellenmeli.
    # Seviye de o mimaride sifirsiz olmali, yoksa (1-p) carpani sifirlari iki
    # kez saymis olur: hist_mean zaten ~ (1-p_gecmis) * pozitif_seviye.
    pos = hist.loc[hist["is_zero"] == 0]
    for days, name in ((7, "7"), (28, "28"), (91, "91")):
        sl = pos.loc[pos[C.DATE] > origin - pd.Timedelta(days=days)]
        out[f"hist_pos_mean_{name}"] = sl.groupby(C.ENTITY, sort=False)["log1p"].mean()
    out["hist_pos_mean"] = pos.groupby(C.ENTITY, sort=False)["log1p"].mean()

    dow_mean = (
        hist.groupby([C.ENTITY, "dow"], sort=False)["log1p"].mean().unstack("dow")
    )
    for k in range(7):
        out[f"hist_dow_{k}"] = dow_mean[k] if k in dow_mean.columns else np.nan
    wknd = dow_mean.reindex(columns=[5, 6]).mean(axis=1)
    wkdy = dow_mean.reindex(columns=[0, 1, 2, 3, 4]).mean(axis=1)
    out["hist_weekend_drop"] = wknd - wkdy

    # trend: cov(t, y) / var(t)  —  en az 14 gozlem
    hist["t_idx"] = (hist[C.DATE] - g[C.DATE].transform("min")).dt.days.astype("float64")
    n = g["log1p"].transform("size")
    mx = g["t_idx"].transform("mean")
    my = g["log1p"].transform("mean")
    cov = ((hist["t_idx"] - mx) * (hist["log1p"] - my)).groupby(hist[C.ENTITY]).sum()
    var = ((hist["t_idx"] - mx) ** 2).groupby(hist[C.ENTITY]).sum()
    trend = cov / var.replace(0, np.nan)
    trend = trend.where(g.size() >= 14, np.nan)
    out["hist_trend"] = trend

    def _window_mean(start_delta: int, length: int, col: str) -> pd.Series:
        a = origin - pd.Timedelta(days=start_delta)
        b = a + pd.Timedelta(days=length)
        sl = hist.loc[(hist[C.DATE] > a) & (hist[C.DATE] <= b)]
        return sl.groupby(C.ENTITY, sort=False)["log1p"].mean()

    out["hist_yoy_mean"] = _window_mean(365, C.HORIZON_DAYS, "yoy")
    out["hist_hijri_mean"] = _window_mean(355, C.HORIZON_DAYS, "hijri")

    # Momentum / ortalamaya donus. Seviye kaymasinin en guclu tek yordayicisi
    # kisa pencerenin uzun pencereye gore konumu: corr(kayma, m7-m91) = -0.31,
    # yani son gunler uzun ortalamanin uzerindeyse ufukta geri geliyor.
    # Agac eksen-hizali boldugu icin farki kendisi kurmasi zor; acikca veriyoruz.
    for a, b in (("7", "28"), ("7", "91"), ("28", "91"), ("7", "365")):
        ca, cb = f"hist_mean_{a}", f"hist_mean_{b}"
        if ca in out.columns and cb in out.columns:
            out[f"hist_mom_{a}_{b}"] = out[ca] - out[cb]

    # Ayni pencerenin gecen yilki hali, varligin genel seviyesine gore. Ufuk
    # Nis-Tem; gonderim origin'inde trafolarin %54'unde bu pencere dolu.
    out["hist_yoy_gap"] = out["hist_yoy_mean"] - out["hist_mean"]

    # Son 91 gunun ceyreklikleri: seviye ortalamadan cok medyan/IQR ile
    # tanimlanan trafolarda saglam bir olcek verir.
    recent = hist.loc[hist[C.DATE] > origin - pd.Timedelta(days=91)]
    rq = recent.groupby(C.ENTITY, sort=False)["log1p"].quantile([0.25, 0.75]).unstack()
    if not rq.empty:
        out["hist_q25_91"] = rq[0.25]
        out["hist_q75_91"] = rq[0.75]
        out["hist_iqr_91"] = rq[0.75] - rq[0.25]

    # Hafta gunu profili varligin kendi seviyesine gore (mutlak degil goreli):
    # sanayi/konut imzasi buradan okunuyor.
    for k in range(7):
        out[f"hist_dow_rel_{k}"] = out[f"hist_dow_{k}"] - out["hist_mean"]

    # v3 denendi ve REDDEDILDI (reports/31_features_v3.md, 33_cdd_interaction.md):
    # sifir blogu geometrisi, aylik seviye sacilimi, akran momentumu, tatil
    # duyarliligi ve trafo-basi hava katsayisi CARPIMLARI. Fold A'da kazandirip
    # B/C'de kaybettirdi; fold A'nin egitim cercevesi kis-agirlikli ve 45 gunluk
    # ufuklu oldugu icin oradaki kazanc yapilandirma artifakti sayildi.
    return out


def entity_weather_sensitivity(
    observed: pd.DataFrame,
    origin: pd.Timestamp,
    weather: pd.DataFrame | None,
) -> pd.DataFrame:
    """Trafo basina sicaklik duyarliligi (log1p birimi / derece-gun).

    Ufuk Nisan-Temmuz, yani sogutma rampasinin tam ustu. Ayni lokasyondaki iki
    trafo ayni havayi gorur ama tepkileri farklidir (sanayi vs konut); bu
    katsayi o farki tasir. Egim origin oncesi gozlemlerden en kucuk kareler:
    cov(dd, log1p) / var(dd).
    """
    idx = pd.Index([], name=C.ENTITY)
    if weather is None:
        return pd.DataFrame(index=idx)
    hist = observed.loc[observed[C.DATE] <= origin, [C.ENTITY, C.DATE, C.LOCATION, C.TARGET]]
    if hist.empty:
        return pd.DataFrame(index=idx)

    cols = [c for c in ("cdd", "hdd") if c in weather.columns]
    if not cols:
        return pd.DataFrame(index=idx)
    w = weather[[C.LOCATION, C.DATE, *cols]]
    h = hist.merge(w, on=[C.LOCATION, C.DATE], how="inner")
    if h.empty:
        return pd.DataFrame(index=idx)
    h = h.loc[h[C.TARGET] > 0]
    if h.empty:
        return pd.DataFrame(index=idx)
    h["y"] = np.log1p(h[C.TARGET])

    g = h.groupby(C.ENTITY, sort=False)
    out = pd.DataFrame(index=g.size().index)
    my = g["y"].transform("mean")
    for c in cols:
        mx = g[c].transform("mean")
        num = ((h[c] - mx) * (h["y"] - my)).groupby(h[C.ENTITY], sort=False).sum()
        den = ((h[c] - mx) ** 2).groupby(h[C.ENTITY], sort=False).sum()
        slope = num / den.replace(0, np.nan)
        # 30 gozlemin altinda egim gurultuden ibaret
        out[f"hist_{c}_slope"] = slope.where(g.size() >= 30, np.nan)
    return out


def add_lagged_history(df: pd.DataFrame) -> pd.DataFrame:
    """Egitim satirlari icin shift(1) gecmisi. Mevcut satirin hedefi kullanilmaz."""
    out = df.copy()
    out["_y"] = np.log1p(out[C.TARGET].clip(lower=0))
    out["_z"] = out["_y"] - np.log(out[C.POWER].clip(lower=1).astype("float64"))
    out["_zero"] = (out[C.TARGET] <= 0).astype("float64")
    g = out.groupby(C.ENTITY, sort=False)

    def _exp_mean(col: str) -> pd.Series:
        prev = g[col].shift(1)
        return prev.groupby(out[C.ENTITY], sort=False).expanding(min_periods=1).mean().reset_index(
            level=0, drop=True
        )

    def _roll_mean(col: str, win: int) -> pd.Series:
        prev = g[col].shift(1)
        return prev.groupby(out[C.ENTITY], sort=False).rolling(win, min_periods=1).mean().reset_index(
            level=0, drop=True
        )

    out["hist_mean"] = _exp_mean("_y")
    out["hist_mean_z"] = _exp_mean("_z")
    out["hist_zero_rate"] = _exp_mean("_zero")
    out["hist_mean_7"] = _roll_mean("_y", 7)
    out["hist_mean_28"] = _roll_mean("_y", 28)
    out["hist_mean_91"] = _roll_mean("_y", 91)
    out["hist_n"] = g.cumcount().astype("float64")
    out = out.drop(columns=["_y", "_z", "_zero"])
    return out



def seasonal_lag_columns(
    frame: pd.DataFrame,
    observed: pd.DataFrame,
    origin: pd.Timestamp,
    *,
    lag_days: int = 364,
    window: int = 15,
) -> pd.DataFrame:
    """Satir duzeyinde gecen yil ayni gun (364 = 52 hafta, hafta gunu korunur).

    `hist_yoy_mean` varlik duzeyinde ve ufkun TAMAMININ gecen yilki ortalamasi;
    bu ise ufkun ICINDEKI sekli tasir: gecen yil hangi haftalar yuksekti.

    Sizinti korumasi iki katli: kaynak satirlar `origin`'e kadar kirpilir ve
    `lag_days` (364) ufuktan (122) buyuk oldugu icin aranan tarih zaten daima
    origin'den once dusuyor.

    KAPSAMA UYARISI: fold A ve C'de aranan tarihler panel baslangicindan
    onceye dusuyor, yani kolon orada nerdeyse tamamen NaN. Gonderim
    origin'inde (ufuk Nis-Tem 2026, aranan Nis-Tem 2025) kapsama tam. Bu
    asimetri yuzunden ozellik CV'de hak ettiginden dusuk gorunur.
    """
    src = observed.loc[observed[C.DATE] <= origin, [C.ENTITY, C.DATE, C.TARGET]].copy()
    out = pd.DataFrame(index=frame.index)
    if src.empty:
        out["y_lag364"] = np.nan
        out["y_lag364_win"] = np.nan
        return out

    src["y"] = np.log1p(src[C.TARGET].clip(lower=0))
    daily = src.set_index([C.ENTITY, C.DATE])["y"]
    daily = daily[~daily.index.duplicated()]

    target_date = frame[C.DATE] - pd.Timedelta(days=lag_days)
    key = pd.MultiIndex.from_arrays([frame[C.ENTITY], target_date])
    out["y_lag364"] = daily.reindex(key).to_numpy()

    # Ayni tarihin etrafinda pencere ortalamasi: tek gun cok gurultulu ve
    # gecen yilin takvimi (bayram, hafta sonu) birebir hizalanmiyor.
    roll = (
        src.sort_values([C.ENTITY, C.DATE])
        .set_index(C.DATE)
        .groupby(C.ENTITY)["y"]
        .rolling(f"{window}D", min_periods=3)
        .mean()
    )
    roll.index = roll.index.set_names([C.ENTITY, C.DATE])
    roll = roll[~roll.index.duplicated(keep="last")]
    out["y_lag364_win"] = roll.reindex(key).to_numpy()
    return out
