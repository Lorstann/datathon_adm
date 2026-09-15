"""Havuza eklenecek YENI yonler: kasitli olarak farkli tahminciler.

`Leaderboard Geometrisi` notunun kilit cumlesi: kazanc, bilesenlerin ne kadar
FARKLI olduguyla orantili -- ne kadar IYI olduguyla degil. Havuzdaki 16
bagimsiz gonderimin hepsi ayni boru hattinin (seviye+sekil ayrisimi, ayni
ozellik seti, LightGBM) varyanti. FINAL_4 bu uzayin optimumunda; artik hicbir
yeniden agirliklandirma 0.002'den fazla vermiyor.

Bu yuzden burada agac YOK. Uc tahminci de ham panelden, tamamen farkli
tumevarim onyargilariyla kuruluyor:

Q_SVD    Trafo x gun matrisinin dusuk-ranki. 2025 Nis-Tem penceresinde SVD;
         trafoya ozgu yuklemeler x zamana ozgu faktorler. Agacin hic
         kuramadigi bir yapi: trafolar arasi ORTAK gizli mevsim faktorleri.
Q_ANALOG Gecen yilin ayni gununu (d-364, hafta gunu hizali) birebir tekrar
         oynatir, yalnizca seviye farkiyla kaydirarak. Gun duzeyi
         idiyosenkratik gurultuyu KORUR -- tek basina kotu skor verir ama
         havuzdaki hicbir bilesende olmayan bir yon tasir.
Q_SMOOTH Agirlikli grup profili: seviye + grup mevsim rampasi + grup hafta
         gunu etkisi. Asiri puruzsuz. Notta P_EMP'in (tek basina 1.268)
         -0.44 agirlikla havuzun en degerli bileseni oldugu gorulmustu;
         puruzsuz tahminciler keskinlestirme yonu tasiyor.

Hicbiri tek basina iyi olmak zorunda degil. Olculen sey novelty: mevcut
havuzun gerdigi altuzaya dik bileseni ne kadar buyuk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import load_test, load_train, split_location
from src.validation.folds import guc_band

W_START = pd.Timestamp("2025-03-25")   # analog pencere (d-364 hepsini kapsar)
W_END = pd.Timestamp("2025-08-05")
MAR26 = (pd.Timestamp("2026-03-04"), pd.Timestamp("2026-03-31"))
MAR25 = (pd.Timestamp("2025-03-04"), pd.Timestamp("2025-03-31"))
RANK = 8
PRIOR = 20.0
CLIP = (0.0, 12.0)

OUT: list[str] = []


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def mean_log(df: pd.DataFrame, lo, hi) -> pd.Series:
    w = df.loc[(df[C.DATE] >= lo) & (df[C.DATE] <= hi)]
    return w.groupby(C.ENTITY)[C.TARGET].apply(
        lambda s: float(np.log1p(s.clip(lower=0)).mean()))


def shrunk(vals: pd.Series, keys: pd.Series, prior: float = PRIOR) -> pd.Series:
    """Ampirik Bayes ile globale cekilmis grup ortalamasi."""
    g = pd.DataFrame({"v": vals, "k": keys}).dropna()
    glob = float(g["v"].mean())
    agg = g.groupby("k", observed=True)["v"].agg(["mean", "size"])
    sm = (agg["mean"] * agg["size"] + glob * prior) / (agg["size"] + prior)
    sm.loc["__GLOBAL__"] = glob
    return sm


def build_base(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Her test satiri icin taban seviye ve grup anahtari."""
    lvl26 = mean_log(train, *MAR26)
    lvl25 = mean_log(train, *MAR25)
    lvl_all = train.groupby(C.ENTITY)[C.TARGET].apply(
        lambda s: float(np.log1p(s.clip(lower=0)).mean()))

    stat = train.groupby(C.ENTITY).agg(guc=(C.POWER, "first"),
                                       lok=(C.LOCATION, "first"))
    stat["band"] = guc_band(stat["guc"]).astype("string")
    stat["ilce"] = split_location(stat["lok"])["ilce"].fillna(
        split_location(stat["lok"])["il"]).astype("string")
    stat["grp"] = stat["band"] + "|" + stat["ilce"]
    stat["lvl"] = lvl26.reindex(stat.index)
    stat["lvl"] = stat["lvl"].fillna(lvl_all.reindex(stat.index))
    stat["z"] = stat["lvl"] - np.log(stat["guc"].clip(lower=1))

    z_grp = shrunk(stat["z"], stat["grp"])

    t = test[["id", C.ENTITY, C.DATE, C.POWER, C.LOCATION]].copy()
    t["band"] = guc_band(t[C.POWER]).astype("string")
    loc = split_location(t[C.LOCATION])
    t["ilce"] = loc["ilce"].fillna(loc["il"]).astype("string")
    t["grp"] = t["band"] + "|" + t["ilce"]
    t["cold"] = ~t[C.ENTITY].isin(set(stat.index))
    t["lvl"] = t[C.ENTITY].map(stat["lvl"])
    t["lvl25"] = t[C.ENTITY].map(lvl25)
    zg = t["grp"].map(z_grp).fillna(z_grp["__GLOBAL__"])
    t["lvl"] = t["lvl"].fillna(zg + np.log(t[C.POWER].clip(lower=1)))
    t["analog"] = t[C.DATE] - pd.Timedelta(days=364)
    t["dow"] = t[C.DATE].dt.dayofweek
    t["ay"] = t[C.DATE].dt.month
    return t, stat


def group_ramp(train: pd.DataFrame, stat: pd.DataFrame) -> pd.DataFrame:
    """Grup bazinda (ay, hafta gunu) mevsim etkisi: 2025 Mart'ina gore sapma."""
    w = train.loc[(train[C.DATE] >= pd.Timestamp("2025-04-01"))
                  & (train[C.DATE] <= pd.Timestamp("2025-07-31"))].copy()
    w["lp"] = np.log1p(w[C.TARGET].clip(lower=0))
    w["base"] = w[C.ENTITY].map(mean_log(train, *MAR25))
    w = w.dropna(subset=["base"])
    w["dev"] = w["lp"] - w["base"]
    w["grp"] = w[C.ENTITY].map(stat["grp"])
    w["ay"] = w[C.DATE].dt.month
    w["dow"] = w[C.DATE].dt.dayofweek
    ramp = w.groupby(["grp", "ay"], observed=True)["dev"].mean().rename("ramp")
    glob_ramp = w.groupby("ay")["dev"].mean().rename("gramp")
    dowe = w.groupby(["grp", "dow"], observed=True)["dev"].mean().rename("dw")
    glob_dow = w.groupby("dow")["dev"].mean().rename("gdw")
    return ramp, glob_ramp, dowe, glob_dow


def q_smooth(t: pd.DataFrame, parts) -> np.ndarray:
    ramp, gramp, dowe, gdw = parts
    key = pd.MultiIndex.from_arrays([t["grp"], t["ay"]])
    r = pd.Series(key.map(ramp)).astype("float64")
    r = r.fillna(pd.Series(t["ay"].map(gramp)).astype("float64")).fillna(0.0)
    key2 = pd.MultiIndex.from_arrays([t["grp"], t["dow"]])
    d = pd.Series(key2.map(dowe)).astype("float64")
    d = d.fillna(pd.Series(t["dow"].map(gdw)).astype("float64")).fillna(0.0)
    dm = pd.Series(t["ay"].map(gramp)).astype("float64").fillna(0.0)
    # dow etkisi ay etkisiyle ortusuyor; ay ortalamasini dow'dan cikar
    return t["lvl"].to_numpy() + r.to_numpy() + (d.to_numpy() - dm.to_numpy())


def q_analog(train: pd.DataFrame, t: pd.DataFrame, parts) -> np.ndarray:
    """Gecen yilin ayni gununu tekrar oynat, seviye farkiyla kaydir."""
    hist = train[[C.ENTITY, C.DATE, C.TARGET]].copy()
    hist["lp"] = np.log1p(hist[C.TARGET].clip(lower=0))
    key = hist.set_index([C.ENTITY, C.DATE])["lp"]
    idx = pd.MultiIndex.from_arrays([t[C.ENTITY], t["analog"]])
    v = pd.Series(idx.map(key)).astype("float64")
    shift = (t["lvl"] - t["lvl25"]).astype("float64")
    out = v + shift.fillna(0.0)
    # dusen satirlar: puruzsuz tahminciye dus
    fb = q_smooth(t, parts)
    return np.where(np.isfinite(out.to_numpy()), out.to_numpy(), fb)


def q_svd(train: pd.DataFrame, t: pd.DataFrame, stat: pd.DataFrame,
          parts) -> np.ndarray:
    """Trafo x gun matrisinin dusuk-ranki: ortak gizli mevsim faktorleri."""
    w = train.loc[(train[C.DATE] >= W_START) & (train[C.DATE] <= W_END)].copy()
    w["lp"] = np.log1p(w[C.TARGET].clip(lower=0))
    cnt = w.groupby(C.ENTITY)["lp"].size()
    keep = cnt[cnt >= 90].index
    w = w.loc[w[C.ENTITY].isin(set(keep))]
    M = w.pivot_table(index=C.ENTITY, columns=C.DATE, values="lp",
                      aggfunc="mean")
    row_mean = M.mean(axis=1)
    Dv = M.sub(row_mean, axis=0)
    col_mean = Dv.mean(axis=0)
    Dv = Dv.sub(col_mean, axis=1).fillna(0.0)

    U, S, Vt = np.linalg.svd(Dv.to_numpy(), full_matrices=False)
    U, S, Vt = U[:, :RANK], S[:RANK], Vt[:RANK]
    say(f"  SVD: {Dv.shape[0]:,} trafo x {Dv.shape[1]} gun, ilk {RANK} "
        f"tekil deger aciklanan varyans "
        f"{float((S ** 2).sum() / (Dv.to_numpy() ** 2).sum()):.3f}")

    load = pd.DataFrame(U * S, index=Dv.index)
    grp = stat["grp"].reindex(load.index)
    grp_load = load.groupby(grp, observed=True).mean()
    glob_load = load.mean()

    date_pos = {d: j for j, d in enumerate(Dv.columns)}
    col_mean_arr = col_mean.reindex(Dv.columns).to_numpy()

    pos = t["analog"].map(date_pos)
    ok = pos.notna().to_numpy()
    j = pos.fillna(0).astype("int64").to_numpy()

    L = load.reindex(t[C.ENTITY]).to_numpy()
    miss = ~np.isfinite(L).all(axis=1)
    if miss.any():
        gl = grp_load.reindex(t["grp"]).to_numpy()
        gl = np.where(np.isfinite(gl), gl, glob_load.to_numpy()[None, :])
        L[miss] = gl[miss]

    shape = np.einsum("ij,ji->i", L, Vt[:, j]) + col_mean_arr[j]
    base = t["lvl"].to_numpy()
    # SVD sapmalari pencere ortalamasina gore; mevsim rampasini ayrica ekle
    ramp_only = q_smooth(t, parts) - base
    out = base + ramp_only + shape
    fb = q_smooth(t, parts)
    return np.where(ok, out, fb)


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    say("# Yeni yon adaylari: kasitli olarak farkli tahminciler")
    say(f"Uretim: {pd.Timestamp.now()}")
    say()

    t, stat = build_base(train, test)
    parts = group_ramp(train, stat)
    say(f"test satiri {len(t):,}, cold {int(t['cold'].sum()):,} "
        f"({t['cold'].mean():.2%})")
    say()

    cands = {}
    say("## Uretim")
    cands["Q_SMOOTH"] = q_smooth(t, parts)
    say("  Q_SMOOTH tamam")
    cands["Q_ANALOG"] = q_analog(train, t, parts)
    say("  Q_ANALOG tamam")
    cands["Q_SVD"] = q_svd(train, t, stat, parts)
    say("  Q_SVD tamam")
    say()

    say("## Ozet")
    say(f"{'aday':<12} {'ort log1p':>10} {'std':>8} {'min':>7} {'max':>7} "
        f"{'NaN':>6}")
    for nm, v in cands.items():
        v = np.clip(np.nan_to_num(v, nan=np.nan), *CLIP)
        cands[nm] = v
        say(f"{nm:<12} {np.nanmean(v):10.4f} {np.nanstd(v):8.4f} "
            f"{np.nanmin(v):7.3f} {np.nanmax(v):7.3f} "
            f"{int(np.isnan(v).sum()):6d}")
    say()

    for nm, v in cands.items():
        pred = np.expm1(np.clip(v, 0, None))
        pd.DataFrame({"id": t["id"].to_numpy(), C.TARGET: pred}).to_csv(
            C.SUBMISSIONS_DIR / f"probe_{nm}.csv", index=False)
        say(f"yazildi: submissions/probe_{nm}.csv")
    say()

    p = C.REPORTS_DIR / "71_probe_candidates.md"
    p.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
