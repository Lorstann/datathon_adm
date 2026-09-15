"""Sekil terimleri ne kadar kazandirabilir: DOW profili ve trafo-bazli hava egimi.

`reports/22` sunu olcmustu: yalnizca seviye ile warm RMSLE 0.6723, tam model
0.6593. Yani sekil modeli warm tarafinda 0.013 katiyor. `reports/43` ise
trafo x gun varyansini 0.1892 (RMSE 0.435) ve bunun %53'unun trafo-bazli
CDD/HDD egimleriyle aciklanabildigini buldu -- ama bunu "toplam varyansin
%3,75'i" diye kucumsedi. Yanlis payda: dogru payda modelin ARTIK hatasi
(warm MSE ~0.47), ve 0.10 orada %21 demek.

Hipotez: model bu terimleri ogrenemiyor cunku carpim olarak verilmiyor.
`hist_dow_rel_0..6` yedi ayri kolon; agacin once `dow`'a sonra dogru kolona
bolmesi gerekiyor. `hist_cdd_slope` var ama `cdd` ile carpimi yok.

Burada olculen: origin oncesi tahmin edilen profillerin, origin SONRASI
pencerede artigi ne kadar kucultttugu. Hepsi ornek-disi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import load_train
from src.external.weather import CACHE_PATH, load_weather, weather_features

OUT = []
WINDOWS = [
    ("A 2025-03-31", "2025-03-31", "2025-04-01", "2025-07-31"),
    ("B 2025-11-30", "2025-11-30", "2025-12-01", "2026-03-31"),
    ("C 2025-09-30", "2025-09-30", "2025-10-01", "2026-01-31"),
    ("D 2025-05-31", "2025-05-31", "2025-06-01", "2025-09-30"),
]
MIN_HIST = 60


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def fit_profiles(hist: pd.DataFrame, wx: pd.DataFrame | None) -> dict:
    """Origin oncesi veriden trafo bazli profiller. Yalnizca sifir-olmayan gunler.

    Sifirlar seviyeyi degil DURUMU tasiyor; sekil terimlerini onlarla
    kestirmek profili bozar (olu gunler tum gunlere -6 log yaziyor).
    """
    h = hist.loc[hist[C.TARGET] > 0].copy()
    h["logp"] = np.log1p(h[C.TARGET])
    h["dow"] = h[C.DATE].dt.dayofweek
    ent_mean = h.groupby(C.ENTITY)["logp"].transform("mean")
    h["dev"] = h["logp"] - ent_mean

    n = h.groupby(C.ENTITY)["logp"].size()
    ok = set(n[n >= MIN_HIST].index)
    h = h.loc[h[C.ENTITY].isin(ok)]

    # DOW profili: trafo x haftagunu ortalama sapma, globale cekilmis
    g = h.groupby([C.ENTITY, "dow"])["dev"].agg(["mean", "size"])
    glob_dow = h.groupby("dow")["dev"].mean()
    prior = 8.0
    g["sm"] = (g["mean"] * g["size"]
               + g.index.get_level_values("dow").map(glob_dow) * prior) / (g["size"] + prior)
    dow_prof = g["sm"]

    out = {"dow": dow_prof, "glob_dow": glob_dow}

    if wx is not None:
        hw = h.merge(wx[[C.LOCATION, C.DATE, "cdd", "hdd"]], on=[C.LOCATION, C.DATE],
                     how="left")
        hw = hw.dropna(subset=["cdd", "hdd"])
        # dow etkisi cikarilmis artik uzerinde egim
        key = pd.MultiIndex.from_arrays([hw[C.ENTITY], hw["dow"]])
        hw["dev2"] = hw["dev"] - pd.Series(key.map(dow_prof)).astype("float64").fillna(0).to_numpy()

        def slopes(gr: pd.DataFrame) -> pd.Series:
            X = np.column_stack([np.ones(len(gr)), gr["cdd"].to_numpy(),
                                 gr["hdd"].to_numpy()])
            if len(gr) < MIN_HIST or np.linalg.matrix_rank(X) < 3:
                return pd.Series({"b_cdd": 0.0, "b_hdd": 0.0})
            b, *_ = np.linalg.lstsq(X, gr["dev2"].to_numpy(), rcond=None)
            return pd.Series({"b_cdd": float(np.clip(b[1], -0.3, 0.3)),
                              "b_hdd": float(np.clip(b[2], -0.3, 0.3))})

        sl = hw.groupby(C.ENTITY)[["cdd", "hdd", "dev2"]].apply(slopes)
        out["slope"] = sl
    return out


def evaluate(train: pd.DataFrame, wx: pd.DataFrame | None, label: str,
             origin, start, end) -> dict:
    origin, start, end = (pd.Timestamp(x) for x in (origin, start, end))
    hist = train.loc[train[C.DATE] <= origin]
    prof = fit_profiles(hist, wx)

    # seviye: origin oncesi son 7 gun (boru hattiyla ayni zincir basi)
    w7 = hist.loc[hist[C.DATE] > origin - pd.Timedelta(days=7)]
    lvl = w7.groupby(C.ENTITY)[C.TARGET].apply(
        lambda s: float(np.log1p(s.clip(lower=0)).mean()))

    v = train.loc[(train[C.DATE] >= start) & (train[C.DATE] <= end)].copy()
    v = v.loc[v[C.ENTITY].isin(set(lvl.index)) & (v[C.TARGET] > 0)].copy()
    if v.empty:
        return {}
    v["a"] = np.log1p(v[C.TARGET])
    v["lvl"] = v[C.ENTITY].map(lvl)
    v["dow"] = v[C.DATE].dt.dayofweek
    v = v.dropna(subset=["lvl"])

    key = pd.MultiIndex.from_arrays([v[C.ENTITY], v["dow"]])
    v["dow_off"] = pd.Series(key.map(prof["dow"])).astype("float64").to_numpy()
    v["dow_off"] = v["dow_off"].fillna(v["dow"].map(prof["glob_dow"])).fillna(0.0)
    v["glob_dow_off"] = v["dow"].map(prof["glob_dow"]).fillna(0.0)

    if wx is not None and "slope" in prof:
        v = v.merge(wx[[C.LOCATION, C.DATE, "cdd", "hdd"]], on=[C.LOCATION, C.DATE],
                    how="left")
        v[["cdd", "hdd"]] = v[["cdd", "hdd"]].fillna(0.0)
        v = v.join(prof["slope"], on=C.ENTITY)
        v[["b_cdd", "b_hdd"]] = v[["b_cdd", "b_hdd"]].fillna(0.0)
        v["wx_off"] = v["b_cdd"] * v["cdd"] + v["b_hdd"] * v["hdd"]
        # egitim penceresindeki ortalama hava etkisi zaten seviyede var:
        # ofseti origin oncesi ortalamasina gore merkezle
        base = v.groupby(C.ENTITY)["wx_off"].transform("mean") * 0.0
        v["wx_off"] = v["wx_off"] - base
    else:
        v["wx_off"] = 0.0

    a = v["a"].to_numpy()
    lv = v["lvl"].to_numpy()

    # her varyanti kendi optimal sabit kaydirmasiyla degerlendir: seviye ile
    # pencere arasindaki mevsimsel kayma bu analizin konusu degil
    def score(off: np.ndarray) -> float:
        r = a - (lv + off)
        return float(np.sqrt(np.mean((r - r.mean()) ** 2)))

    z = np.zeros(len(v))
    res = {
        "yalniz seviye": score(z),
        "+ global DOW": score(v["glob_dow_off"].to_numpy()),
        "+ trafo DOW": score(v["dow_off"].to_numpy()),
        "+ trafo hava": score(v["wx_off"].to_numpy()),
        "+ trafo DOW + hava": score(v["dow_off"].to_numpy() + v["wx_off"].to_numpy()),
    }
    say(f"## {label}  warm sifir-olmayan satir {len(v):,}, trafo {v[C.ENTITY].nunique():,}")
    base = res["yalniz seviye"]
    say(f"{'varyant':<24} {'RMSE(log)':>10} {'delta':>8} {'dMSE':>9}")
    for k, val in res.items():
        say(f"{k:<24} {val:10.4f} {val - base:8.4f} {val**2 - base**2:9.4f}")
    say()
    return res


def main() -> None:
    train = load_train()
    wx = weather_features(load_weather()) if CACHE_PATH.exists() else None
    say("# Sekil terimlerinin bos alani")
    say(f"Uretim: {pd.Timestamp.now()}")
    say()
    say("Olcum: warm, sifir-olmayan satirlar. Tahmin = origin son-7-gun seviyesi")
    say("+ ofset. Her varyant kendi optimal sabitiyle degerlendiriliyor, yani")
    say("mevsimsel kayma disariida; olculen yalnizca GUN-ICI sekil.")
    say()
    allres: dict[str, list[float]] = {}
    for label, o, s, e in WINDOWS:
        r = evaluate(train, wx, label, o, s, e)
        for k, val in r.items():
            allres.setdefault(k, []).append(val)

    say("# Dort pencere ortalamasi")
    say()
    base = float(np.mean([x ** 2 for x in allres["yalniz seviye"]]))
    say(f"{'varyant':<24} {'MSE':>9} {'dMSE':>9} {'warm payi':>10} {'LB tahmini':>11}")
    lb_mse = 1.06713 ** 2
    for k, vals in allres.items():
        mse = float(np.mean([x ** 2 for x in vals]))
        d = mse - base
        # warm satirlar test'in %77,84'u; sifir-olmayan warm ~%75
        tot = 0.75 * d
        say(f"{k:<24} {mse:9.4f} {d:9.4f} {tot:10.4f} "
            f"{np.sqrt(max(lb_mse + tot, 1e-9)):11.4f}")
    say()
    say("Not: bu ust sinir degil alt sinir tarafinda -- ofsetler ham, model")
    say("bunlari ozellik olarak alirsa daha iyisini yapabilir. Ama isaret ve")
    say("buyukluk mertebesi buradan okunur.")

    p = C.REPORTS_DIR / "65_shape_headroom.md"
    p.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
