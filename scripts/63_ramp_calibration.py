"""Gonderimin mevsimsel rampasi dogru mu: origin seviyesinden aya gore lift.

Bu olcum model gerektirmiyor. Origin'de canli trafolarin son-28-gun log1p
seviyesi ile hedef aydaki gerceklesen ortalama log1p arasindaki fark
("lift") bir mevsim imzasidir. 2025'te Mart 31 origin'i icin bu imza
dogrudan olculebilir; gonderimde ayni imzayi uygulayip uygulamadigimizi
karsilastiriyoruz.

Ayrica ayni tabloyu diger origin'ler icin uretip lift'in ne kadar
tekrarlanabilir oldugunu gosteriyor -- tek yillik veride bu terimin
belirsizligi karari belirliyor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import load_test, load_train

OUT = []


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def alive_level(train: pd.DataFrame, origin: pd.Timestamp, days: int = 28) -> pd.Series:
    h = train.loc[(train[C.DATE] <= origin)
                  & (train[C.DATE] > origin - pd.Timedelta(days=days))]
    g = h.groupby(C.ENTITY)[C.TARGET]
    lvl = g.apply(lambda s: float(np.log1p(s.clip(lower=0)).mean()))
    zr = g.apply(lambda s: float((s <= 0).mean()))
    return lvl.loc[zr <= 0.9]


def lift_table(train: pd.DataFrame, origin: pd.Timestamp, months: int = 4) -> pd.DataFrame:
    """Origin'de canli trafolarin, sonraki aylardaki lift'i (yalniz ayni trafolar)."""
    lvl = alive_level(train, origin)
    end = origin + pd.Timedelta(days=122)
    fut = train.loc[(train[C.DATE] > origin) & (train[C.DATE] <= end)
                    & train[C.ENTITY].isin(set(lvl.index))].copy()
    if fut.empty:
        return pd.DataFrame()
    fut["logp"] = np.log1p(fut[C.TARGET].clip(lower=0))
    fut["lvl"] = fut[C.ENTITY].map(lvl)
    fut["ay_ofset"] = ((fut[C.DATE].dt.to_period("M")
                        - origin.to_period("M")).apply(lambda x: x.n))
    t = fut.groupby("ay_ofset").agg(
        n=("logp", "size"),
        trafo=(C.ENTITY, "nunique"),
        gercek=("logp", "mean"),
        origin_lvl=("lvl", "mean"),
    )
    t["lift"] = t["gercek"] - t["origin_lvl"]
    return t


def main() -> None:
    train, test = load_train(), load_test()
    say("# Mevsimsel rampa kalibrasyonu")
    say(f"Uretim: {pd.Timestamp.now()}")
    say()
    say("Lift = (hedef aydaki ortalama log1p) - (origin'de son 28 gun seviyesi),")
    say("yalnizca origin'de CANLI olan ayni trafolar uzerinde.")
    say()

    origins = ["2025-03-31", "2025-05-31", "2025-07-31", "2025-09-30", "2025-11-30"]
    for o in origins:
        o = pd.Timestamp(o)
        t = lift_table(train, o)
        if t.empty:
            continue
        say(f"## origin {o.date()}  (takvim ayi {o.month})")
        say(t.round(4).to_string())
        say()

    # Gonderimin ayni imzasi
    say("# Gonderimin uyguladigi lift")
    say()
    lvl26 = alive_level(train, C.TRAIN_END)
    say(f"2026-03-31'de canli trafo: {len(lvl26):,}, ortalama son-28g seviye "
        f"{lvl26.mean():.4f}")
    say()
    p = C.SUBMISSIONS_DIR / "submission_BEST_1.06713.csv"
    if not p.exists():
        say(f"(gonderim yok: {p})")
    else:
        sub = pd.read_csv(p, dtype={"id": "string"})
        m = test.merge(sub, on="id", how="left")
        m = m.loc[m[C.ENTITY].isin(set(lvl26.index))].copy()
        m["logp"] = np.log1p(m[C.TARGET].clip(lower=0))
        m["lvl"] = m[C.ENTITY].map(lvl26)
        m["ay_ofset"] = m[C.DATE].dt.month - 3
        t = m.groupby("ay_ofset").agg(
            n=("logp", "size"), trafo=(C.ENTITY, "nunique"),
            tahmin=("logp", "mean"), origin_lvl=("lvl", "mean"))
        t["lift"] = t["tahmin"] - t["origin_lvl"]
        say("## Gonderim: origin'de canli trafolar icin uygulanan lift")
        say(t.round(4).to_string())
        say()

        ref = lift_table(train, pd.Timestamp("2025-03-31"))
        say("## Yan yana: 2025 gerceklesen vs 2026 gonderim (ayni takvim aylari)")
        cmp = pd.DataFrame({
            "lift_2025_gercek": ref["lift"],
            "lift_2026_tahmin": t["lift"],
        })
        cmp["fark"] = cmp["lift_2026_tahmin"] - cmp["lift_2025_gercek"]
        say(cmp.round(4).to_string())
        say()
        say("Fark pozitifse gonderim 2025'ten daha dik, negatifse daha duz.")
        say()

        # Hava farkiyla beklenen duzeltme
        say("## 2026 yazinin 2025'e gore hava farki")
        try:
            from src.external.weather import load_weather, weather_features
            w = weather_features(load_weather())
            w["ay"] = w[C.DATE].dt.month
            w["yil"] = w[C.DATE].dt.year
            piv = w.loc[w["ay"].isin([4, 5, 6, 7])].groupby(["ay", "yil"])[
                ["cdd", "temperature_2m_mean"]].mean().unstack()
            say(piv.round(3).to_string())
        except Exception as e:  # pragma: no cover
            say(f"(hava okunamadi: {e})")
        say()

        say("## Kazanc aritmetigi")
        say("Bir ayda ortalama sapma d ise, o ayin MSE katkisi d^2 kadar fazla.")
        say(f"{'ay':>4} {'n_pay':>8} {'fark':>8} {'dMSE':>9}")
        tot = len(test)
        toplam = 0.0
        for ay_ofset, row in cmp.iterrows():
            n = int(t.loc[ay_ofset, "n"]) if ay_ofset in t.index else 0
            d = float(row["fark"])
            contrib = n / tot * d * d
            toplam += contrib
            say(f"{ay_ofset + 3:>4} {n / tot:8.3f} {d:8.4f} {contrib:9.5f}")
        say(f"{'toplam':>4} {'':>8} {'':>8} {toplam:9.5f}")
        say()
        say("Not: bu, 2026'nin 2025 ile ayni mevsimsel imzaya sahip oldugunu")
        say("varsayarsa gecerli ust sinir. Hava farki bu farkin bir kismini")
        say("mesru kiliyor; ustteki CDD tablosu ne kadarini acikladigini gosterir.")

    p = C.REPORTS_DIR / "63_ramp_calibration.md"
    p.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
