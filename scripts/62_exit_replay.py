"""Cikis sinyalinin test-ozdes tekrari: pencere = 122 gun, origin = pencere oncesi.

`61` iki pencerede celisik sonuc verdi (Nis-Tem 2025'te cikanlarin sifir orani
0,71; Oca-Eki 2025'te 0,20). Fark, "temiz cikis" tanimindan ve pencere
uzunlugundan geliyordu. Burada test'in tam ozdesi kuruluyor:

  - pencere uzunlugu 122 gun,
  - "erken cikis" = pencere ici son satir < pencere son gunu
    (test'te bilebilecegimiz tek tanim; sonradan donup donmedigi bilinmez),
  - origin = pencere baslangicindan onceki gun; trafonun origin'deki durumu
    (canli/olu) ayrilarak raporlanir, cunku model bunu zaten goruyor.

Boylece "modelin gecmisten zaten bildigi" ile "yalnizca panel deseninden
gelen yeni bilgi" birbirinden ayrilir. Kazanc yalnizca ikincisinden gelebilir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import load_test, load_train

OUT = []
WINDOWS = [
    ("A 2025-04-01..2025-07-31", "2025-03-31", "2025-04-01", "2025-07-31"),
    ("B 2025-12-01..2026-03-31", "2025-11-30", "2025-12-01", "2026-03-31"),
    ("C 2025-10-01..2026-01-31", "2025-09-30", "2025-10-01", "2026-01-31"),
    ("D 2025-06-01..2025-09-30", "2025-05-31", "2025-06-01", "2025-09-30"),
]


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def origin_state(train: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Origin'de her trafonun modelin gordugu durumu: son 28 gun seviye + sifir orani."""
    h = train.loc[train[C.DATE] <= origin]
    w28 = h.loc[h[C.DATE] > origin - pd.Timedelta(days=28)]
    g = w28.groupby(C.ENTITY)[C.TARGET]
    out = pd.DataFrame({
        "lvl28": g.apply(lambda s: float(np.log1p(s.clip(lower=0)).mean())),
        "zero28": g.apply(lambda s: float((s <= 0).mean())),
        "n28": g.size(),
    })
    out["olu"] = out["zero28"] > 0.9
    return out


def replay(train: pd.DataFrame, label: str, origin, start, end) -> None:
    origin, start, end = (pd.Timestamp(x) for x in (origin, start, end))
    w = train.loc[(train[C.DATE] >= start) & (train[C.DATE] <= end)].copy()
    last = w.groupby(C.ENTITY)[C.DATE].max()
    w = w.merge(last.rename("son"), left_on=C.ENTITY, right_index=True)
    w["gun_cikisa"] = (w["son"] - w[C.DATE]).dt.days
    w["erken"] = w["son"] < end
    w["logp"] = np.log1p(w[C.TARGET].clip(lower=0))

    st = origin_state(train, origin)
    w = w.join(st[["lvl28", "zero28", "olu"]], on=C.ENTITY)
    w["gecmisli"] = w["lvl28"].notna()
    w["olu"] = w["olu"].fillna(False).astype(bool)

    say(f"## {label}  (origin {origin.date()})")
    ent = w.groupby(C.ENTITY)["erken"].first()
    say(f"pencere trafosu {len(ent):,}; erken cikan {int(ent.sum()):,} "
        f"({ent.mean():.1%}); erken cikan satir {int(w['erken'].sum()):,} "
        f"({w['erken'].mean():.2%})")
    say()

    say("### Origin'de gecmisi olan trafolar: origin durumu x cikis")
    rows = []
    for gecmisli in (True,):
        for olu in (False, True):
            for erken in (False, True):
                s = w.loc[(w["gecmisli"] == gecmisli) & (w["olu"] == olu)
                          & (w["erken"] == erken)]
                if not len(s):
                    continue
                rows.append({
                    "origin_durum": "olu" if olu else "canli",
                    "cikis": "erken" if erken else "pencere sonuna kadar",
                    "n": len(s),
                    "trafo": s[C.ENTITY].nunique(),
                    "sifir_orani": float((s[C.TARGET] <= 0).mean()),
                    "gercek_logp": float(s["logp"].mean()),
                    "origin_lvl28": float(s["lvl28"].mean()),
                })
    say(pd.DataFrame(rows).to_string(index=False))
    say()

    canli_erken = w.loc[w["gecmisli"] & ~w["olu"] & w["erken"]]
    if len(canli_erken):
        say("### KRITIK DILIM: origin'de CANLI ama pencerede erken cikan")
        t = canli_erken.groupby(
            pd.cut(canli_erken["gun_cikisa"], [-0.5, 0.5, 2, 6, 13, 29, 59, 400]),
            observed=True,
        ).agg(n=("logp", "size"),
              sifir_orani=(C.TARGET, lambda s: float((s <= 0).mean())),
              gercek_logp=("logp", "mean"),
              origin_lvl28=("lvl28", "mean"))
        t["fark"] = t["gercek_logp"] - t["origin_lvl28"]
        say(t.to_string())
        say()

    cold = w.loc[~w["gecmisli"]]
    if len(cold):
        say("### Origin'de gecmisi OLMAYAN (cold) trafolar x cikis")
        t = cold.groupby("erken").agg(
            n=("logp", "size"), trafo=(C.ENTITY, "nunique"),
            sifir_orani=(C.TARGET, lambda s: float((s <= 0).mean())),
            gercek_logp=("logp", "mean"))
        say(t.to_string())
        cold_erken = cold.loc[cold["erken"]]
        if len(cold_erken):
            t = cold_erken.groupby(
                pd.cut(cold_erken["gun_cikisa"], [-0.5, 0.5, 2, 6, 13, 29, 59, 400]),
                observed=True,
            ).agg(n=("logp", "size"),
                  sifir_orani=(C.TARGET, lambda s: float((s <= 0).mean())),
                  gercek_logp=("logp", "mean"))
            say(t.to_string())
    say()


def test_side(train: pd.DataFrame, test: pd.DataFrame) -> None:
    say("# Test tarafi: erken cikanlarin Mart 2026 durumu")
    say()
    last = test.groupby(C.ENTITY)[C.DATE].max()
    erken = set(last[last < C.TEST_END].index)
    st = origin_state(train, C.TRAIN_END)

    te = test.copy()
    te["erken"] = te[C.ENTITY].isin(erken)
    te = te.join(st[["lvl28", "zero28", "olu"]], on=C.ENTITY)
    te["gecmisli"] = te["lvl28"].notna()

    ent = te.groupby(C.ENTITY).agg(erken=("erken", "first"),
                                   gecmisli=("gecmisli", "first"),
                                   olu=("olu", "first"))
    say("## Trafo sayilari")
    say(pd.crosstab([ent["gecmisli"], ent["olu"].fillna(False)], ent["erken"]).to_string())
    say()
    say("## Satir sayilari")
    say(pd.crosstab([te["gecmisli"], te["olu"].fillna(False)], te["erken"]).to_string())
    say()

    kritik = te.loc[te["gecmisli"] & ~te["olu"].fillna(False) & te["erken"]]
    say(f"KRITIK DILIM (Mart'ta canli, test'te erken cikan): "
        f"{kritik[C.ENTITY].nunique():,} trafo, {len(kritik):,} satir "
        f"({len(kritik) / len(te):.2%} tum test)")
    cold_erken = te.loc[~te["gecmisli"] & te["erken"]]
    say(f"COLD + erken cikan: {cold_erken[C.ENTITY].nunique():,} trafo, "
        f"{len(cold_erken):,} satir ({len(cold_erken) / len(te):.2%})")
    olu_erken = te.loc[te["olu"].fillna(False) & te["erken"]]
    say(f"Mart'ta zaten OLU + erken cikan: {olu_erken[C.ENTITY].nunique():,} trafo, "
        f"{len(olu_erken):,} satir  (model bunu gecmisten zaten biliyor)")
    say()

    p = C.SUBMISSIONS_DIR / "submission_BEST_1.06713.csv"
    if p.exists():
        sub = pd.read_csv(p, dtype={"id": "string"})
        m = te.merge(sub, on="id", how="left")
        m["logp"] = np.log1p(m[C.TARGET].clip(lower=0))
        say("## Gonderimin bu dilimlerdeki ortalama log1p tahmini")
        t = m.groupby([m["gecmisli"], m["olu"].fillna(False), m["erken"]]).agg(
            n=("logp", "size"), tahmin=("logp", "mean"))
        say(t.to_string())
    say()


def main() -> None:
    train = load_train()
    test = load_test()
    say("# Cikis sinyali: test-ozdes tekrar")
    say(f"Uretim: {pd.Timestamp.now()}")
    say()
    for label, origin, start, end in WINDOWS:
        replay(train, label, origin, start, end)
    test_side(train, test)

    p = C.REPORTS_DIR / "62_exit_replay.md"
    p.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
