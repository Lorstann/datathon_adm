"""H1'in buyuklugu: test penceresinde erken biten trafolarin son gunleri.

`60_panel_signal` sunu buldu: 2025 Nis-Tem penceresinde kaydi pencere
bitmeden sona eren trafolarin son 10 gunundeki sifir orani %55,8; pencere
sonuna kadar devam edenlerde %2,7. Bu bilgi test tarafinda TAMAMEN bilinir --
her trafonun test'te hangi gunlerde satiri oldugunu goruyoruz.

Burada olculen:
  1. Train'de temiz cikis tanimi (bir daha hic donmeyen trafo) ile sifir
     orani ve seviye, cikisa kalan gune gore.
  2. Ayni sey giris tarafinda.
  3. Test'te bu desenin kac satiri kapsadigi.
  4. Mevcut en iyi gonderimin o satirlarda ne tahmin ettigi ve tahmini kazanc.
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


def exit_profile(train: pd.DataFrame, start, end, label: str) -> None:
    """Pencere icinde TEMIZ cikan trafolar: pencereden sonra hic satiri yok."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    w = train.loc[(train[C.DATE] >= start) & (train[C.DATE] <= end)].copy()
    last_in_w = w.groupby(C.ENTITY)[C.DATE].max()
    after = train.loc[train[C.DATE] > end, C.ENTITY].unique()
    donmeyen = set(last_in_w.index) - set(after)
    erken = last_in_w[(last_in_w < end) & last_in_w.index.isin(donmeyen)]

    say(f"## {label}")
    say(f"pencere trafosu {last_in_w.size:,}; pencere bitmeden kaydi biten ve")
    say(f"pencereden sonra hic donmeyen: {len(erken):,}")
    if not len(erken):
        say()
        return

    sub = w.loc[w[C.ENTITY].isin(set(erken.index))].copy()
    sub = sub.merge(erken.rename("son"), left_on=C.ENTITY, right_index=True)
    sub["gun_cikisa"] = (sub["son"] - sub[C.DATE]).dt.days
    t = sub.groupby(pd.cut(sub["gun_cikisa"], [-0.5, 0.5, 2, 6, 13, 29, 59, 400]),
                    observed=True).agg(
        n=(C.TARGET, "size"),
        sifir_orani=(C.TARGET, lambda s: float((s <= 0).mean())),
        ort_log=(C.TARGET, lambda s: float(np.log1p(s).mean())),
    )
    say("### Temiz cikanlarda cikisa kalan gune gore")
    say(t.to_string())

    kalan = w.loc[~w[C.ENTITY].isin(set(erken.index))]
    say(f"karsilastirma — pencere sonuna kadar devam edenler: n={len(kalan):,}, "
        f"sifir orani {float((kalan[C.TARGET] <= 0).mean()):.4f}, "
        f"ort_log {float(np.log1p(kalan[C.TARGET]).mean()):.4f}")
    say()


def entry_profile(train: pd.DataFrame, start, end, label: str) -> None:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    w = train.loc[(train[C.DATE] >= start) & (train[C.DATE] <= end)].copy()
    first_in_w = w.groupby(C.ENTITY)[C.DATE].min()
    before = train.loc[train[C.DATE] < start, C.ENTITY].unique()
    yeni = first_in_w[(first_in_w > start) & ~first_in_w.index.isin(set(before))]
    say(f"## {label} — giris tarafi")
    say(f"pencere icinde ilk kez gorulen (once hic yok): {len(yeni):,}")
    if not len(yeni):
        say()
        return
    sub = w.loc[w[C.ENTITY].isin(set(yeni.index))].copy()
    sub = sub.merge(yeni.rename("ilk"), left_on=C.ENTITY, right_index=True)
    sub["gun_giristen"] = (sub[C.DATE] - sub["ilk"]).dt.days
    t = sub.groupby(pd.cut(sub["gun_giristen"], [-0.5, 0.5, 2, 6, 13, 29, 59, 400]),
                    observed=True).agg(
        n=(C.TARGET, "size"),
        sifir_orani=(C.TARGET, lambda s: float((s <= 0).mean())),
        ort_log=(C.TARGET, lambda s: float(np.log1p(s).mean())),
    )
    say(t.to_string())
    say()


def test_exposure(test: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    say("# Test tarafinda kac satiri kapsiyor")
    say()
    last = test.groupby(C.ENTITY)[C.DATE].max()
    first = test.groupby(C.ENTITY)[C.DATE].min()
    n_days = test.groupby(C.ENTITY)[C.DATE].nunique()
    erken = last[last < C.TEST_END]
    say(f"test trafosu {last.size:,}; 2026-07-31'den once kaydi biten: "
        f"{len(erken):,} ({len(erken) / last.size:.1%})")

    te = test.copy()
    te = te.merge(last.rename("son"), left_on=C.ENTITY, right_index=True)
    te = te.merge(first.rename("ilk"), left_on=C.ENTITY, right_index=True)
    te = te.merge(n_days.rename("n_gun"), left_on=C.ENTITY, right_index=True)
    te["gun_cikisa"] = (te["son"] - te[C.DATE]).dt.days
    te["gun_giristen"] = (te[C.DATE] - te["ilk"]).dt.days
    te["erken_cikis"] = te["son"] < C.TEST_END
    tr_ent = set(train[C.ENTITY].unique())
    te["is_cold"] = ~te[C.ENTITY].isin(tr_ent)

    say()
    say("## Erken cikan trafolarin satirlari, cikisa kalan gune gore")
    e = te.loc[te["erken_cikis"]]
    t = e.groupby(pd.cut(e["gun_cikisa"], [-0.5, 0.5, 2, 6, 13, 29, 59, 400]),
                  observed=True).agg(n=("id", "size"),
                                     cold_pay=("is_cold", "mean"))
    t["tum_test_payi"] = t["n"] / len(te)
    say(t.to_string())
    say()
    say(f"erken cikan trafolarin toplam satiri: {len(e):,} "
        f"({len(e) / len(te):.2%} tum test)")
    say(f"bunlarin son 14 gunu: {int((e['gun_cikisa'] <= 13).sum()):,} satir "
        f"({(e['gun_cikisa'] <= 13).sum() / len(te):.2%} tum test)")
    say(f"bunlarin son 30 gunu: {int((e['gun_cikisa'] <= 29).sum()):,} satir "
        f"({(e['gun_cikisa'] <= 29).sum() / len(te):.2%} tum test)")
    say()

    say("## Erken cikis tarihlerinin dagilimi (en yogun 15)")
    say(erken.value_counts().sort_values(ascending=False).head(15).to_string())
    say()

    say("## Test penceresinde gec giren trafolar")
    gec = first[first > C.TEST_START]
    say(f"2026-04-01'den sonra baslayan: {len(gec):,} ({len(gec) / last.size:.1%})")
    say(gec.value_counts().sort_values(ascending=False).head(10).to_string())
    say()
    return te


def submission_view(te: pd.DataFrame) -> None:
    p = C.SUBMISSIONS_DIR / "submission_BEST_1.06713.csv"
    if not p.exists():
        say(f"(gonderim dosyasi yok: {p})")
        return
    sub = pd.read_csv(p, dtype={"id": "string"})
    m = te.merge(sub, on="id", how="left")
    m["logp"] = np.log1p(m[C.TARGET].clip(lower=0))
    say("# Mevcut en iyi gonderim bu satirlarda ne diyor")
    say()
    e = m.loc[m["erken_cikis"]]
    t = e.groupby(pd.cut(e["gun_cikisa"], [-0.5, 0.5, 2, 6, 13, 29, 59, 400]),
                  observed=True).agg(n=("id", "size"), ort_log_tahmin=("logp", "mean"))
    say("## Erken cikanlarda ortalama log1p tahmini")
    say(t.to_string())
    say(f"pencere sonuna kadar devam edenlerde: "
        f"{float(m.loc[~m['erken_cikis'], 'logp'].mean()):.4f}")
    say()

    # Kazanc aritmetigi: son N gunu p oraninda sifir varsayarak optimal
    # log-uzayi kucultmesi (1-p) uygulanirsa MSE'de ne kazanilir.
    say("## Kazanc aritmetigi")
    say("Bir satirda gercek sifir olasiligi p, sifir olmayan hali m ise")
    say("RMSLE-optimal tahmin (1-p)*m ve mevcut m'ye gore MSE kazanci p^2*m^2.")
    say("Toplam test satiri = 714.688; kazanc paylari buna gore.")
    say()
    say(f"{'dilim':<18} {'n':>8} {'m':>7} {'p':>6} {'dMSE_toplam':>12}")
    tot = len(m)
    for lo, hi, p_hat in [(0, 0, 0.56), (1, 2, 0.56), (3, 6, 0.56),
                          (7, 13, 0.56), (14, 29, 0.40), (30, 59, 0.25)]:
        sel = e.loc[(e["gun_cikisa"] >= lo) & (e["gun_cikisa"] <= hi)]
        if not len(sel):
            continue
        mm = float(sel["logp"].mean())
        d = len(sel) / tot * (p_hat ** 2) * (mm ** 2)
        say(f"cikisa {lo}-{hi:<10} {len(sel):8,} {mm:7.3f} {p_hat:6.2f} {d:12.5f}")
    say()
    say("Referans: LB 1.06713 -> MSE 1.13877. Hedef ilk 10 = 0.99971 -> MSE 0.99942.")
    say("Gereken toplam MSE dususu: 0.139.")
    say()


def main() -> None:
    train = load_train()
    test = load_test()
    say("# Cikis/giris deseni: buyukluk olcumu")
    say(f"Uretim: {pd.Timestamp.now()}")
    say()
    say("# Train tarafi")
    say()
    exit_profile(train, "2025-04-01", "2025-07-31", "Pencere 2025-04-01..2025-07-31")
    exit_profile(train, "2025-01-01", "2025-10-31", "Pencere 2025-01-01..2025-10-31")
    entry_profile(train, "2025-04-01", "2025-07-31", "Pencere 2025-04-01..2025-07-31")
    te = test_exposure(test, train)
    submission_view(te)

    p = C.REPORTS_DIR / "61_exit_signal.md"
    p.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
