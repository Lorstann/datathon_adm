"""Dort hipotezin olcumu: panel uyelik deseni, tanim kodu, toplu giris, YoY.

Hicbiri modele dokunmaz; hepsi ham veri uzerinde tanidir. Amac, LB'de
1.067'de sikismis boru hattinin hangi bilgiyi hic kullanmadigini bulmak.

H1  Test panelinin uyelik deseni (bir trafonun test penceresindeki ilk/son
    gunu, aradaki bosluklar) sifir durumunu ele veriyor mu? Bu bilgi test
    tarafinda TAMAMEN bilinir; modelde yalnizca `test_gun_sayisi` ve
    `devreye_alinma_yasi` olarak, yani sayim olarak kullaniliyor.
H2  2026-05-11'de giren 1.326 cold trafonun train tarafinda bir analogu var mi?
H3  `tanim` kodunun onegi guc+lokasyon'un otesinde seviye bilgisi tasiyor mu?
H4  Trafoya ozgu mevsimsel sekil yildan yila tekrar ediyor mu?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import load_test, load_train

pd.set_option("display.width", 200)
OUT = []


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def window_panel_features(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Bir 122 gunluk pencere icinde, test tarafinda bilinen uyelik ozellikleri.

    Test'te bir trafonun hangi gunlerde satiri oldugunu biliyoruz. Ayni
    ozellikleri train penceresinde uretip sifirla iliskisini olcuyoruz.
    """
    w = df.loc[(df[C.DATE] >= start) & (df[C.DATE] <= end)].copy()
    g = w.groupby(C.ENTITY)[C.DATE]
    w["win_first"] = g.transform("min")
    w["win_last"] = g.transform("max")
    w["win_n"] = g.transform("nunique")
    w["gun_giristen"] = (w[C.DATE] - w["win_first"]).dt.days
    w["gun_cikisa"] = (w["win_last"] - w[C.DATE]).dt.days
    w = w.sort_values([C.ENTITY, C.DATE])
    prev = w.groupby(C.ENTITY)[C.DATE].shift(1)
    nxt = w.groupby(C.ENTITY)[C.DATE].shift(-1)
    w["bosluk_once"] = (w[C.DATE] - prev).dt.days.fillna(0)
    w["bosluk_sonra"] = (nxt - w[C.DATE]).dt.days.fillna(0)
    w["kapsam"] = w["win_n"] / ((w["win_last"] - w["win_first"]).dt.days + 1)
    return w


def rate_table(w: pd.DataFrame, col: str, bins) -> pd.DataFrame:
    b = pd.cut(w[col], bins=bins, include_lowest=True)
    t = w.groupby(b, observed=True).agg(
        n=(C.TARGET, "size"),
        sifir_orani=(C.TARGET, lambda s: float((s <= 0).mean())),
        ort_log=(C.TARGET, lambda s: float(np.log1p(s).mean())),
    )
    return t


def h1_panel_pattern(train: pd.DataFrame) -> None:
    say("# H1 — Panel uyelik deseni sifiri ele veriyor mu")
    say()
    say("Test'te her trafonun hangi gunlerde satiri oldugu bilinir. Ayni")
    say("ozellikleri train'in iki 122 gunluk penceresinde uretip sifir")
    say("oraniyla karsilastiriyorum.")
    say()
    for name, start, end in [
        ("2025-04-01..2025-07-31 (test'in mevsimsel ikizi)", "2025-04-01", "2025-07-31"),
        ("2025-12-01..2026-03-31 (en guncel 122 gun)", "2025-12-01", "2026-03-31"),
    ]:
        w = window_panel_features(train, pd.Timestamp(start), pd.Timestamp(end))
        say(f"## Pencere {name}")
        say(f"satir {len(w):,}  trafo {w[C.ENTITY].nunique():,}  "
            f"taban sifir orani {float((w[C.TARGET] <= 0).mean()):.4f}")
        say()

        say("### Pencere sonuna kalan gun (`gun_cikisa`)")
        t = rate_table(w, "gun_cikisa", [-0.5, 0.5, 3, 7, 14, 30, 60, 200])
        say(t.to_string())
        say()

        say("### Pencere basindan gecen gun (`gun_giristen`)")
        t = rate_table(w, "gun_giristen", [-0.5, 0.5, 3, 7, 14, 30, 60, 200])
        say(t.to_string())
        say()

        say("### Sonraki satira kalan gun (`bosluk_sonra`) — bosluk oncesi satir")
        t = rate_table(w, "bosluk_sonra", [-0.5, 1.5, 2.5, 7, 30, 200])
        say(t.to_string())
        say()

        say("### Pencere ici kapsam (`kapsam` = satir / gun araligi)")
        t = rate_table(w, "kapsam", [0, 0.5, 0.8, 0.95, 0.999, 1.001])
        say(t.to_string())
        say()

        # Pencere sonu penceresinin kendi sonuna denk gelmesi onemli:
        # trafo pencerenin son gununde bitiyorsa bu "hala aktif" demek.
        son_gun = pd.Timestamp(end)
        erken_biten = w.loc[w["win_last"] < son_gun]
        say(f"Pencere bitmeden kaydi biten trafo: "
            f"{erken_biten[C.ENTITY].nunique():,} / {w[C.ENTITY].nunique():,}")
        if len(erken_biten):
            son10 = erken_biten.loc[erken_biten["gun_cikisa"] <= 9]
            say(f"  bu trafolarin son 10 gunundeki sifir orani: "
                f"{float((son10[C.TARGET] <= 0).mean()):.4f}")
            digerleri = w.loc[w["win_last"] >= son_gun]
            say(f"  pencere sonuna kadar devam edenlerin sifir orani: "
                f"{float((digerleri[C.TARGET] <= 0).mean()):.4f}")
        say()


def h2_bulk_entry(train: pd.DataFrame, test: pd.DataFrame) -> None:
    say("# H2 — 2026-05-11 toplu girisinin train tarafinda analogu")
    say()
    first_tr = train.groupby(C.ENTITY)[C.DATE].min()
    cnt = first_tr.value_counts().sort_values(ascending=False).head(12)
    say("## Train'de ilk gorulme tarihine gore en yogun 12 gun")
    say(cnt.to_string())
    say()

    first_te = test.groupby(C.ENTITY)[C.DATE].min()
    tr_ent = set(train[C.ENTITY].unique())
    cold = set(first_te.index) - tr_ent
    bulk = set(first_te[first_te == pd.Timestamp("2026-05-11")].index)
    bulk_cold = bulk & cold
    say(f"2026-05-11'de test'e giren trafo: {len(bulk):,}, bunlarin cold olani: "
        f"{len(bulk_cold):,}")

    # Bu kohortun guc/lokasyon dagilimi diger cold trafolardan farkli mi?
    st = test.groupby(C.ENTITY).agg(guc=(C.POWER, "first"), lok=(C.LOCATION, "first"))
    a = st.loc[sorted(bulk_cold)]
    b = st.loc[sorted(cold - bulk_cold)]
    say(f"guc medyani — 05-11 cold kohortu {a['guc'].median():.0f}, "
        f"diger cold {b['guc'].median():.0f}")
    say(f"IZMIR payi — 05-11 cold {a['lok'].str.startswith('İZMİR').mean():.3f}, "
        f"diger cold {b['lok'].str.startswith('İZMİR').mean():.3f}")
    say()

    # Train'de sonradan giren trafolarin ilk 122 gunundeki davranisi
    panel_start = train[C.DATE].min()
    yeni = first_tr[first_tr > panel_start + pd.Timedelta(days=14)]
    say(f"Train'de sonradan giren trafo: {len(yeni):,}")
    sub = train.loc[train[C.ENTITY].isin(set(yeni.index))].copy()
    sub = sub.merge(yeni.rename("giris"), left_on=C.ENTITY, right_index=True)
    sub["yas"] = (sub[C.DATE] - sub["giris"]).dt.days
    ilk122 = sub.loc[sub["yas"] <= 121]
    say("### Yeni trafolarin yasa gore sifir orani ve seviyesi")
    t = rate_table(ilk122, "yas", [-0.5, 7, 14, 30, 60, 90, 121])
    say(t.to_string())
    say()


def h3_tanim_prefix(train: pd.DataFrame, test: pd.DataFrame) -> None:
    say("# H3 — `tanim` kodu guc+lokasyon otesinde bilgi tasiyor mu")
    say()
    lvl = (
        train.assign(z=np.log1p(train[C.TARGET].clip(lower=0)))
        .groupby(C.ENTITY)
        .agg(seviye=("z", "mean"), guc=(C.POWER, "first"), lok=(C.LOCATION, "first"))
    )
    lvl = lvl.loc[lvl["seviye"].notna()]
    lvl["log_guc"] = np.log(lvl["guc"].clip(lower=1))

    num = pd.to_numeric(lvl.index.to_series(), errors="coerce")
    say(f"sayisal olmayan tanim: {int(num.isna().sum())}")
    s = num.astype("Int64").astype("string").fillna("")
    for k in (2, 3, 4, 5):
        lvl[f"p{k}"] = s.str[:k].to_numpy()

    y = lvl["seviye"].to_numpy()
    tot = float(np.var(y))

    def r2_group(keys: list[str]) -> float:
        gm = lvl.groupby(keys, observed=True)["seviye"].transform("mean")
        return 1.0 - float(np.var(y - gm.to_numpy())) / tot

    def r2_lin_plus_group(keys: list[str] | None) -> float:
        # log_guc uzerine dogrusal + grup sabit etkisi
        X = [np.ones(len(lvl)), lvl["log_guc"].to_numpy()]
        if keys:
            d = pd.get_dummies(lvl[keys].astype("string").agg("|".join, axis=1))
            X = np.column_stack([np.column_stack(X), d.to_numpy(dtype="float64")])
        else:
            X = np.column_stack(X)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return 1.0 - float(np.var(y - X @ beta)) / tot

    say("## Trafo seviyesinin (ortalama log1p) aciklanan varyansi")
    say(f"{'anahtar':<28} {'R2':>8}  {'grup':>8}")
    rows = [
        ("guc (kategorik)", ["guc"]),
        ("lokasyon", ["lok"]),
        ("guc + lokasyon", ["guc", "lok"]),
        ("tanim onek 2", ["p2"]),
        ("tanim onek 3", ["p3"]),
        ("tanim onek 4", ["p4"]),
        ("tanim onek 5", ["p5"]),
        ("guc + onek 3", ["guc", "p3"]),
        ("guc + onek 4", ["guc", "p4"]),
        ("guc + lokasyon + onek 3", ["guc", "lok", "p3"]),
        ("guc + lokasyon + onek 4", ["guc", "lok", "p4"]),
    ]
    for name, keys in rows:
        n_grp = lvl.groupby(keys, observed=True).ngroups
        say(f"{name:<28} {r2_group(keys):8.4f}  {n_grp:8d}")
    say()
    say("Uyari: cok gruplu anahtarlarda R2 ornek-ici sisli. Asagida ayni")
    say("olcum, cold benzeri bir bolmede (trafolarin %30'u dislanarak).")
    say()

    rng = np.random.default_rng(C.SEED)
    ent = lvl.index.to_numpy()
    hold = rng.choice(len(ent), size=int(0.3 * len(ent)), replace=False)
    mask = np.zeros(len(ent), dtype=bool)
    mask[hold] = True
    tr_l, te_l = lvl.loc[~mask], lvl.loc[mask]
    gm_all = float(tr_l["seviye"].mean())
    tot_te = float(np.var(te_l["seviye"]))

    def oos(keys: list[str], prior: float = 10.0) -> float:
        g = tr_l.groupby(keys, observed=True)["seviye"].agg(["mean", "size"])
        # ampirik Bayes ile globale cekilmis grup ortalamasi
        g["sm"] = (g["mean"] * g["size"] + gm_all * prior) / (g["size"] + prior)
        idx = te_l[keys].astype("string").agg("|".join, axis=1) if len(keys) > 1 else te_l[keys[0]]
        gi = g.copy()
        gi.index = (
            gi.index.to_frame().astype("string").agg("|".join, axis=1)
            if len(keys) > 1
            else gi.index
        )
        p = pd.Series(idx).map(gi["sm"]).fillna(gm_all).to_numpy()
        return 1.0 - float(np.var(te_l["seviye"].to_numpy() - p)) / tot_te

    say(f"{'anahtar':<28} {'OOS R2':>8}")
    for name, keys in rows:
        say(f"{name:<28} {oos(keys):8.4f}")
    say()

    # test'teki cold trafolarin onekleri train'de var mi
    cold = sorted(set(test[C.ENTITY]) - set(train[C.ENTITY]))
    cn = pd.to_numeric(pd.Series(cold), errors="coerce").astype("Int64").astype("string").fillna("")
    for k in (3, 4):
        tr_pref = set(lvl[f"p{k}"])
        cover = float(cn.str[:k].isin(tr_pref).mean())
        say(f"cold trafolarin onek-{k} kapsamasi (train'de ayni onekten trafo var): {cover:.3f}")
    say()


def h4_yoy_shape(train: pd.DataFrame) -> None:
    say("# H4 — Trafoya ozgu mevsimsel sekil yildan yila tekrar ediyor mu")
    say()
    t = train.copy()
    t["z"] = np.log1p(t[C.TARGET].clip(lower=0))
    t["ym"] = t[C.DATE].dt.to_period("M")
    m = t.groupby([C.ENTITY, "ym"])["z"].mean().unstack()

    def dev(a, b):
        """a ayinin b ayina gore trafoya ozgu sapmasi (global kaymadan arindirilmis)."""
        d = m[pd.Period(a)] - m[pd.Period(b)]
        return d - d.mean()

    say("## Ayni ay ciftinin iki yildaki trafo-bazli sapmasinin korelasyonu")
    say(f"{'cift':<22} {'n':>6} {'pearson':>9} {'spearman':>9}")
    for a1, b1, a2, b2 in [
        ("2025-02", "2025-01", "2026-02", "2026-01"),
        ("2025-03", "2025-01", "2026-03", "2026-01"),
        ("2025-03", "2025-02", "2026-03", "2026-02"),
    ]:
        d1, d2 = dev(a1, b1), dev(a2, b2)
        ok = d1.notna() & d2.notna()
        say(f"{a1[-2:]}-{b1[-2:]} 25 vs 26{'':<10} {int(ok.sum()):6d} "
            f"{d1[ok].corr(d2[ok]):9.4f} {d1[ok].corr(d2[ok], method='spearman'):9.4f}")
    say()

    say("## Trafo seviyesinin yildan yila kalıcılığı (ayni ay)")
    say(f"{'ay':<10} {'n':>6} {'pearson':>9}")
    for a, b in [("2025-01", "2026-01"), ("2025-02", "2026-02"), ("2025-03", "2026-03")]:
        x, y = m[pd.Period(a)], m[pd.Period(b)]
        ok = x.notna() & y.notna()
        say(f"{a[-2:]}->{b[-2:]}{'':<4} {int(ok.sum()):6d} {x[ok].corr(y[ok]):9.4f}")
    say()

    say("## Nis-Tem 2025 capasinin gucu: Oca-Mar seviyesinden ne kadar sapiyor")
    q1_25 = m[[pd.Period(p) for p in ("2025-01", "2025-02", "2025-03")]].mean(axis=1)
    q2_25 = m[[pd.Period(p) for p in ("2025-04", "2025-05", "2025-06", "2025-07")]].mean(axis=1)
    q1_26 = m[[pd.Period(p) for p in ("2026-01", "2026-02", "2026-03")]].mean(axis=1)
    delta = q2_25 - q1_25
    ok = delta.notna() & q1_26.notna()
    say(f"trafo sayisi (her uc pencerede de veri var): {int(ok.sum()):,}")
    say(f"Nis-Tem 2025 eksi Oca-Mar 2025 farki: ortalama {delta[ok].mean():.4f}, "
        f"std {delta[ok].std():.4f}, medyan {delta[ok].median():.4f}")
    say(f"farkin %10-%90 araligi: {delta[ok].quantile(0.1):.3f} .. "
        f"{delta[ok].quantile(0.9):.3f}")
    say()
    say("Bu farkin trafoya ozgu kismi gercekse, gonderimde Oca-Mar 2026")
    say("seviyesine eklenerek Nis-Tem 2026 seviyesi kestirilebilir. Ustteki")
    say("kis-ici korelasyonlar bu terimin ne kadar guvenilir oldugunu soyler.")
    say()


def main() -> None:
    train = load_train()
    test = load_test()
    say("# Panel/kod/kohort/YoY sinyal taramasi")
    say(f"Uretim: {pd.Timestamp.now()}")
    say()
    h1_panel_pattern(train)
    h2_bulk_entry(train, test)
    h3_tanim_prefix(train, test)
    h4_yoy_shape(train)

    p = C.REPORTS_DIR / "60_panel_signal.md"
    p.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
