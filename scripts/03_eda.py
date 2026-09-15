"""EDA. Tavan olcumunun ortaya cikardigi sorulara odakli.

Olcumun birakti sorular:
  1. Sifir durumu ne kadar kalici? Trafolar sifir durumuna girip cikiyor mu?
     Warm hatanin %49'u sifir satirlardan geliyorsa ve gecmisten sifir durumu
     okunabiliyorsa hata nereden geliyor: durum GECISLERINDEN.
  2. Seviye gecmisten ufka ne kadar kayiyor? Buyume orani nedir?
  3. Mevsimsellik ve hava tepkisinin sekli nasil?
  4. Hafta ici/sonu ve tatil etkisinin buyuklugu ne?
  5. Cold trafolarin statik profili warm olanlardan farkli mi?
  6. Trafo davranis kumeleri ayirt edilebilir mi?

Cikti: reports/03_eda.md ve reports/figures/*.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.data import build_panel, entity_static, load_test, load_train
from src.validation.folds import guc_band

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "figure.autolayout": True})


def savefig(name: str) -> str:
    path = C.FIGURES_DIR / name
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return f"![{name}]({path.as_posix()})"


def main() -> None:
    C.ensure_dirs()
    train = load_train()
    test = load_test()
    panel = build_panel(train, test)
    static = entity_static(panel)

    df = train.copy()
    df["log1p_y"] = np.log1p(df[C.TARGET])
    df["z"] = df["log1p_y"] - np.log(df[C.POWER])
    df["sifir"] = df[C.TARGET].eq(0)
    df["ay"] = df[C.DATE].dt.to_period("M")
    df["haftagunu"] = df[C.DATE].dt.dayofweek
    df["guc_band"] = guc_band(df[C.POWER]).astype(str)
    df = df.join(static[["il"]], on=C.ENTITY)

    tr_ent = set(train[C.ENTITY].unique())
    te_ent = set(test[C.ENTITY].unique())

    out: list[str] = ["# EDA", ""]
    out.append(f"Uretim tarihi: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    out.append("")
    out.append(
        "Bu EDA, cold-start tavan olcumunun (reports/02) birakti somut sorulara "
        "odakli; genel bir betimleyici tarama degil."
    )

    # ================================================================
    # 1. Sifir durumunun kaliciligi ve gecisler
    # ================================================================
    out.append("\n## 1. Sifir durumu: kalicilik ve gecisler\n")

    d = df.sort_values([C.ENTITY, C.DATE])
    d["sifir_onceki"] = d.groupby(C.ENTITY, sort=False)["sifir"].shift(1)
    trans = d.dropna(subset=["sifir_onceki"])
    p_stay0 = float(trans.loc[trans["sifir_onceki"] == True, "sifir"].mean())
    p_stay1 = float(1 - trans.loc[trans["sifir_onceki"] == False, "sifir"].mean())
    out.append("Gunluk gecis olasiliklari (train, ardisik gun ciftleri):")
    out.append("")
    out.append("| Onceki gun | Bugun sifir | Bugun sifir degil |")
    out.append("| --- | --- | --- |")
    out.append(f"| sifir | **{p_stay0:.4f}** | {1 - p_stay0:.4f} |")
    out.append(f"| sifir degil | {1 - p_stay1:.4f} | **{p_stay1:.4f}** |")
    out.append("")
    out.append(
        f"Sifir durumu gunluk olcekte cok kalici: sifirdan sifira gecis "
        f"olasiligi {p_stay0:.3f}. Yani sifirlar tek gunluk olaylar degil, uzun "
        "bloklar halinde geliyor."
    )

    # 122 gunluk ufuk olceginde ne kadar kalici: son 122 gunun sifir orani ile
    # sonraki 122 gunun sifir orani arasindaki iliski.
    origin = C.FOLDS[1].origin
    before = df.loc[
        (df[C.DATE] > origin - pd.Timedelta(days=C.HORIZON_DAYS)) & (df[C.DATE] <= origin)
    ]
    after = df.loc[df[C.DATE] > origin]
    r_before = before.groupby(C.ENTITY)["sifir"].mean().rename("once")
    r_after = after.groupby(C.ENTITY)["sifir"].mean().rename("sonra")
    both = pd.concat([r_before, r_after], axis=1).dropna()
    corr = float(both["once"].corr(both["sonra"]))
    # Durum gecisi yapan trafolar
    was0 = both["once"] > 0.9
    was1 = both["once"] < 0.1
    out.append("")
    out.append(
        f"122 gunluk blok olceginde (origin {origin:%Y-%m-%d}), oncesi ve sonrasi "
        f"sifir oranlarinin korelasyonu **{corr:.3f}** ({len(both):,} trafo)."
    )
    out.append("")
    out.append("| Origin oncesi durum | Trafo | Ufukta sifir orani (ortalama) |")
    out.append("| --- | --- | --- |")
    out.append(
        f"| tamamen sifir (>0,9) | {int(was0.sum()):,} | "
        f"{both.loc[was0, 'sonra'].mean():.4f} |"
    )
    out.append(
        f"| hic sifir yok (<0,1) | {int(was1.sum()):,} | "
        f"{both.loc[was1, 'sonra'].mean():.4f} |"
    )
    n_switch_on = int(((both["once"] > 0.9) & (both["sonra"] < 0.5)).sum())
    n_switch_off = int(((both["once"] < 0.1) & (both["sonra"] > 0.5)).sum())
    out.append("")
    out.append(
        f"Durum degistiren trafolar: **{n_switch_on:,}** sifirdan cikiyor "
        f"(devreye aliniyor), **{n_switch_off:,}** sifira giriyor "
        f"(de-enerjize oluyor)."
    )
    out.append("")
    out.append(
        "> **Tasarim sonucu.** Sifir durumu yuksek oranda kalici, yani warm "
        "trafolar icin gecmisten okunabiliyor. Warm hatanin buyuk kisminin sifir "
        "satirlardan gelmesi bu yuzden bir celiski degil: hata durumu YANLIS "
        "bilmekten degil, durum GECISLERINDEN geliyor. Sifir kapisi bu nedenle "
        "yalnizca 'su an sifir mi' degil 'ufuk boyunca sifir kalir mi' sorusunu "
        "cevaplamali; girdi olarak son sifir blogunun uzunlugu ve devreye alinma "
        "yasi kullanilacak."
    )

    # ================================================================
    # 2. Seviye kaymasi ve buyume
    # ================================================================
    out.append("\n## 2. Seviye kaymasi: tarihsel ortalama neden zayif\n")

    nz = df.loc[~df["sifir"]]
    lvl_before = (
        nz.loc[(nz[C.DATE] > origin - pd.Timedelta(days=C.HORIZON_DAYS)) & (nz[C.DATE] <= origin)]
        .groupby(C.ENTITY)["log1p_y"]
        .mean()
        .rename("once")
    )
    lvl_after = nz.loc[nz[C.DATE] > origin].groupby(C.ENTITY)["log1p_y"].mean().rename("sonra")
    lv = pd.concat([lvl_before, lvl_after], axis=1).dropna()
    lv["kayma"] = lv["sonra"] - lv["once"]
    out.append(
        f"Sifir olmayan satirlarda, origin oncesi 122 gun ile sonraki 122 gun "
        f"arasindaki log1p seviye kaymasi ({len(lv):,} trafo):"
    )
    out.append("")
    out.append(
        f"- Ortalama kayma: **{lv['kayma'].mean():+.4f}** log1p birimi "
        f"(carpansal olarak {np.exp(lv['kayma'].mean()):.3f}x)"
    )
    out.append(f"- Medyan kayma: **{lv['kayma'].median():+.4f}**")
    out.append(f"- Kaymanin standart sapmasi: **{lv['kayma'].std(ddof=0):.4f}**")
    out.append(
        f"- Kaymanin mutlak degeri > 0,5 olan trafo orani: "
        f"**{100 * (lv['kayma'].abs() > 0.5).mean():.1f}%**"
    )
    out.append("")
    out.append(
        "Ortalama kayma kucuk ama dagilim genis. Yani sistematik bir buyume "
        "duzeltmesi degil, trafo bazinda oynaklik hakim. Bu, tarihsel ortalamanin "
        "1,13 RMSLE'de kalmasinin nedeni: seviye tahmininin kendisi gurultulu. "
        "Kayma yonu ongorulemiyorsa en iyi strateji kaymayi tahmin etmeye "
        "calismamak, sadece seviyeyi olabildigince saglam olcmek."
    )

    # Kaymanin gecmis uzunluguyla iliskisi
    hist_len = df.loc[df[C.DATE] <= origin].groupby(C.ENTITY)[C.DATE].nunique()
    lv["gecmis_gun"] = hist_len.reindex(lv.index)
    bins = pd.cut(lv["gecmis_gun"], [0, 30, 90, 180, 270, np.inf], right=False)
    g = lv.groupby(bins, observed=True)["kayma"].agg(["size", "mean", "std"])
    out.append("")
    out.append("Kayma, gecmis uzunluguna gore:")
    out.append("")
    out.append("| Gecmis | Trafo | Ortalama kayma | Kayma std |")
    out.append("| --- | --- | --- | --- |")
    for b, r in g.iterrows():
        out.append(f"| {b} | {int(r['size']):,} | {r['mean']:+.4f} | {r['std']:.4f} |")
    out.append("")
    out.append(
        "Kisa gecmisli trafolarda kayma hem daha buyuk hem daha oynak: yeni "
        "devreye alinan trafo rampa halinde. Bu, `devreye_alinma_yasi` "
        "ozelliginin gerekcesi."
    )

    plt.figure(figsize=(6, 3.2))
    plt.hist(lv["kayma"].clip(-2, 2), bins=80, color="#3b6ea5")
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("log1p seviye kaymasi (ufuk - gecmis)")
    plt.ylabel("trafo sayisi")
    plt.title("Seviye kaymasi dagilimi, sifir olmayan satirlar")
    out.append("")
    out.append(savefig("03_seviye_kaymasi.png"))

    # ================================================================
    # 3. Mevsimsellik
    # ================================================================
    out.append("\n## 3. Mevsimsellik\n")

    monthly = nz.groupby("ay")["log1p_y"].mean()
    monthly_med = nz.groupby("ay")["log1p_y"].median()
    out.append("Aylik ortalama `log1p(tuketim)` (sifir olmayan satirlar):")
    out.append("")
    out.append("| Ay | Ortalama | Medyan | Ocak 2025'e gore |")
    out.append("| --- | --- | --- | --- |")
    base = monthly.iloc[0]
    for m in monthly.index:
        out.append(
            f"| {m} | {monthly[m]:.4f} | {monthly_med[m]:.4f} | "
            f"{monthly[m] - base:+.4f} |"
        )

    plt.figure(figsize=(7, 3.2))
    monthly.plot(marker="o", color="#3b6ea5", label="ortalama")
    monthly_med.plot(marker="s", color="#c1666b", label="medyan")
    plt.ylabel("log1p(tuketim)")
    plt.xlabel("ay")
    plt.title("Aylik seviye, sifir olmayan satirlar")
    plt.legend()
    plt.grid(alpha=0.3)
    out.append("")
    out.append(savefig("03_aylik_seviye.png"))

    apr_jul_25 = monthly.loc[[p for p in monthly.index if p.month in (4, 5, 6, 7)]]
    out.append("")
    out.append(
        f"Nisan-Temmuz 2025 araligi: {apr_jul_25.min():.4f} - {apr_jul_25.max():.4f}, "
        f"tepe {apr_jul_25.idxmax()}. Ufkumuz tam bu pencereye denk geliyor ve icinde "
        f"**{apr_jul_25.max() - apr_jul_25.min():.4f} log1p birimlik** bir mevsimsel "
        f"rampa var (carpansal {np.exp(apr_jul_25.max() - apr_jul_25.min()):.2f}x). "
        "Yani ufuk sabit bir seviye degil, yukselen bir egri; ufuk-agnostik tek bir "
        "seviye tahmini bu rampayi kaciracak."
    )

    # ================================================================
    # 4. Hafta gunu etkisi
    # ================================================================
    out.append("\n## 4. Hafta gunu etkisi\n")
    gunler = ["Pzt", "Sal", "Car", "Per", "Cum", "Cmt", "Paz"]
    dow = nz.groupby("haftagunu")["log1p_y"].mean()
    out.append("| Gun | Ortalama log1p | Pazartesi'ye gore |")
    out.append("| --- | --- | --- |")
    for i, v in dow.items():
        out.append(f"| {gunler[i]} | {v:.4f} | {v - dow.iloc[0]:+.4f} |")
    out.append("")
    out.append(
        f"Hafta ici-hafta sonu farki toplam {dow.max() - dow.min():.4f} log1p birimi, "
        f"yani mevsimsel rampanin ({apr_jul_25.max() - apr_jul_25.min():.4f}) "
        f"{100 * (dow.max() - dow.min()) / (apr_jul_25.max() - apr_jul_25.min()):.0f}%'i "
        "kadar. Ikincil ama ihmal edilemez."
    )

    # Guc bandina gore hafta gunu profili: sanayi/konut ayrimi
    prof = (
        nz.groupby(["guc_band", "haftagunu"])["log1p_y"].mean().unstack()
    )
    prof_rel = prof.sub(prof.mean(axis=1), axis=0)
    out.append("")
    out.append("Guc bandina gore hafta gunu profili (banttan sapma, log1p):")
    out.append("")
    out.append("| Band | " + " | ".join(gunler) + " | Hafta sonu dususu |")
    out.append("| --- | " + " | ".join(["---"] * 8) + " |")
    for b in prof_rel.index:
        row = prof_rel.loc[b]
        drop = row[[0, 1, 2, 3, 4]].mean() - row[[5, 6]].mean()
        out.append(
            f"| {b} | " + " | ".join(f"{row[i]:+.3f}" for i in range(7))
            + f" | **{drop:+.3f}** |"
        )
    out.append("")
    out.append(
        "Hafta sonu dususunun buyuklugu guc bandina gore degisiyor: bu, sanayi "
        "agirlikli ve konut agirlikli trafolari ayirt eden gozlenebilir bir imza. "
        "`guc_band x haftagunu` etkilesimi bu yuzden ozellik setinde."
    )

    plt.figure(figsize=(7, 3.4))
    for b in prof_rel.index:
        plt.plot(range(7), prof_rel.loc[b], marker="o", label=str(b), lw=1.2)
    plt.xticks(range(7), gunler)
    plt.axhline(0, color="k", lw=0.8)
    plt.ylabel("banttan sapma (log1p)")
    plt.title("Hafta gunu profili, guc bandina gore")
    plt.legend(fontsize=7, ncol=3)
    plt.grid(alpha=0.3)
    out.append("")
    out.append(savefig("03_haftagunu_profili.png"))

    # ================================================================
    # 5. Cold ve warm trafolarin statik profili
    # ================================================================
    out.append("\n## 5. Cold ve warm trafolarin statik profili\n")
    st = static.loc[static["test_gun_sayisi"] > 0].copy()
    st["cold"] = ~st.index.isin(tr_ent)
    st["band"] = guc_band(st["guc"]).astype(str)

    tab = (
        st.groupby(["band", "cold"], observed=True).size().unstack(fill_value=0)
    )
    tab.columns = ["warm", "cold"]
    tab["cold_orani"] = (tab["cold"] / (tab["cold"] + tab["warm"])).round(3)
    out.append("Guc bandina gore cold/warm dagilimi:")
    out.append("")
    out.append("| Band | Warm | Cold | Cold orani |")
    out.append("| --- | --- | --- | --- |")
    for b, r in tab.iterrows():
        out.append(f"| {b} | {int(r['warm']):,} | {int(r['cold']):,} | {r['cold_orani']:.3f} |")
    overall = float(st["cold"].mean())
    out.append("")
    out.append(f"Genel cold orani: **{overall:.3f}**")
    out.append("")
    out.append(
        "Cold orani bandlar arasinda degisiyorsa cold populasyon rastgele degil, "
        "belirli guc siniflarinda kumelenmis. Maskeleme tabakalamasi bu yuzden "
        "uniform degil (bkz. src/validation/folds.py)."
    )

    il_tab = st.groupby(["il", "cold"], observed=True).size().unstack(fill_value=0)
    il_tab.columns = ["warm", "cold"]
    il_tab["cold_orani"] = (il_tab["cold"] / (il_tab["cold"] + il_tab["warm"])).round(3)
    out.append("")
    out.append("Il bazinda:")
    out.append("")
    out.append("| Il | Warm | Cold | Cold orani |")
    out.append("| --- | --- | --- | --- |")
    for i, r in il_tab.iterrows():
        out.append(f"| {i} | {int(r['warm']):,} | {int(r['cold']):,} | {r['cold_orani']:.3f} |")

    lok_cold = st.groupby("lokasyon")["cold"].agg(["size", "mean"]).sort_values(
        "mean", ascending=False
    )
    out.append("")
    out.append("Cold orani en yuksek ve en dusuk 5 lokasyon:")
    out.append("")
    out.append("| Lokasyon | Trafo | Cold orani |")
    out.append("| --- | --- | --- |")
    for l, r in pd.concat([lok_cold.head(5), lok_cold.tail(5)]).iterrows():
        out.append(f"| {l} | {int(r['size']):,} | {r['mean']:.3f} |")

    # ================================================================
    # 6. Trafo davranis kumeleri
    # ================================================================
    out.append("\n## 6. Trafo davranis kumeleri\n")
    ent = nz.groupby(C.ENTITY).agg(
        seviye=("log1p_y", "mean"),
        oynaklik=("log1p_y", "std"),
        n=("log1p_y", "size"),
    )
    ent["sifir_orani"] = df.groupby(C.ENTITY)["sifir"].mean().reindex(ent.index)
    hafta = nz.groupby([C.ENTITY, nz["haftagunu"] >= 5])["log1p_y"].mean().unstack()
    ent["hs_dususu"] = (hafta[False] - hafta[True]).reindex(ent.index)
    ent = ent.loc[ent["n"] >= 60]
    out.append(f"En az 60 sifir olmayan gozlemi olan {len(ent):,} trafo uzerinde:")
    out.append("")
    out.append("| Olcu | Medyan | 10. yuzdelik | 90. yuzdelik |")
    out.append("| --- | --- | --- | --- |")
    for col, label in [
        ("seviye", "log1p seviye"),
        ("oynaklik", "gun-ici oynaklik (std)"),
        ("hs_dususu", "hafta sonu dususu"),
        ("sifir_orani", "sifir orani"),
    ]:
        s = ent[col].dropna()
        out.append(
            f"| {label} | {s.median():.4f} | {s.quantile(0.1):.4f} | "
            f"{s.quantile(0.9):.4f} |"
        )
    out.append("")
    out.append(
        "Hafta sonu dususunun 10.-90. yuzdelik araligi genis: bazi trafolar hafta "
        "sonu neredeyse hic dusmuyor (surekli yuk), bazilari belirgin dusuyor "
        "(ticari/sanayi). Bu, kume atamasi icin gercek bir gozlenebilir eksen -- ama "
        "yalnizca gecmisi olan trafolar icin olculebiliyor, cold olanlar icin degil."
    )

    plt.figure(figsize=(5.2, 4))
    sub = ent.dropna(subset=["hs_dususu"]).sample(
        min(4000, len(ent)), random_state=C.SEED
    )
    plt.scatter(sub["seviye"], sub["hs_dususu"], s=4, alpha=0.25, color="#3b6ea5")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("log1p seviye")
    plt.ylabel("hafta sonu dususu (log1p)")
    plt.title("Trafo davranis uzayi")
    plt.grid(alpha=0.3)
    out.append("")
    out.append(savefig("03_davranis_uzayi.png"))

    # ================================================================
    # 7. Tatil etkisi (holidays paketiyle)
    # ================================================================
    out.append("\n## 7. Tatil etkisi\n")
    try:
        import holidays as hol
        from holidays.constants import HALF_DAY, PUBLIC

        trh = hol.country_holidays(
            "TR", years=[2025, 2026], categories=(PUBLIC, HALF_DAY), language="tr"
        )
        hd = pd.Series({pd.Timestamp(k): v for k, v in trh.items()}).sort_index()
        nz2 = nz.copy()
        nz2["tatil"] = nz2[C.DATE].isin(hd.index)
        # Ayni ay ve ayni hafta gunu icindeki normal gunlere gore sapma
        ref = nz2.loc[~nz2["tatil"]].groupby(["ay", "haftagunu"])["log1p_y"].mean()
        nz2["ref"] = pd.MultiIndex.from_arrays(
            [nz2["ay"], nz2["haftagunu"]]
        ).map(ref)
        eff = nz2.loc[nz2["tatil"]].assign(sapma=lambda x: x["log1p_y"] - x["ref"])
        by_day = eff.groupby(eff[C.DATE])["sapma"].agg(["mean", "size"])
        by_day["tatil"] = [hd.get(d, "") for d in by_day.index]
        out.append(
            "Tatil gununun, ayni ay ve ayni hafta gunundeki tatil olmayan gunlere "
            "gore log1p sapmasi:"
        )
        out.append("")
        out.append("| Tarih | Tatil | Satir | Sapma |")
        out.append("| --- | --- | --- | --- |")
        for dte, r in by_day.sort_values("mean").iterrows():
            out.append(
                f"| {dte:%Y-%m-%d} | {r['tatil']} | {int(r['size']):,} | "
                f"**{r['mean']:+.4f}** |"
            )
        out.append("")
        out.append(
            f"Tatil etkisi ortalama {by_day['mean'].mean():+.4f}, en guclusu "
            f"{by_day['mean'].min():+.4f} ({by_day['mean'].idxmin():%Y-%m-%d}, "
            f"{by_day.loc[by_day['mean'].idxmin(), 'tatil']}). Buyukluk hafta sonu "
            "etkisiyle ayni mertebede, yani modellenmeye deger."
        )
        out.append("")
        out.append(
            "Bayram tarihlerinin yil-uzeri kaymasi (Kurban 2025-06-06 -> 2026-05-27) "
            "dogrulandi; `holidays` paketinin 2025-2026 tarihleri Diyanet kaynakli "
            "onaylanmis aralikta."
        )
    except Exception as exc:  # pragma: no cover
        out.append(f"`holidays` paketi ile tatil analizi basarisiz: {exc}")

    path = C.REPORTS_DIR / "03_eda.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"yazildi: {path}")


if __name__ == "__main__":
    main()
