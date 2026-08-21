"""Veri kalitesi denetimi.

Hedefte sifir/negatif/NaN, tekrarli trafo-gun, guc ve lokasyon tutarliligi,
panel giris/cikis desenleri ve 2026-05-11 toplu giris artifakti.

Cikti: reports/01_data_quality.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.data import build_panel, entity_static, load_test, load_train
from src.validation.folds import guc_band


def section(title: str) -> str:
    return f"\n## {title}\n"


def main() -> None:
    C.ensure_dirs()
    train = load_train()
    test = load_test()
    panel = build_panel(train, test)
    static = entity_static(panel)

    out: list[str] = ["# Veri Kalitesi Denetimi", ""]
    out.append(f"Uretim tarihi: {pd.Timestamp.now():%Y-%m-%d %H:%M}")

    # --- Boyut ve kapsam ---
    out.append(section("Boyut ve kapsam"))
    for name, df in (("train", train), ("test", test)):
        out.append(
            f"- **{name}**: {len(df):,} satir, {df[C.ENTITY].nunique():,} trafo, "
            f"{df[C.DATE].min():%Y-%m-%d} - {df[C.DATE].max():%Y-%m-%d} "
            f"({df[C.DATE].nunique()} tekil gun)"
        )

    # --- Hedef degerleri: en kritik kontrol ---
    y = train[C.TARGET]
    n_nan = int(y.isna().sum())
    n_neg = int((y < 0).sum())
    n_zero = int((y == 0).sum())
    n_tiny = int(((y > 0) & (y < 1)).sum())

    out.append(section("Hedef (`tuketim`) degerleri"))
    out.append(f"- NaN: **{n_nan:,}** ({100 * n_nan / len(y):.3f}%)")
    out.append(f"- Negatif: **{n_neg:,}** ({100 * n_neg / len(y):.3f}%)")
    out.append(f"- Tam sifir: **{n_zero:,}** ({100 * n_zero / len(y):.3f}%)")
    out.append(f"- 0 < x < 1: **{n_tiny:,}** ({100 * n_tiny / len(y):.3f}%)")
    if n_neg:
        neg = y[y < 0]
        out.append(
            f"- Negatiflerin araligi: {neg.min():.3f} .. {neg.max():.3f}, "
            f"medyan {neg.median():.3f}"
        )
        out.append(
            f"- Negatif iceren trafo sayisi: "
            f"{train.loc[y < 0, C.ENTITY].nunique():,}"
        )
        out.append("")
        out.append(
            "> **Karar gerektirir.** `log1p` -1'in altinda tanimsiz. Negatif degerler "
            "ya olcum hatasi ya da sayac arkasi uretim kaynakli geri besleme. "
            "Yarisma metrigi negatif TAHMINLERI 0 sayiyor ama negatif GERCEK "
            "degerlerin nasil skorlandigi belirsiz; gercek test setinde de negatif "
            "varsa metrik hesabi bizim kontrolumuz disinda."
        )

    # Sifir/dusuk tuketimin RMSLE'deki kaldiraci
    if n_zero:
        out.append("")
        out.append(
            f"- Sifir satirlarin RMSLE kaldiraci: gercek 0'a karsi 500 kWh tahmini "
            f"tek satirda {np.log1p(500) ** 2:.1f} kare-log hata uretir."
        )

    # --- Sifirlarin trafo bazinda yogunlasmasi ---
    out.append(section("Sifirlarin dagilimi: gurultu mu de-enerjizasyon mu"))
    z = train.assign(_z=y.eq(0)).groupby(C.ENTITY)["_z"].agg(["sum", "size"])
    z["oran"] = z["sum"] / z["size"]
    n_any = int((z["sum"] > 0).sum())
    n_all = int((z["oran"] == 1).sum())
    n_most = int((z["oran"] > 0.5).sum())
    out.append(f"- Hic sifir icermeyen trafo: **{len(z) - n_any:,}** / {len(z):,}")
    out.append(f"- En az bir sifir iceren trafo: **{n_any:,}**")
    out.append(f"- TAMAMI sifir olan trafo: **{n_all:,}**")
    out.append(f"- Sifir orani > 0,5 olan trafo: **{n_most:,}**")
    zero_in_all = int(z.loc[z["oran"] == 1, "size"].sum())
    out.append(
        f"- Tamami sifir trafolarin kapsadigi satir: **{zero_in_all:,}** "
        f"({100 * zero_in_all / n_zero:.1f}% tum sifirlarin)"
    )
    out.append("")
    out.append(
        "> **Tasarim sonucu.** Sifirlar dagilmis gurultu degil, trafo duzeyinde "
        "yogunlasmis: sifirlarin buyuk kismi kalici olarak de-enerjize trafolardan "
        "geliyor. Bu, sifir kapisini satir duzeyinde bir olay tahmini yerine "
        "**varlik duzeyinde bir durum siniflandirmasi** yapiyor. Warm trafolarda "
        "durum gecmisten neredeyse kesin okunur; risk tamamen cold trafolarda, "
        "cunku orada durumu yalnizca statik nitelikler ve panel uyeligi desenleri "
        "ima edebilir."
    )

    # Cold trafolarda ayni oran ne olabilir: train'de yeni giren trafolarin
    # ilk 30 gunundeki sifir orani, cold segment icin en iyi vekil.
    first_seen_tr = train.groupby(C.ENTITY)[C.DATE].min()
    late = first_seen_tr[first_seen_tr > C.TRAIN_START + pd.Timedelta(days=30)].index
    if len(late):
        early = train.loc[train[C.ENTITY].isin(set(late))].merge(
            first_seen_tr.rename("ilk").reset_index(), on=C.ENTITY, how="left"
        )
        early = early.loc[(early[C.DATE] - early["ilk"]).dt.days < 30]
        out.append("")
        out.append(
            f"- Vekil olcum: train icinde sonradan devreye giren {len(late):,} "
            f"trafonun ilk 30 gunundeki sifir orani "
            f"**{100 * early[C.TARGET].eq(0).mean():.2f}%** "
            f"(genel oran {100 * n_zero / len(y):.2f}%). Yeni devreye alinan "
            "trafolarda sifir riski farkli; cold segment tahmininde bu oran "
            "referans alinacak."
        )

    # --- Tekrarli trafo-gun ---
    out.append(section("Tekrarli (trafo, gun) ciftleri"))
    for name, df in (("train", train), ("test", test)):
        d = int(df.duplicated([C.ENTITY, C.DATE]).sum())
        out.append(f"- {name}: **{d:,}**")
        if d:
            dup_rows = df[df.duplicated([C.ENTITY, C.DATE], keep=False)]
            n_conflict = 0
            if C.TARGET in df.columns:
                agg = dup_rows.groupby([C.ENTITY, C.DATE])[C.TARGET].nunique()
                n_conflict = int((agg > 1).sum())
            out.append(
                f"  - etkilenen trafo: {dup_rows[C.ENTITY].nunique():,}, "
                f"celisen hedefli cift: {n_conflict:,}"
            )

    # --- Statik nitelik tutarliligi ---
    out.append(section("Statik nitelik tutarliligi"))
    g = panel.groupby(C.ENTITY, sort=False)
    guc_var = int((g[C.POWER].nunique() > 1).sum())
    lok_var = int((g[C.LOCATION].nunique() > 1).sum())
    out.append(f"- `guc` birden fazla degere sahip trafo: **{guc_var}**")
    out.append(f"- `lokasyon` birden fazla degere sahip trafo: **{lok_var}**")
    out.append(
        "- Ikisi de 0 ise test donemi statik degerini egitim satirinda kullanmak "
        "ileri bakis yaratmaz."
    )

    # --- Lokasyon hiyerarsisi ---
    out.append(section("Lokasyon hiyerarsisi"))
    out.append(f"- Tekil `lokasyon` degeri: **{panel[C.LOCATION].nunique()}**")
    depth = panel.groupby("lokasyon_derinlik")[C.LOCATION].nunique()
    for d, n in depth.items():
        out.append(f"  - {d} seviyeli: {n} tekil deger")
    il_counts = static["il"].value_counts()
    out.append("- Trafo sayisi il bazinda:")
    for il, n in il_counts.items():
        out.append(f"  - {il}: {n:,}")
    n_ilce = static["ilce"].nunique(dropna=True)
    n_bolge = static["bolge"].nunique(dropna=True)
    out.append(f"- Tekil bolge: {n_bolge}, tekil ilce: {n_ilce}")
    n_no_ilce = int(static["ilce"].isna().sum())
    out.append(f"- `ilce` seviyesi olmayan trafo: **{n_no_ilce:,}**")
    out.append("- Derinlik il bazinda:")
    for (il, d), n in static.groupby(["il", "lokasyon_derinlik"]).size().items():
        out.append(f"  - {il}, {d} seviye: {n:,}")
    out.append("")
    out.append(
        "> **Tasarim sonucu.** Manisa kayitlari `IL>BOLGE` formatinda, yani ilce "
        "seviyesi hic yok; Izmir kayitlari `IL>BOLGE>ILCE`. Bu yuzden gruplama ve "
        "tabakalama anahtari `ilce` degil **`lokasyon`** (47 tekil deger): her iki "
        "ilde de mevcut olan en ince cozunurluk. `ilce` uzerinden tabakalamak "
        "1.788 Manisa trafosunun tamamini tek bir NA tabakasinda cokertirdi."
    )
    generic = panel.loc[panel["lokasyon_derinlik"] == 1, C.LOCATION].unique()
    if len(generic):
        out.append(f"- Tek seviyeli (jenerik) degerler: {list(generic)}")

    # --- guc dagilimi ---
    out.append(section("`guc` dagilimi"))
    vc = static["guc"].value_counts().sort_index()
    out.append(f"- Tekil deger sayisi: {len(vc)}")
    out.append(f"- Aralik: {static['guc'].min()} - {static['guc'].max()} kVA")
    out.append("- En yaygin 10 deger (trafo sayisi):")
    for v, n in static["guc"].value_counts().head(10).items():
        out.append(f"  - {v} kVA: {n:,}")
    big = static.loc[static["guc"] >= 2600]
    out.append(
        f"- **>= 2600 kVA: {len(big):,} varlik** (10-36 MVA). Bunlar dagitim "
        "trafosu degil, dagitim merkezi sinifi; ayri bir guc bandinda tutuluyor."
    )
    out.append("")
    out.append("Kullanilan guc bandlari ve trafo sayilari:")
    out.append("")
    out.append("| Band | Trafo | Test satiri |")
    out.append("| --- | --- | --- |")
    bands = guc_band(static["guc"])
    test_rows_by_ent = test.groupby(C.ENTITY).size()
    band_rows = (
        pd.DataFrame({"band": bands, "rows": test_rows_by_ent.reindex(static.index).fillna(0)})
        .groupby("band", observed=True)
        .agg(trafo=("rows", "size"), satir=("rows", "sum"))
    )
    for b, r in band_rows.iterrows():
        out.append(f"| {b} | {int(r.trafo):,} | {int(r.satir):,} |")

    # --- Panel giris/cikis ---
    out.append(section("Panel giris ve cikis desenleri"))
    tr_ent = set(train[C.ENTITY].unique())
    te_ent = set(test[C.ENTITY].unique())
    out.append(f"- Yalnizca train'de: **{len(tr_ent - te_ent):,}** trafo")
    out.append(f"- Yalnizca test'te (cold start): **{len(te_ent - tr_ent):,}** trafo")
    out.append(f"- Her ikisinde: **{len(tr_ent & te_ent):,}** trafo")

    cold_rows = int(test.loc[~test[C.ENTITY].isin(tr_ent)].shape[0])
    out.append(
        f"- Cold-start test satiri: **{cold_rows:,}** / {len(test):,} = "
        f"**{100 * cold_rows / len(test):.2f}%**"
    )

    # Gecmis uzunluguna gore test satir dagilimi
    hist = train.groupby(C.ENTITY)[C.DATE].nunique()
    te_ent_rows = test.groupby(C.ENTITY).size().rename("test_satir")
    seg = pd.DataFrame(te_ent_rows)
    seg["gecmis_gun"] = hist.reindex(seg.index).fillna(0)
    seg["segment"] = pd.cut(
        seg["gecmis_gun"],
        bins=list(C.HISTORY_SEGMENT_EDGES) + [np.inf],
        labels=list(C.HISTORY_SEGMENT_LABELS),
        right=False,
    )
    tab = seg.groupby("segment", observed=True).agg(
        trafo=("test_satir", "size"), test_satir=("test_satir", "sum")
    )
    tab["pay_%"] = (100 * tab["test_satir"] / tab["test_satir"].sum()).round(2)
    out.append("")
    out.append("Gecmis uzunluguna gore test dagilimi:")
    out.append("")
    out.append("| Segment | Trafo | Test satiri | Pay |")
    out.append("| --- | --- | --- | --- |")
    for s, r in tab.iterrows():
        out.append(f"| {s} | {int(r.trafo):,} | {int(r.test_satir):,} | {r['pay_%']:.2f}% |")

    # --- Gun ici bosluklar ---
    out.append(section("Trafo ici gun bosluklari (train)"))
    span = train.groupby(C.ENTITY)[C.DATE].agg(["min", "max", "nunique"])
    span["span_gun"] = (span["max"] - span["min"]).dt.days + 1
    span["bosluk_orani"] = 1 - span["nunique"] / span["span_gun"]
    out.append(f"- Ortalama bosluk orani: {span['bosluk_orani'].mean():.4f}")
    out.append(f"- Bosluksuz trafo: {int((span['bosluk_orani'] == 0).sum()):,}")
    out.append(
        f"- Bosluk orani > 0,5 olan trafo: "
        f"{int((span['bosluk_orani'] > 0.5).sum()):,}"
    )

    # --- 2026-05-11 artifakti ---
    out.append(section("2026-05-11 toplu giris artifakti"))
    first_seen = test.groupby(C.ENTITY)[C.DATE].min()
    top = first_seen.value_counts().head(8)
    out.append("Test'te ilk gorulme tarihine gore trafo sayisi (en yogun 8):")
    out.append("")
    out.append("| Tarih | Trafo | Bunlarin cold olani |")
    out.append("| --- | --- | --- |")
    for d, n in top.items():
        ents = first_seen[first_seen == d].index
        n_cold = int(sum(e not in tr_ent for e in ents))
        out.append(f"| {d:%Y-%m-%d} | {n:,} | {n_cold:,} |")

    rows_by_date = test.groupby(C.DATE).size()
    jump = rows_by_date.diff()
    biggest = jump.abs().nlargest(5)
    out.append("")
    out.append("Gunluk test satir sayisindaki en buyuk sicramalar:")
    for d, v in biggest.items():
        out.append(f"- {d:%Y-%m-%d}: {v:+.0f} satir (toplam {rows_by_date[d]:,})")

    train_rows_by_date = train.groupby(C.DATE).size()
    out.append("")
    out.append(
        f"Train gunluk satir sayisi: {train_rows_by_date.min():,} - "
        f"{train_rows_by_date.max():,} (ilk gun {train_rows_by_date.iloc[0]:,}, "
        f"son gun {train_rows_by_date.iloc[-1]:,}). Panel buyuyor."
    )

    # 2026-05-11'de giren gruptaki warm trafolar: train gecmisleri nerede bitiyor?
    may11 = first_seen[first_seen == pd.Timestamp("2026-05-11")].index
    warm_may11 = [e for e in may11 if e in tr_ent]
    out.append("")
    out.append(
        f"2026-05-11 grubundaki **{len(warm_may11):,} trafo train'de zaten var**. "
        "Train gecmislerinin bittigi tarih:"
    )
    if warm_may11:
        last_tr = train.loc[train[C.ENTITY].isin(set(warm_may11))].groupby(C.ENTITY)[
            C.DATE
        ].max()
        for d, n in last_tr.dt.to_period("M").value_counts().sort_index().items():
            out.append(f"- {d}: {n:,} trafo")
        still_alive = int((last_tr >= pd.Timestamp("2026-03-01")).sum())
        out.append("")
        out.append(
            f"Bunlarin **{still_alive:,}**'i Mart 2026'da hala aktif, yani test'te "
            "2026-04-01 - 2026-05-10 arasi 40 gunluk bir bosluk var ve sonra geri "
            "geliyorlar. Gercek bir devreye alma degil, **veri teslim artifakti**: "
            "test paneli iki partide olusturulmus. Modelleme sonucu: bu trafolarin "
            "gecmisi tam ve kullanilabilir; 2026-05-11 tarihini bir rejim degisimi "
            "sinyali olarak yorumlamak hata olur. Buna karsilik ayni gun giren "
            f"{len(may11) - len(warm_may11):,} cold trafo icin gecmis gercekten yok."
        )

    # --- id format kontrolu ---
    out.append(section("`id` format kontrolu"))
    expected = (
        test[C.ENTITY].astype(str) + "_" + test[C.DATE].dt.strftime("%Y-%m-%d")
    )
    n_bad = int((test["id"] != expected).sum())
    out.append(f"- `id` != `tanim_tarih` olan satir: **{n_bad:,}**")
    sub = pd.read_csv(C.SAMPLE_SUBMISSION_CSV, dtype={"id": "string"})
    out.append(f"- sample_submission satir sayisi: {len(sub):,}")
    out.append(
        f"- sample_submission id kumesi test id kumesine esit mi: "
        f"**{set(sub['id']) == set(test['id'])}**"
    )
    out.append(
        f"- sample_submission id sirasi test ile ayni mi: "
        f"**{sub['id'].tolist() == test['id'].tolist()}**"
    )

    path = C.REPORTS_DIR / "01_data_quality.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"yazildi: {path}")
    print("\n".join(out[:4]))


if __name__ == "__main__":
    main()
