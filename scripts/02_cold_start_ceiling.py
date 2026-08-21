"""Cold-start tavan olcumu.

Soru: hic gecmisi olmayan bir trafo icin, statik metadata (`guc`, `lokasyon`)
ve takvim `log1p(tuketim)` varyansinin ne kadarini aciklayabilir?

Neden bu deney once yapiliyor: tum cold-start mimarisi (metadata-kNN, kume
atama, seviye+sekil ayristirmasi) statik niteliklerin bilgilendirici olmasina
bagli. Algorithms 2026, 19, 114 calismasinin negatif kontrolu bunu acikca
soyluyor. Tavan dusukse o mimariye yatirim yapmak bos emek.

Olcum tasarimi:
  - Varliklar fit / holdout olarak bolunur. Holdout varliklarin gecmisi HIC
    kullanilmaz, yani gercek cold-start durumu birebir taklit edilir.
  - Grup istatistikleri yalnizca fit varliklarindan ve yalnizca origin'e kadar
    olan veriden hesaplanir.
  - Degerlendirme, origin sonrasi 122 gunluk ufukta yapilir.

Metrik dogrudan RMSLE: z = log1p(tuketim) - log(guc) tahmin edildiginde
log1p(pred) = z_hat + log(guc) oldugu ve log(guc) tam bilindigi icin z
uzayindaki RMSE, RMSLE'ye birebir esittir.

Iki tani kolonu ayrica raporlanir:
  - kapsama: tahmincinin grup anahtarinin dogrulama satirinda eslesme orani.
    Dusuk kapsama, sonucun aslinda geri dusum degeriyle uretildigini gosterir.
  - hata ayristirmasi: kare hatanin ne kadari sifir tuketimli satirlardan
    geliyor. RMSLE'de sifirlar en yuksek kaldiracli satirlar.

Cikti: reports/02_cold_start_ceiling.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.data import build_panel, entity_static, load_test, load_train
from src.validation.folds import guc_band

HOLDOUT_FRAC = 0.25


def prepare(train: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    df = train[[C.ENTITY, C.DATE, C.POWER, C.TARGET]].copy()
    df["log1p_y"] = np.log1p(df[C.TARGET])
    df["z"] = df["log1p_y"] - np.log(df[C.POWER])
    df["ay"] = df[C.DATE].dt.month.astype("int8")
    df["haftagunu"] = df[C.DATE].dt.dayofweek.astype("int8")
    df["haftasonu"] = (df["haftagunu"] >= 5).astype("int8")
    df["sifir"] = df[C.TARGET].eq(0)
    df = df.join(static[["lokasyon", "il", "bolge"]], on=C.ENTITY)
    df["guc_band"] = guc_band(df[C.POWER]).astype(str)
    return df


def split_entities(static: pd.DataFrame, entities: pd.Index, seed: int) -> pd.Index:
    """Holdout varliklarini lokasyon x guc bandi icinde tabakali secer."""
    rng = np.random.default_rng(seed)
    s = static.loc[entities].copy()
    s["_band"] = guc_band(s["guc"]).astype(str)
    picked: list = []
    for _, idx in s.groupby(["lokasyon", "_band"], observed=True).groups.items():
        idx = list(idx)
        k = int(round(len(idx) * HOLDOUT_FRAC))
        if k == 0:
            continue
        sel = rng.choice(len(idx), size=k, replace=False)
        picked.extend(idx[int(i)] for i in sel)
    return pd.Index(picked, name=C.ENTITY)


class Evaluator:
    """Bir dogrulama penceresinde log1p uzayinda tahminci skorlar."""

    def __init__(self, ev: pd.DataFrame, fit: pd.DataFrame) -> None:
        self.ev = ev
        self.fit = fit
        self.y = ev["log1p_y"].to_numpy(dtype="float64")
        self.log_guc = np.log(ev[C.POWER].to_numpy(dtype="float64"))
        self.is_zero = ev["sifir"].to_numpy()
        self.rows: list[dict] = []

    def add(
        self,
        name: str,
        pred: np.ndarray,
        space: str,
        note: str = "",
        coverage: float = 1.0,
    ) -> None:
        p = pred + self.log_guc if space == "z" else pred
        p = np.clip(p, 0.0, None)
        sq = (p - self.y) ** 2
        mse = float(np.mean(sq))
        zero_share = float(np.sum(sq[self.is_zero]) / np.sum(sq)) if np.sum(sq) else 0.0
        self.rows.append(
            {
                "tahminci": name,
                "uzay": space,
                "rmsle": float(np.sqrt(mse)),
                "rmsle_sifirsiz": float(np.sqrt(np.mean(sq[~self.is_zero]))),
                "sifir_hata_payi": zero_share,
                "kapsama": coverage,
                "not": note,
            }
        )

    def add_group(
        self, name: str, keys: list[str], col: str, agg: str, space: str, note: str = ""
    ) -> None:
        stat = self.fit.groupby(keys, observed=True)[col].agg(agg)
        fallback = float(getattr(self.fit[col], agg)())
        mapped = pd.Series(
            self.ev.set_index(keys).index.map(stat), index=self.ev.index, dtype="float64"
        )
        coverage = float(mapped.notna().mean())
        self.add(
            name,
            mapped.fillna(fallback).to_numpy(),
            space,
            f"{agg}, n_grup={len(stat)}{'; ' + note if note else ''}",
            coverage,
        )

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def evaluate_window(
    df: pd.DataFrame,
    static: pd.DataFrame,
    origin: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    hist = df.loc[df[C.DATE] <= origin]
    valid = df.loc[(df[C.DATE] >= valid_start) & (df[C.DATE] <= valid_end)]

    eligible = pd.Index(
        sorted(set(hist[C.ENTITY].unique()) & set(valid[C.ENTITY].unique())),
        name=C.ENTITY,
    )
    holdout = set(split_entities(static, eligible, seed))

    fit = hist.loc[~hist[C.ENTITY].isin(holdout)]
    ev = valid.loc[valid[C.ENTITY].isin(holdout)]
    ev_warm = valid.loc[~valid[C.ENTITY].isin(holdout)]

    e = Evaluator(ev, fit)

    # --- Metadata bilmeyen tabanlar ---
    e.add("global ortalama", np.full(len(ev), fit["log1p_y"].mean()), "log1p")
    e.add("global medyan", np.full(len(ev), fit["log1p_y"].median()), "log1p")
    e.add("global ortalama", np.full(len(ev), fit["z"].mean()), "z", "guc ile olceklenmis")
    e.add("global medyan", np.full(len(ev), fit["z"].median()), "z", "guc ile olceklenmis")

    # --- Artan metadata, z uzayinda ---
    e.add_group("guc_band", ["guc_band"], "z", "mean", "z")
    e.add_group("lokasyon", ["lokasyon"], "z", "mean", "z")
    e.add_group("il", ["il"], "z", "mean", "z")
    e.add_group("lokasyon x guc_band", ["lokasyon", "guc_band"], "z", "mean", "z")
    e.add_group(
        "lokasyon x guc_band x haftasonu",
        ["lokasyon", "guc_band", "haftasonu"],
        "z",
        "mean",
        "z",
    )
    e.add_group(
        "lokasyon x guc_band x ay", ["lokasyon", "guc_band", "ay"], "z", "mean", "z"
    )
    e.add_group(
        "lokasyon x guc_band x haftasonu",
        ["lokasyon", "guc_band", "haftasonu"],
        "z",
        "median",
        "z",
    )

    # --- log1p parametrizasyonu, guc ofseti olmadan ---
    e.add_group(
        "lokasyon x guc_band x haftasonu",
        ["lokasyon", "guc_band", "haftasonu"],
        "log1p_y",
        "mean",
        "log1p",
        "guc ofseti YOK",
    )

    # --- Sifir kapisinin degeri: sifir durumu bilinirse ne olur ---
    # Kahin kapi: satirin sifir olup olmadigini bil, sifir olmayanlarda
    # metadata grup ortalamasini kullan. Bu, sifir kapisinin cold segmentteki
    # ust siniridir.
    keys = ["lokasyon", "guc_band", "haftasonu"]
    nz_fit = fit.loc[~fit["sifir"]]
    stat_nz = nz_fit.groupby(keys, observed=True)["z"].mean()
    fb_nz = float(nz_fit["z"].mean())
    mapped = pd.Series(
        ev.set_index(keys).index.map(stat_nz), index=ev.index, dtype="float64"
    ).fillna(fb_nz)
    gated = np.where(ev["sifir"].to_numpy(), -np.inf, mapped.to_numpy())
    # -inf yerine log1p(0)=0 tahmini: z uzayindan cikip dogrudan log1p veriyoruz
    gated_log1p = np.where(
        ev["sifir"].to_numpy(),
        0.0,
        mapped.to_numpy() + np.log(ev[C.POWER].to_numpy(dtype="float64")),
    )
    e.add(
        "KAHIN KAPI: sifir bilinir + metadata",
        gated_log1p,
        "log1p",
        "sifir kapisinin cold ust siniri",
    )
    del gated

    # Sifir olasiligini bilen ama kesin bilmeyen kapi: grup sifir orani ile
    # beklenen log1p (L2-optimal karisim tahmini).
    p_zero = fit.groupby(keys, observed=True)["sifir"].mean()
    p_map = pd.Series(
        ev.set_index(keys).index.map(p_zero), index=ev.index, dtype="float64"
    ).fillna(float(fit["sifir"].mean()))
    soft = (1 - p_map.to_numpy()) * (
        mapped.to_numpy() + np.log(ev[C.POWER].to_numpy(dtype="float64"))
    )
    e.add(
        "YUMUSAK KAPI: grup sifir olasiligi ile agirlikli",
        soft,
        "log1p",
        "kesin bilgi yok, L2-optimal karisim",
    )

    # --- Tavan: varligin kendi seviyesini bilen kahin ---
    e.add(
        "KAHIN SEVIYE: varligin ufuk-ici log1p ortalamasi",
        ev.groupby(C.ENTITY, observed=True)["log1p_y"].transform("mean").to_numpy(),
        "log1p",
        "ulasilamaz alt sinir",
    )
    e.add(
        "KAHIN SEVIYE: varlik x haftagunu",
        ev.groupby([C.ENTITY, "haftagunu"], observed=True)["log1p_y"]
        .transform("mean")
        .to_numpy(),
        "log1p",
        "ulasilamaz alt sinir",
    )

    # --- Warm referans ---
    warm_level = fit.groupby(C.ENTITY, observed=True)["log1p_y"].mean()
    warm_pred = (
        ev_warm[C.ENTITY].map(warm_level).fillna(float(fit["log1p_y"].mean())).to_numpy()
    )
    warm_y = ev_warm["log1p_y"].to_numpy(dtype="float64")
    warm_sq = (np.clip(warm_pred, 0, None) - warm_y) ** 2
    warm_zero = ev_warm["sifir"].to_numpy()

    info = {
        "n_holdout_varlik": float(len(holdout)),
        "n_fit_varlik": float(fit[C.ENTITY].nunique()),
        "n_cold_satir": float(len(ev)),
        "n_warm_satir": float(len(ev_warm)),
        "cold_sifir_orani": float(ev["sifir"].mean()),
        "warm_sifir_orani": float(ev_warm["sifir"].mean()),
        "warm_rmsle": float(np.sqrt(np.mean(warm_sq))),
        "warm_rmsle_sifirsiz": float(np.sqrt(np.mean(warm_sq[~warm_zero]))),
        "warm_sifir_hata_payi": float(np.sum(warm_sq[warm_zero]) / np.sum(warm_sq)),
        "log1p_std": float(ev["log1p_y"].std(ddof=0)),
        "fit_aylar": ",".join(str(m) for m in sorted(fit["ay"].unique())),
        "valid_aylar": ",".join(str(m) for m in sorted(ev["ay"].unique())),
    }
    return e.table(), info


def main() -> None:
    C.ensure_dirs()
    train = load_train()
    test = load_test()
    static = entity_static(build_panel(train, test))
    df = prepare(train, static)

    out: list[str] = ["# Cold-Start Tavan Olcumu", ""]
    out.append(f"Uretim tarihi: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    out.append("")
    out.append(
        f"Varliklarin %{100 * HOLDOUT_FRAC:.0f}'i lokasyon x guc bandi icinde tabakali "
        "secilip gecmisleri tamamen gizlendi, yani gercek cold-start durumu birebir "
        "taklit edildi. Butun sayilar **RMSLE**."
    )
    out.append("")
    out.append(
        "`kapsama` kolonu onemli: tahmincinin grup anahtari dogrulama satirlarinda "
        "ne oranda eslesiyor. Kapsama dusukse raporlanan skor aslinda geri dusum "
        "degerinin skorudur, o anahtarin skoru degil."
    )

    results: dict[str, tuple[pd.DataFrame, dict]] = {}
    for fold in C.FOLDS:
        tab, info = evaluate_window(
            df, static, fold.origin, fold.valid_start, fold.valid_end, C.SEED
        )
        results[fold.name] = (tab, info)

        out.append(
            f"\n## Fold {fold.name}: origin {fold.origin:%Y-%m-%d}, ufuk "
            f"{fold.valid_start:%Y-%m-%d} - {fold.valid_end:%Y-%m-%d}\n"
        )
        out.append(
            f"- Holdout (cold taklidi) varlik **{int(info['n_holdout_varlik']):,}**, "
            f"fit varlik {int(info['n_fit_varlik']):,}"
        )
        out.append(
            f"- Cold satir **{int(info['n_cold_satir']):,}**, "
            f"warm satir {int(info['n_warm_satir']):,}"
        )
        out.append(
            f"- Cold satirlarda sifir orani **{100 * info['cold_sifir_orani']:.2f}%**, "
            f"warm satirlarda {100 * info['warm_sifir_orani']:.2f}%"
        )
        out.append(
            f"- Fit'te bulunan aylar: `{info['fit_aylar']}` | "
            f"dogrulamada gereken aylar: `{info['valid_aylar']}`"
        )
        out.append(
            f"- **Warm referans** (gecmisten trafo log1p ortalamasi): RMSLE "
            f"**{info['warm_rmsle']:.4f}**, sifir satirlar cikarilinca "
            f"{info['warm_rmsle_sifirsiz']:.4f}; kare hatanin "
            f"%{100 * info['warm_sifir_hata_payi']:.1f}'i sifir satirlardan"
        )
        out.append("")
        out.append(
            "| Tahminci | Uzay | RMSLE | RMSLE (sifirsiz) | Sifir hata payi | Kapsama | Not |"
        )
        out.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, r in tab.iterrows():
            out.append(
                f"| {r['tahminci']} | {r['uzay']} | **{r['rmsle']:.4f}** | "
                f"{r['rmsle_sifirsiz']:.4f} | {100 * r['sifir_hata_payi']:.1f}% | "
                f"{100 * r['kapsama']:.1f}% | {r['not']} |"
            )

    # --- Karar bolumu ---
    out.append("\n## Sonuclar ve stratejik karar\n")

    def pick(tab: pd.DataFrame, name: str) -> pd.Series:
        return tab.loc[tab["tahminci"] == name].iloc[0]

    out.append("### 1. Sifir satirlar cold-start hatasinin ana kaynagi\n")
    out.append("| Fold | Cold sifir orani | Metadata tabani RMSLE | Sifir hata payi | Kahin kapi RMSLE | Kazanc |")
    out.append("| --- | --- | --- | --- | --- | --- |")
    for name, (tab, info) in results.items():
        base = tab.loc[(tab["tahminci"] == "global ortalama") & (tab["uzay"] == "z")].iloc[0]
        gate = pick(tab, "KAHIN KAPI: sifir bilinir + metadata")
        out.append(
            f"| {name} | {100 * info['cold_sifir_orani']:.2f}% | {base['rmsle']:.4f} | "
            f"{100 * base['sifir_hata_payi']:.1f}% | {gate['rmsle']:.4f} | "
            f"**{base['rmsle'] - gate['rmsle']:.4f}** |"
        )
    out.append("")
    out.append(
        "Cold satirlarin yalnizca %2-6'si sifir, ama kare hatanin %38-74'unu bunlar "
        "uretiyor. Satirin sifir olup olmadigini bilmek tek basina RMSLE'yi 0,43-0,89 "
        "dusuruyor. **Bu, projedeki en yuksek getirili tek bilgi parcasi.**"
    )

    out.append("\n### 2. `guc` disindaki statik metadata cold-start icin bilgilendirici degil\n")
    out.append("| Fold | log1p global | z global (guc ile) | + lokasyon | + lokasyon x guc_band |")
    out.append("| --- | --- | --- | --- | --- |")
    for name, (tab, _) in results.items():
        a = tab.loc[(tab["tahminci"] == "global ortalama") & (tab["uzay"] == "log1p")].iloc[0]
        b = tab.loc[(tab["tahminci"] == "global ortalama") & (tab["uzay"] == "z")].iloc[0]
        c = pick(tab, "lokasyon")
        d = pick(tab, "lokasyon x guc_band")
        out.append(
            f"| {name} | {a['rmsle']:.4f} | {b['rmsle']:.4f} | {c['rmsle']:.4f} | "
            f"{d['rmsle']:.4f} |"
        )
    out.append("")
    out.append(
        "`guc` ile olcekleme gercek ve tutarli bir kazanc (fold A'da 2,2203 -> 1,9780). "
        "Ama `lokasyon` eklemek skoru **kotulestiriyor**, `lokasyon x guc_band` daha da "
        "kotulestiriyor. Bu grup ortalamasi asiri uyumu: 285-304 grup, fit "
        "varliklarindan tahmin edilip hic gorulmemis varliklara tasindiginda gurultu "
        "tasiyor."
    )
    out.append("")
    out.append(
        "> **Cerceve dokumanindaki H2 hipotezi reddedildi.** `guc` bilgilendirici, "
        "`lokasyon` degil. Sonuc: metadata-kNN ve DTW-kume atama mimarisine yatirim "
        "yapmak gerekcesiz. Zaragoza calismasinin %95,6'lik kume atama dogrulugu "
        "bagli musteri tipi ve sayisi gibi niteliklere dayaniyordu; bizim elimizde "
        "yalnizca kurulu guc ve idari bolge var, o kadar bilgi tasimiyorlar. "
        "Kaynak, cold segmentte metadata retrieval'a degil sifir kapisina ve "
        "duzenlilestirilmis global modele ayrilacak."
    )

    out.append("\n### 3. Regresor sifir olmayan satirlarda kurulmali\n")
    out.append("| Fold | z global ORTALAMA (sifirsiz RMSLE) | z global MEDYAN (sifirsiz RMSLE) |")
    out.append("| --- | --- | --- |")
    for name, (tab, _) in results.items():
        m = tab.loc[(tab["tahminci"] == "global ortalama") & (tab["uzay"] == "z")].iloc[0]
        md = tab.loc[(tab["tahminci"] == "global medyan") & (tab["uzay"] == "z")].iloc[0]
        out.append(f"| {name} | {m['rmsle_sifirsiz']:.4f} | {md['rmsle_sifirsiz']:.4f} |")
    out.append("")
    out.append(
        "Sifir satirlar cikarildiginda medyan ortalamayi geciyor (fold A'da 1,0681'e "
        "karsi 1,2172). Nedeni mekanik: sifir satirlarin `z` degeri `-log(guc)` gibi "
        "cok negatif bir sayi, dolayisiyla fit setindeki sifirlar ortalamayi asagi "
        "cekiyor ve sifir olmayan satirlarda sistematik dusuk tahmin uretiyor. "
        "**Karar: regresor yalnizca sifir olmayan satirlarda egitilecek, sifir "
        "olasiligi ayri bir siniflandiriciyla modellenecek.** Bu, iki asamali "
        "mimarinin ampirik gerekcesi."
    )

    out.append("\n### 4. Fold A yilin ayini ogrenemez, bu yapisal bir kisit\n")
    out.append("| Fold | Fit'te bulunan aylar | Dogrulamada gereken aylar | `ay` anahtarinin kapsamasi |")
    out.append("| --- | --- | --- | --- |")
    for name, (tab, info) in results.items():
        cov = pick(tab, "lokasyon x guc_band x ay")["kapsama"]
        out.append(
            f"| {name} | {info['fit_aylar']} | {info['valid_aylar']} | "
            f"**{100 * cov:.1f}%** |"
        )
    out.append("")
    out.append(
        "Fold A'nin egitim penceresi 2025-03-31'de bitiyor ve veri 2025-01-01'de "
        "basliyor, yani Nisan-Temmuz aylarina ait **hicbir** gecmis gozlem yok: `ay` "
        "anahtarinin kapsamasi tam olarak %0. Bu duzeltilebilir bir hata degil, "
        "verinin yapisal kisiti. Panel yalnizca bir yaz iceriyor (Nis-Tem 2025) ve o "
        "yaz fold A'nin dogrulama penceresinin kendisi."
    )
    out.append("")
    out.append(
        "Sonuclar, uc ayri baslikta:"
    )
    out.append("")
    out.append(
        "1. **Fold A saf mevsim ekstrapolasyonunu test ediyor.** Model Ocak-Mart'tan "
        "Temmuz'a gitmek zorunda. Bunu yapabilmesinin tek yolu, ufuk boyunca zaten "
        "elimizde olan **hava degiskenleri**. Yani fold A hava ozelliklerinin degerini "
        "olcen fold; takvim-mevsim ozelliklerinin degerini olcemez."
    )
    out.append(
        "2. **Fold B yil-uzeri mevsimsel ozellikleri test edebilen tek fold** "
        "(kapsama %72,4): Ara 2025-Mar 2026 ufkunun Oca-Mar kismi icin Oca-Mar 2025 "
        "analogu var. `gecen yil ayni ay` tipi ozellikler burada dogrulanacak."
    )
    out.append(
        "3. **Gercek gonderim, hicbir fold'un dogrulayamadigi bir avantaja sahip:** "
        "origin 2026-03-31 oldugunda Nis-Tem 2025 tamamen train icinde. Yani mevsimsel "
        "naif ve gecen-yil ozellikleri gonderimde calisacak ama fold A'da olculemez. "
        "Bu asimetri, fold A skorlarinin gercek test skorundan **kotumser** olmasi "
        "beklendigi anlamina gelir ve fold'lar arasi karsilastirmada akilda tutulmali."
    )

    out.append("\n### 5. Warm satirlarda bile tarihsel seviye zayif bir tahminci\n")
    out.append("| Fold | Warm: gecmis ortalamasi | Cold: kahin ufuk-ici seviye |")
    out.append("| --- | --- | --- |")
    for name, (tab, info) in results.items():
        orc = pick(tab, "KAHIN SEVIYE: varligin ufuk-ici log1p ortalamasi")
        out.append(f"| {name} | {info['warm_rmsle']:.4f} | {orc['rmsle']:.4f} |")
    out.append("")
    out.append(
        "Gecmisi olan bir trafo icin tarihsel ortalamayi tasimak 1,13-1,44 RMSLE "
        "veriyor; oysa ayni trafonun ufuk-ici gercek ortalamasini bilmek 0,46-0,56 "
        "veriyor. Aradaki bosluk, seviyenin gecmisten ufka **kaydigini** gosteriyor: "
        "buyume, rejim degisimi ve mevsimsel seviye kaymasi. Yani warm segmentte de "
        "is bitmiyor; seviye tahminini duzeltmek (trend, buyume orani, mevsimsel "
        "olcekleme) gercek bir kazanc alani."
    )

    path = C.REPORTS_DIR / "02_cold_start_ceiling.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"yazildi: {path}")
    for name, (tab, info) in results.items():
        print(f"\n=== {name} === cold sifir orani {info['cold_sifir_orani']:.4f} "
              f"| warm ref {info['warm_rmsle']:.4f}")
        print(
            tab[
                ["tahminci", "uzay", "rmsle", "rmsle_sifirsiz", "sifir_hata_payi", "kapsama"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
