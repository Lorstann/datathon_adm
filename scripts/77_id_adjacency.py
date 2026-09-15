"""`tanim` kimliginin sayisal komsulugu cold seviyeyi tahmin ediyor mu?

Neden bu soru. LB cebiri gosterdi ki FINAL_4'un YANLILIGI her olculebilir
dilimde sifir: global, cold, warm-canli. Yani dogrusal duzeltme bitti. Kalan
hata VARYANS ve `76_zero_floor`'a gore o varyansin buyuk kismi cold
satirlarda: test'in %22'si, MSE'nin ~%75'i. Cold hatasi buyuk cunku yeni bir
trafo hakkinda yalnizca guc + lokasyon biliyoruz; `reports/43` bu ikisinin
tavanini R2 = 0.577 olarak olcmustu.

`reports/60` (H3) `tanim` kodunun ONEKini denedi (ilk 2-5 hane) ve
guc+lokasyon otesinde deger bulamadi. Ama onek kaba bir grup anahtari.
Trafo kimlikleri saha kurulum sirasina gore verilir; 12344 ile 12345 buyuk
ihtimalle AYNI fiderde, yan yana iki trafodur. Onek bunu yakalamaz (12399 ve
12400 farkli oneklerde), sayisal KOMSULUK yakalar.

Test: bir trafonun seviyesini, sayisal olarak en yakin K trafonun seviyesi
tahmin ediyor mu -- guc+lokasyon'un uzerine bir sey katarak? Cold'u taklit
etmek icin trafolarin %30'u disarida birakiliyor; komsuluk yalnizca kalan
%70'ten kuruluyor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import load_test, load_train, split_location
from src.validation.folds import guc_band

OUT: list[str] = []
PRIOR = 10.0


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def main() -> None:
    train, test = load_train(), load_test()

    say("# `tanim` sayisal komsulugu cold seviyeyi tasiyor mu")
    say(f"Uretim: {pd.Timestamp.now()}")
    say()

    g = train.groupby(C.ENTITY)
    ent = pd.DataFrame({
        "lvl": g[C.TARGET].apply(lambda s: float(np.log1p(s.clip(lower=0)).mean())),
        "guc": g[C.POWER].first(),
        "lok": g[C.LOCATION].first(),
        "n": g.size(),
    })
    ent = ent.loc[ent["n"] >= 30]
    ent["log_guc"] = np.log(ent["guc"].clip(lower=1).astype("float64"))
    ent["z"] = ent["lvl"] - ent["log_guc"]
    ent["band"] = guc_band(ent["guc"]).astype("string")
    loc = split_location(ent["lok"])
    ent["ilce"] = loc["ilce"].fillna(loc["il"]).astype("string")

    num = pd.to_numeric(ent.index.to_series(), errors="coerce")
    ent["idnum"] = num.to_numpy()
    ok = ent["idnum"].notna()
    say(f"train trafosu (>=30 gun): {len(ent):,}, sayisal kimlikli: "
        f"{int(ok.sum()):,}")
    ent = ent.loc[ok].copy()
    ent["idnum"] = ent["idnum"].astype("int64")
    ent = ent.sort_values("idnum")
    say(f"kimlik araligi {ent['idnum'].min():,} .. {ent['idnum'].max():,}")
    say()

    # kimlik farki dagilimi: kodlar ne kadar yogun
    d = np.diff(ent["idnum"].to_numpy())
    say("## Ardisik kimlikler arasi fark")
    say(f"medyan {np.median(d):.0f}, %25 {np.percentile(d,25):.0f}, "
        f"%75 {np.percentile(d,75):.0f}, fark==1 orani "
        f"{float((d == 1).mean()):.3f}")
    say()

    # --- ayni fiderde mi: kimlik farki kucukse lokasyon da ayni mi ---
    say("## Kimlik farki kucuk olan ciftler ayni lokasyonda mi")
    say(f"{'|fark|':<12} {'cift':>9} {'ayni ilce':>10} {'ayni band':>10} "
        f"{'|z farki| ort':>14}")
    idn = ent["idnum"].to_numpy()
    zz = ent["z"].to_numpy()
    ilce = ent["ilce"].to_numpy()
    band = ent["band"].to_numpy()
    for lo, hi in [(1, 1), (2, 3), (4, 10), (11, 50), (51, 500)]:
        rows = []
        for step in range(1, min(hi, 60) + 1):
            if step > len(idn) - 1:
                break
            df = idn[step:] - idn[:-step]
            sel = (df >= lo) & (df <= hi)
            if sel.any():
                rows.append((np.abs(zz[step:][sel] - zz[:-step][sel]),
                             ilce[step:][sel] == ilce[:-step][sel],
                             band[step:][sel] == band[:-step][sel]))
        if not rows:
            continue
        dz = np.concatenate([r[0] for r in rows])
        si = np.concatenate([r[1] for r in rows])
        sb = np.concatenate([r[2] for r in rows])
        say(f"{f'{lo}-{hi}':<12} {len(dz):9,} {si.mean():10.3f} "
            f"{sb.mean():10.3f} {dz.mean():14.4f}")
    # rastgele cift referansi
    rng = np.random.default_rng(C.SEED)
    a = rng.integers(0, len(zz), 200000)
    b = rng.integers(0, len(zz), 200000)
    say(f"{'rastgele':<12} {200000:9,} {float((ilce[a]==ilce[b]).mean()):10.3f} "
        f"{float((band[a]==band[b]).mean()):10.3f} "
        f"{float(np.abs(zz[a]-zz[b]).mean()):14.4f}")
    say()
    say("Kimlik farki kucuk ciftlerde |z farki| rastgele ciftlerden belirgin")
    say("kucukse, kimlik komsulugu gercek bir sinyaldir.")
    say()

    # --- cold benzeri bolme: %30 disarida ---
    say("## Cold taklidi: trafolarin %30'u disarida, OOS R2")
    say()
    hold = rng.random(len(ent)) < 0.30
    tr, te = ent.loc[~hold], ent.loc[hold]
    say(f"referans havuzu {len(tr):,}, degerlendirme {len(te):,}")
    ybar = float(tr["z"].mean())
    tot = float(((te["z"] - te["z"].mean()) ** 2).mean())

    def oos(pred: np.ndarray) -> float:
        return 1.0 - float(((te["z"].to_numpy() - pred) ** 2).mean()) / tot

    def grp_pred(keys: list[str]) -> np.ndarray:
        a_ = tr.groupby(keys, observed=True)["z"].agg(["mean", "size"])
        sm = (a_["mean"] * a_["size"] + ybar * PRIOR) / (a_["size"] + PRIOR)
        if len(keys) == 1:
            return te[keys[0]].map(sm).astype("float64").fillna(ybar).to_numpy()
        idx = pd.MultiIndex.from_frame(te[keys])
        return pd.Series(idx.map(sm)).astype("float64").fillna(ybar).to_numpy()

    base_preds = {
        "global": np.full(len(te), ybar),
        "band": grp_pred(["band"]),
        "ilce": grp_pred(["ilce"]),
        "band+ilce": grp_pred(["band", "ilce"]),
    }

    # kimlik komsulugu: referans havuzundaki en yakin K kimlik
    tr_id = tr["idnum"].to_numpy()
    tr_z = tr["z"].to_numpy()
    order = np.argsort(tr_id)
    tr_id, tr_z = tr_id[order], tr_z[order]
    te_id = te["idnum"].to_numpy()

    def knn_id(K: int, max_gap: int | None = None) -> np.ndarray:
        pos = np.searchsorted(tr_id, te_id)
        out = np.full(len(te_id), np.nan)
        for i, p in enumerate(pos):
            lo, hi = max(0, p - K), min(len(tr_id), p + K)
            cand = np.arange(lo, hi)
            if len(cand) == 0:
                continue
            dist = np.abs(tr_id[cand] - te_id[i])
            if max_gap is not None:
                cand = cand[dist <= max_gap]
                dist = dist[dist <= max_gap]
            if len(cand) == 0:
                continue
            sel = np.argsort(dist)[:K]
            out[i] = tr_z[cand[sel]].mean()
        return out

    say(f"{'yontem':<34} {'kapsam':>8} {'OOS R2':>9}")
    for nm, p in base_preds.items():
        say(f"{nm:<34} {1.0:8.3f} {oos(p):9.4f}")

    best_combo = None
    for K in (2, 4, 8, 16):
        for gap in (None, 50, 500):
            v = knn_id(K, gap)
            cov = float(np.isfinite(v).mean())
            filled = np.where(np.isfinite(v), v, base_preds["band+ilce"])
            lbl = f"kimlik-kNN K={K} gap={gap}"
            say(f"{lbl:<34} {cov:8.3f} {oos(filled):9.4f}")
            # band+ilce ile birlestir
            comb = 0.5 * filled + 0.5 * base_preds["band+ilce"]
            lbl2 = f"  + band+ilce (yari yariya)"
            s = oos(comb)
            say(f"{lbl2:<34} {cov:8.3f} {s:9.4f}")
            if best_combo is None or s > best_combo[1]:
                best_combo = (lbl, s)
    say()
    say(f"En iyi kimlik tabanli: {best_combo[0]} -> OOS R2 {best_combo[1]:.4f}")
    say(f"band+ilce referansi:                  OOS R2 "
        f"{oos(base_preds['band+ilce']):.4f}")
    say()

    # --- test'teki cold trafolarin kimlik komsulugu var mi ---
    seen = set(train[C.ENTITY].unique())
    cold_ent = sorted(set(test[C.ENTITY]) - seen)
    cn = pd.to_numeric(pd.Series(cold_ent), errors="coerce")
    say("## Test'teki cold trafolarin kimlik komsulugu")
    say(f"cold trafo {len(cold_ent):,}, sayisal kimlikli "
        f"{int(cn.notna().sum()):,}")
    cv = cn.dropna().astype("int64").to_numpy()
    allid = np.sort(ent["idnum"].to_numpy())
    pos = np.searchsorted(allid, cv)
    nearest = []
    for i, p in enumerate(pos):
        cands = []
        if p > 0:
            cands.append(abs(allid[p - 1] - cv[i]))
        if p < len(allid):
            cands.append(abs(allid[p] - cv[i]))
        nearest.append(min(cands) if cands else np.nan)
    nearest = np.asarray(nearest, dtype="float64")
    say(f"en yakin train kimligine uzaklik: medyan {np.nanmedian(nearest):.0f}, "
        f"%25 {np.nanpercentile(nearest,25):.0f}, "
        f"%75 {np.nanpercentile(nearest,75):.0f}")
    for thr in (1, 2, 5, 10, 50):
        say(f"  uzaklik <= {thr:<4}: {float(np.nanmean(nearest <= thr)):.3f}")
    say()

    p = C.REPORTS_DIR / "77_id_adjacency.md"
    p.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
