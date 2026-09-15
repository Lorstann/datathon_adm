"""Cold seviye kalibrasyonu: dogal cold trafolar uzerinde tek parametreli tarama.

Gozlem: gonderim cold satirlarda ortalama log1p 6.967, warm-canli satirlarda
6.876 tahmin ediyor. Oysa dort dogrulama penceresinin hepsinde cold satirlarin
GERCEK ortalamasi warm'in 0.15-0.35 ALTINDA. Yani cold kolu yukari kaymis
olabilir.

Suphelenilen mekanizma: `entity_level` cold capasini
`median(hist_mean_z) + log(guc)` ile kuruyor. RMSLE log1p uzayinda L2, yani
optimal nokta tahmini MEDYAN degil ORTALAMA. Olu/yari-olu trafolarin z'si
cok negatif; medyan onlari disliyor, ortalama iceriyor.

Burada dogal cold trafolar (origin'e kadar hic satiri olmayan, pencerede
gorunen trafolar) uzerinde aday cold capalari RMSLE ile karsilastiriliyor.
Kullanilan her sey origin'de bilinir: guc, lokasyon, giris tarihi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import load_test, load_train
from src.models.zero_gate import dead_probability, shrink_level
from src.validation.folds import guc_band

OUT = []
WINDOWS = [
    ("A 2025-03-31", "2025-03-31", "2025-04-01", "2025-07-31"),
    ("B 2025-11-30", "2025-11-30", "2025-12-01", "2026-03-31"),
    ("C 2025-09-30", "2025-09-30", "2025-10-01", "2026-01-31"),
    ("D 2025-05-31", "2025-05-31", "2025-06-01", "2025-09-30"),
]


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def rmsle_log(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - a) ** 2)))


def entity_z(train: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Origin oncesi her trafonun ortalama log1p'i ve z'si."""
    h = train.loc[train[C.DATE] <= origin]
    g = h.groupby(C.ENTITY)
    out = pd.DataFrame({
        "lvl": g[C.TARGET].apply(lambda s: float(np.log1p(s.clip(lower=0)).mean())),
        "guc": g[C.POWER].first(),
        "lok": g[C.LOCATION].first(),
        "n": g.size(),
    })
    out["log_guc"] = np.log(out["guc"].clip(lower=1).astype("float64"))
    out["z"] = out["lvl"] - out["log_guc"]
    out["band"] = guc_band(out["guc"]).astype("string")
    return out


def shrunk_group_mean(ref: pd.DataFrame, keys: list[str], prior: float,
                      col: str = "z") -> tuple[pd.Series, float]:
    glob = float(ref[col].mean())
    g = ref.groupby(keys, observed=True)[col].agg(["mean", "size"])
    sm = (g["mean"] * g["size"] + glob * prior) / (g["size"] + prior)
    return sm, glob


def evaluate(train: pd.DataFrame, label: str, origin, start, end) -> dict:
    origin, start, end = (pd.Timestamp(x) for x in (origin, start, end))
    ref = entity_z(train, origin)

    seen = set(ref.index)
    w = train.loc[(train[C.DATE] >= start) & (train[C.DATE] <= end)].copy()
    cold = w.loc[~w[C.ENTITY].isin(seen)].copy()
    if cold.empty:
        return {}
    cold["a"] = np.log1p(cold[C.TARGET].clip(lower=0))
    cold["log_guc"] = np.log(cold[C.POWER].clip(lower=1).astype("float64"))
    cold["band"] = guc_band(cold[C.POWER]).astype("string")

    say(f"## {label}  pencere {start.date()}..{end.date()}")
    say(f"dogal cold trafo {cold[C.ENTITY].nunique():,}, satir {len(cold):,}")
    say(f"gercek ortalama log1p {cold['a'].mean():.4f}, "
        f"sifir orani {float((cold[C.TARGET] <= 0).mean()):.4f}")
    say(f"referans havuzu: {len(ref):,} trafo, "
        f"z medyan {ref['z'].median():.4f}, z ortalama {ref['z'].mean():.4f}")
    say()

    a = cold["a"].to_numpy()
    lg = cold["log_guc"].to_numpy()

    # olu-trafo kapisi, mevcut boru hattiyla ayni
    p_tab = dead_probability(train, origin)
    p_row = cold[C.LOCATION].map(p_tab["by_lokasyon"]).fillna(p_tab["global"]).to_numpy()

    cands: dict[str, np.ndarray] = {}
    cands["medyan_z (MEVCUT)"] = float(ref["z"].median()) + lg
    cands["ortalama_z"] = float(ref["z"].mean()) + lg
    for keys, name, prior in [
        (["band"], "band ortalama z", 20.0),
        (["lok"], "lokasyon ortalama z", 20.0),
        (["band", "lok"], "band x lokasyon ortalama z", 20.0),
    ]:
        sm, glob = shrunk_group_mean(ref, keys, prior)
        idx = (cold[["band", C.LOCATION]].rename(columns={C.LOCATION: "lok"})[keys]
               if len(keys) > 1 else
               cold["band"] if keys == ["band"] else cold[C.LOCATION])
        if len(keys) > 1:
            key_series = pd.MultiIndex.from_frame(idx)
            z = pd.Series(key_series.map(sm)).astype("float64").fillna(glob).to_numpy()
        else:
            z = pd.Series(np.asarray(idx)).map(sm).astype("float64").fillna(glob).to_numpy()
        cands[name] = z + lg

    say(f"{'cold capasi':<28} {'kapi yok':>10} {'kapi k=0.75':>12} "
        f"{'+ en iyi sabit':>15} {'sabit':>7}")
    res = {}
    for name, base in cands.items():
        r0 = rmsle_log(a, base)
        gated = np.log1p(shrink_level(np.expm1(np.clip(base, 0, None)),
                                      p_row, strength=0.75))
        r1 = rmsle_log(a, gated)
        # kapali formdan en iyi sabit kaydirma
        d = float(np.mean(a - gated))
        r2 = rmsle_log(a, gated + d)
        res[name] = (r0, r1, r2, d)
        say(f"{name:<28} {r0:10.4f} {r1:12.4f} {r2:15.4f} {d:7.3f}")
    say()
    return res


def test_side(train: pd.DataFrame, test: pd.DataFrame) -> None:
    say("# Test tarafi: gonderim cold satirlarda ne diyor")
    say()
    ref = entity_z(train, C.TRAIN_END)
    seen = set(ref.index)
    cold = test.loc[~test[C.ENTITY].isin(seen)].copy()
    say(f"cold trafo {cold[C.ENTITY].nunique():,}, satir {len(cold):,} "
        f"({len(cold) / len(test):.2%})")
    say(f"cold guc medyani {cold[C.POWER].median():.0f}; "
        f"train havuzu guc medyani {ref['guc'].median():.0f}")
    say(f"referans z: medyan {ref['z'].median():.4f}, ortalama {ref['z'].mean():.4f}, "
        f"fark {ref['z'].median() - ref['z'].mean():.4f}")
    say()

    p = C.SUBMISSIONS_DIR / "submission_BEST_1.06713.csv"
    if not p.exists():
        say("(gonderim yok)")
        return
    sub = pd.read_csv(p, dtype={"id": "string"})
    m = cold.merge(sub, on="id", how="left")
    m["logp"] = np.log1p(m[C.TARGET].clip(lower=0))
    say(f"gonderim cold ortalama log1p tahmini: {m['logp'].mean():.4f}")

    lg = np.log(m[C.POWER].clip(lower=1).astype("float64")).to_numpy()
    p_tab = dead_probability(train, C.TRAIN_END)
    p_row = m[C.LOCATION].map(p_tab["by_lokasyon"]).fillna(p_tab["global"]).to_numpy()
    for name, z in [("medyan_z", float(ref["z"].median())),
                    ("ortalama_z", float(ref["z"].mean()))]:
        base = z + lg
        gated = np.log1p(shrink_level(np.expm1(base), p_row, strength=0.75))
        say(f"  saf {name} capasi (kapili) ortalama: {gated.mean():.4f}")
    say()
    say("Gonderim tahmini capadan yuksekse sekil modeli cold satirlari")
    say("yukari itiyor demektir; asagidaki fold sonuclari bunun dogru olup")
    say("olmadigini soyler.")
    say()


def main() -> None:
    train, test = load_train(), load_test()
    say("# Cold seviye kalibrasyonu")
    say(f"Uretim: {pd.Timestamp.now()}")
    say()
    allres = {}
    for label, o, s, e in WINDOWS:
        r = evaluate(train, label, o, s, e)
        for k, v in r.items():
            allres.setdefault(k, []).append(v)

    say("# Ozet: dort pencere ortalamasi (kapi k=0.75 uygulanmis)")
    say()
    say(f"{'cold capasi':<28} {'RMSLE':>9} {'en iyi sabitle':>15} "
        f"{'ort sabit':>10}")
    base_key = "medyan_z (MEVCUT)"
    base_mean = float(np.mean([v[1] for v in allres[base_key]]))
    for name, vals in allres.items():
        r1 = float(np.mean([v[1] for v in vals]))
        r2 = float(np.mean([v[2] for v in vals]))
        d = float(np.mean([v[3] for v in vals]))
        say(f"{name:<28} {r1:9.4f} {r2:15.4f} {d:10.3f}")
    say()
    say(f"mevcut cold capasi referansi: {base_mean:.4f}")
    say()
    say("## Kazanc aritmetigi (cold satirlar test'in %22,16'si)")
    say("MSE_toplam degisimi = 0.2216 * (cold_RMSLE^2 farki)")
    say(f"{'aday':<28} {'dMSE':>9} {'LB tahmini':>11}")
    lb_mse = 1.06713 ** 2
    for name, vals in allres.items():
        r2 = float(np.mean([v[2] for v in vals]))
        d = 0.2216 * (base_mean ** 2 - r2 ** 2)
        say(f"{name:<28} {-d:9.4f} {np.sqrt(max(lb_mse - d, 1e-9)):11.4f}")
    say()
    say("Uyari: buradaki cold capalari SAF capa; gercek boru hattinda uzerine")
    say("sekil modeli biniyor. Yine de capalar arasi FARK sekil modelinden")
    say("bagimsiz olarak tasinir, cunku sekil ayni artik uzerinde egitiliyor.")
    say()

    test_side(train, test)
    p = C.REPORTS_DIR / "64_cold_level_calibration.md"
    p.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
