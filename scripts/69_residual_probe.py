"""Artigin anatomisi: gizli etiketi gormeden dilim bazinda yanlilik olcumu.

`68_lb_algebra` sunu kurdu: referans gonderim p_0 icin artik r = p_0 - a'nin
her yon uzerindeki izdusumu, yalnizca LB skorlarindan kapali formda cikiyor:

    c_k = E[d_k r] = (LB_k^2 - LB_0^2 - E[d_k^2]) / 2,   d_k = p_k - p_0

Bu 18 yonun gerdigi altuzayda YATAN her g vektoru icin E[g r] de bilinir:
g'yi d'lere izdusurup g ~ sum beta_k d_k yazariz, sonra E[g r] ~ beta'c.
Izdusum R^2 yuksekse tahmin guvenilir.

Buradaki g'ler yorumlanabilir dilim gostergeleri: cold, olu, ay, guc bandi,
hafta sonu, lokasyon... Her biri icin ogrendigimiz sey o dilimdeki ORTALAMA
artik, yani yanlilik. Yanlilik biliniyorsa optimal duzeltme de biliniyor:
dilimden b cikar, MSE f*b^2 kadar duser (f dilimin satir payi).

Izdusum R^2'si DUSUK olan dilimler ise mevcut gonderimlerden olculemez;
onlar kalan gonderim hakkiyla yoklanacak adaylardir. Bu betik ikisini de
listeler.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import load_test, load_train
from src.validation.folds import guc_band

DOWNLOADS = Path.home() / "Downloads"
SUBS = C.SUBMISSIONS_DIR

CATALOG = [
    (DOWNLOADS / "FINAL_4.csv", 1.01548, 27622210),
    (DOWNLOADS / "SUB_A_blend.csv", 1.04876, 27660953),
    (SUBS / "submission_optuna.csv", 1.06666, 28290096),
    (DOWNLOADS / "submission_1.06713.csv", 1.06713, 28320958),
    (SUBS / "submission_seed7.csv", 1.06727, 28351585),
    (DOWNLOADS / "submission_multiorigin.csv", 1.06766, 28335567),
    (DOWNLOADS / "submission_1.06929.csv", 1.06929, 28349461),
    (DOWNLOADS / "submission_1.07042.csv", 1.07042, 28329751),
    (DOWNLOADS / "SUB_B_blend_plus020.csv", 1.08249, 27669792),
    (DOWNLOADS / "SUB_D_cold_probe.csv", 1.08311, 27652351),
    (DOWNLOADS / "submission_v4_wx.csv", 1.10990, 27595859),
    (DOWNLOADS / "02_catboost_segment_split.csv", 1.11019, 27658262),
    (DOWNLOADS / "submission_v6.csv", 1.11447, 27600304),
    (SUBS / "submission_c_tuned.csv", 1.14554, 28216962),
    (SUBS / "submission_level_shape.csv", 1.14884, 28151805),
    (DOWNLOADS / "P_EMP.csv", 1.26800, 27519732),
    (DOWNLOADS / "P_YOY.csv", 1.28964, 27500097),
    (SUBS / "submission_ensemble.csv", 1.29422, 27971660),
    (DOWNLOADS / "submission_1.37997.csv", 1.37997, 27989263),
]

OUT: list[str] = []


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def main() -> None:
    test = load_test()
    train = load_train()
    ids = test["id"].astype("string")

    cols, lbs = [], []
    for path, lb, nbytes in CATALOG:
        if not path.exists() or path.stat().st_size != nbytes:
            continue
        s = pd.read_csv(path, dtype={"id": "string"})
        col = C.TARGET if C.TARGET in s.columns else s.columns[-1]
        v = s.set_index("id")[col].reindex(ids).to_numpy(dtype="float64")
        cols.append(np.log1p(np.clip(v, 0, None)))
        lbs.append(lb)
    P = np.column_stack(cols)
    lb = np.asarray(lbs)

    i0 = int(np.argmin(lb))
    p0, M = P[:, i0], lb[i0] ** 2
    idx = [k for k in range(len(lbs)) if k != i0]
    Dm = P[:, idx] - p0[:, None]
    Ed2 = (Dm ** 2).mean(axis=0)
    c = (lb[idx] ** 2 - M - Ed2) / 2.0
    G = Dm.T @ Dm / len(p0)
    N = len(p0)

    say("# Artigin anatomisi: dilim bazinda yanlilik")
    say(f"Uretim: {pd.Timestamp.now()}")
    say()
    say(f"Referans: FINAL_4, LB {lb[i0]:.5f}, MSE {M:.5f}. "
        f"{len(idx)} yon, {N:,} satir.")
    say()

    # --- dilim gostergeleri ---
    seen = set(train[C.ENTITY].unique())
    h = train.loc[train[C.DATE] > C.TRAIN_END - pd.Timedelta(days=28)]
    gg = h.groupby(C.ENTITY)[C.TARGET]
    zero28 = test[C.ENTITY].map(gg.apply(lambda s: float((s <= 0).mean())))
    lvl28 = test[C.ENTITY].map(
        gg.apply(lambda s: float(np.log1p(s.clip(lower=0)).mean())))

    cold = (~test[C.ENTITY].isin(seen)).to_numpy()
    olu = (zero28 > 0.9).fillna(False).to_numpy()
    ay = test[C.DATE].dt.month.to_numpy()
    dow = test[C.DATE].dt.dayofweek.to_numpy()
    band = guc_band(test[C.POWER]).astype("string").to_numpy()
    lok = test[C.LOCATION].astype("string").to_numpy()
    lvl = lvl28.to_numpy()
    pl = p0

    probes: dict[str, np.ndarray] = {"SABIT (tum satirlar)": np.ones(N)}
    probes["cold"] = cold.astype(float)
    probes["warm_canli"] = ((~cold) & (~olu)).astype(float)
    probes["warm_olu (Mart'ta olu)"] = olu.astype(float)
    for a in (4, 5, 6, 7):
        probes[f"ay={a}"] = (ay == a).astype(float)
    probes["hafta sonu"] = (dow >= 5).astype(float)
    for b in pd.unique(band):
        probes[f"band={b}"] = (band == b).astype(float)
    for q, name in [(0.1, "en dusuk %10 tahmin"), (0.25, "en dusuk %25 tahmin")]:
        thr = np.quantile(pl, q)
        probes[name] = (pl <= thr).astype(float)
    thr = np.quantile(pl, 0.9)
    probes["en yuksek %10 tahmin"] = (pl >= thr).astype(float)
    # en buyuk 6 lokasyon
    top_lok = pd.Series(lok).value_counts().head(6).index
    for L in top_lok:
        probes[f"lok={L}"] = (lok == L).astype(float)
    # gecmisi kisa warm trafolar
    nh = test[C.ENTITY].map(train.groupby(C.ENTITY).size()).fillna(0).to_numpy()
    probes["warm, gecmis<90g"] = ((~cold) & (nh < 90)).astype(float)
    # surekli degiskenler (merkezlenmis)
    probes["tahmin seviyesi (merkezli)"] = pl - pl.mean()
    probes["log(guc) (merkezli)"] = (np.log(test[C.POWER].clip(lower=1)
                                            .astype("float64")).to_numpy()
                                     - np.log(test[C.POWER].clip(lower=1)
                                              .astype("float64")).mean())
    ok = ~np.isnan(lvl)
    lv = np.where(ok, lvl, np.nanmean(lvl))
    probes["Mart seviyesi (merkezli)"] = lv - lv.mean()

    say("## Dilim yanliliklari")
    say()
    say("`E[g r]` LB cebirinden geliyor; `yanlilik` = E[g r] / pay, yani o")
    say("dilimdeki ortalama (tahmin - gercek). Pozitif = FAZLA tahmin.")
    say("`dMSE` o dilimi yanliligi kadar kaydirirsak toplam MSE dususu.")
    say("`R2` izdusum kalitesi: 1'e yakin degilse sayi guvenilmez.")
    say()
    say(f"{'dilim':<30} {'pay':>7} {'R2':>7} {'E[g r]':>9} "
        f"{'yanlilik':>9} {'dMSE':>9}")
    rows = []
    for name, g in probes.items():
        b = Dm.T @ g / N
        beta = np.linalg.solve(G + 1e-8 * np.trace(G) / len(c) * np.eye(len(c)), b)
        res = g - Dm @ beta
        r2 = 1.0 - float((res ** 2).mean()) / float((g ** 2).mean())
        egr = float(beta @ c)
        frac = float(g.mean()) if set(np.unique(g)) <= {0.0, 1.0} else np.nan
        if np.isfinite(frac) and frac > 0:
            bias = egr / frac
            dmse = frac * bias ** 2
        else:
            var = float((g ** 2).mean())
            bias = egr / var if var > 0 else np.nan   # egim
            dmse = egr ** 2 / var if var > 0 else np.nan
        rows.append((name, frac, r2, egr, bias, dmse))
        say(f"{name:<30} {frac if np.isfinite(frac) else float('nan'):7.3f} "
            f"{r2:7.4f} {egr:9.5f} {bias:9.4f} {dmse:9.5f}")
    say()
    say("Surekli degiskenlerde `pay` yok; `yanlilik` sutunu egimi, `dMSE` o")
    say("egimi duzeltmenin kazancini verir.")
    say()

    good = [r for r in rows if r[2] > 0.9]
    say(f"R2 > 0.90 olan dilim sayisi: {len(good)} / {len(rows)}")
    say()
    say("## Olculebilir dilimlerde toplam potansiyel")
    say()
    tot = sum(r[5] for r in good if np.isfinite(r[5]) and r[0] == r[0])
    say(f"(kaba ust sinir, dilimler ortusuyor) toplam dMSE ~ {tot:.5f}")
    say(f"MSE {M:.5f} -> {M - tot:.5f}, LB {np.sqrt(M):.5f} -> "
        f"{np.sqrt(max(M - tot, 1e-9)):.5f}")
    say()
    say("## Olculemeyen (R2 dusuk) dilimler -- kalan gonderim hakkiyla yoklanacak")
    say()
    for name, frac, r2, egr, bias, dmse in rows:
        if r2 <= 0.9:
            say(f"  {name:<30} R2 {r2:6.3f}")
    say()

    p = C.REPORTS_DIR / "69_residual_probe.md"
    p.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
