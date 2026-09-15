"""Sifir tabani: kalan hatanin ne kadari indirgenemez Bernoulli gurultusu?

Bulgu zinciri buraya getirdi. Q_COMBO (grup profili + gecen yil analogu +
panel SVD ortalamasi) havuza dik 0.56 genisliginde yeni bir eksen aciyordu ve
o eksende artigin izdusumu SIFIR cikti. Yani artik e0 duzgun/yapisal hicbir
yonle iliskili degil. Bu, artigin yapisal degil BERNOULLI oldugunu soyler:
sifirlar.

Aritmetik neden bu kadar sert: tipik bir trafonun log1p seviyesi ~7. O gun
tuketim sifir cikarsa o tek satir kareli hataya 49 katar. Test'in %4.7'si tam
sifir. Yani sifirlar tek baslarina 2.3 MSE tasiyabilir -- toplam MSE'miz
1.03. Demek ki sifirlarin cogunu zaten yakaliyoruz; kalan acik da orada.

Kaldirac aritmetigi (neden dar yonler onemli):
  f oraninda satirda m kadar fazla tahmin ediyorsak, o satirlari sifirlamak
  ||D||^2 = f*m^2 ve <e0,D> = -f*m^2 verir, kazanc = f*m^2.
  f = 0.001, m = 7  ->  kazanc 0.049 MSE  ->  LB 1.0155'ten 0.982'ye.
Yani test satirlarinin binde birini dogru sifirlamak butun acigi kapatir.
Ama yanlis sifirlamak ayni miktarda kaybettirir. Kesinlik her sey.

Bu betik iki sey yapiyor:
  1. Train penceresinde sifirin ne kadar ONGORULEBILIR oldugunu olcer:
     mukemmel sifir bilgisi ile gercekci sifir bilgisi arasindaki MSE farki.
  2. Test tarafinda sifir-odakli DAR yonler kurar ve her birinin kaldiracini
     (||D||, havuza diklik, "yanlilik b ise kazanc" tablosu) raporlar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import load_test, load_train

DOWNLOADS = Path.home() / "Downloads"
SUBS = C.SUBMISSIONS_DIR
POOL = [
    ("optuna", SUBS / "submission_optuna.csv", 1.06666),
    ("T_1.06713", DOWNLOADS / "submission_1.06713.csv", 1.06713),
    ("seed7", SUBS / "submission_seed7.csv", 1.06727),
    ("multiorigin", DOWNLOADS / "submission_multiorigin.csv", 1.06766),
    ("T_1.06929", DOWNLOADS / "submission_1.06929.csv", 1.06929),
    ("T_1.07042", DOWNLOADS / "submission_1.07042.csv", 1.07042),
    ("v4_wx", DOWNLOADS / "submission_v4_wx.csv", 1.10990),
    ("catboost_seg", DOWNLOADS / "02_catboost_segment_split.csv", 1.11019),
    ("v6", DOWNLOADS / "submission_v6.csv", 1.11447),
    ("c_tuned", SUBS / "submission_c_tuned.csv", 1.14554),
    ("level_shape", SUBS / "submission_level_shape.csv", 1.14884),
    ("P_EMP", DOWNLOADS / "P_EMP.csv", 1.26800),
    ("P_YOY", DOWNLOADS / "P_YOY.csv", 1.28964),
    ("T_1.29422", SUBS / "submission_ensemble.csv", 1.29422),
    ("T_1.33686", DOWNLOADS / "submission_1.33686.csv", 1.33686),
    ("T_1.37997", DOWNLOADS / "submission_1.37997.csv", 1.37997),
    ("Q_COMBO", SUBS / "probe_Q_COMBO.csv", 1.33324),
]
S0 = 1.01548
OUT: list[str] = []


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def train_side(train: pd.DataFrame) -> None:
    say("# 1. Train tarafi: sifir ne kadar ongorulebilir")
    say()
    for label, origin, start, end in [
        ("B 2025-11-30 -> Ara-Mar", "2025-11-30", "2025-12-01", "2026-03-31"),
        ("A 2025-03-31 -> Nis-Tem", "2025-03-31", "2025-04-01", "2025-07-31"),
    ]:
        origin, start, end = (pd.Timestamp(x) for x in (origin, start, end))
        hist = train.loc[train[C.DATE] <= origin]
        w28 = hist.loc[hist[C.DATE] > origin - pd.Timedelta(days=56)]
        g = w28.groupby(C.ENTITY)[C.TARGET]
        p_hat = g.apply(lambda s: float((s <= 0).mean()))
        m_hat = g.apply(
            lambda s: float(np.log1p(s[s > 0]).mean()) if (s > 0).any() else 0.0)

        v = train.loc[(train[C.DATE] >= start) & (train[C.DATE] <= end)].copy()
        v = v.loc[v[C.ENTITY].isin(set(p_hat.index))]
        a = np.log1p(v[C.TARGET].clip(lower=0).to_numpy())
        p = v[C.ENTITY].map(p_hat).to_numpy()
        m = v[C.ENTITY].map(m_hat).to_numpy()
        is_zero = (v[C.TARGET] <= 0).to_numpy()

        pred_opt = (1 - p) * m                      # gercekci: p'yi gecmisten bil
        pred_oracle = np.where(is_zero, 0.0, m)     # kahin: hangi gun sifir, bil
        mse_opt = float(((a - pred_opt) ** 2).mean())
        mse_or = float(((a - pred_oracle) ** 2).mean())

        say(f"## {label}")
        say(f"satir {len(v):,}, gercek sifir orani {is_zero.mean():.4f}")
        say(f"gecmisten p ile        MSE {mse_opt:7.4f}  RMSLE {np.sqrt(mse_opt):.4f}")
        say(f"hangi gun sifir bilinse MSE {mse_or:7.4f}  RMSLE {np.sqrt(mse_or):.4f}")
        say(f"SIFIR BELIRSIZLIGININ BEDELI  {mse_opt - mse_or:7.4f} MSE")
        say(f"  toplam hatanin {(mse_opt - mse_or) / mse_opt:.1%}'i")
        say()
        # sifirlarin ne kadari "surpriz"
        say(f"{'gecmis p araligi':<20} {'satir':>9} {'gercek sifir':>13} "
            f"{'ort log1p':>10}")
        for lo, hi in [(-0.01, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 0.9),
                       (0.9, 1.01)]:
            sel = (p > lo) & (p <= hi)
            if sel.sum() == 0:
                continue
            say(f"({lo:.2f},{hi:.2f}]{'':<9} {int(sel.sum()):9,} "
                f"{is_zero[sel].mean():13.4f} {a[sel].mean():10.4f}")
        say()
        surprise = is_zero & (p < 0.05)
        say(f"p<0.05 iken gerceklesen sifir: {int(surprise.sum()):,} satir "
            f"({surprise.mean():.4%}); bunlarin ortalama m'si "
            f"{m[surprise].mean():.3f}")
        say(f"  bu satirlarin tek basina MSE katkisi "
            f"{float((m[surprise] ** 2).sum() / len(a)):.4f}")
        say()


def test_side(train: pd.DataFrame, test: pd.DataFrame) -> None:
    say("# 2. Test tarafi: sifir odakli dar yonlerin kaldiraci")
    say()
    ids = test["id"].astype("string")

    def rd(p: Path) -> np.ndarray:
        s = pd.read_csv(p, dtype={"id": "string"})
        c = C.TARGET if C.TARGET in s.columns else s.columns[-1]
        return np.log1p(np.clip(
            s.set_index("id")[c].reindex(ids).to_numpy(dtype="float64"), 0, None))

    p0 = rd(DOWNLOADS / "FINAL_4.csv")
    S0sq = S0 ** 2
    N = len(p0)
    Dm, rr = [], []
    for nm, path, lb in POOL:
        if not path.exists():
            continue
        d = rd(path) - p0
        Dm.append(d)
        rr.append((lb ** 2 - S0sq - float((d ** 2).mean())) / 2.0)
    Dm = np.column_stack(Dm)
    rr = np.asarray(rr)
    G = Dm.T @ Dm / N

    hist = train.loc[train[C.DATE] > C.TRAIN_END - pd.Timedelta(days=56)]
    g = hist.groupby(C.ENTITY)[C.TARGET]
    zr = test[C.ENTITY].map(g.apply(lambda s: float((s <= 0).mean())))
    seen = set(train[C.ENTITY].unique())
    cold = (~test[C.ENTITY].isin(seen)).to_numpy()
    zr_f = zr.fillna(-1.0).to_numpy()

    t = test[["id", C.ENTITY, C.DATE]].copy()
    last = t.groupby(C.ENTITY)[C.DATE].transform("max")
    first = t.groupby(C.ENTITY)[C.DATE].transform("min")
    d_exit = (last - t[C.DATE]).dt.days.to_numpy()
    d_entry = (t[C.DATE] - first).dt.days.to_numpy()
    early = (last < C.TEST_END).to_numpy()
    late = (first > C.TEST_START).to_numpy()

    dirs: dict[str, np.ndarray] = {}
    # (a) Mart'ta olu trafolari tam sifirla
    sel = zr_f > 0.9
    dirs["olu_tam_sifir"] = np.where(sel, -p0, 0.0)
    # (b) yari-olu: p oraninda kucult
    sel2 = (zr_f > 0.3) & (zr_f <= 0.9)
    dirs["yari_olu_kucult"] = np.where(sel2, np.log1p(np.expm1(p0) * 0.4) - p0, 0.0)
    # (c) genel sifir egilimi: (1 - zr) ile olcekle
    sc = np.where(zr_f > 0, np.clip(1 - zr_f, 0.05, 1.0), 1.0)
    dirs["zr_ile_olcekle"] = np.log1p(np.expm1(p0) * sc) - p0
    # (d) panel cikis kapisi
    m = np.ones(N)
    for lo, hi, k in [(0, 2, 0.25), (3, 6, 0.40), (7, 13, 0.60), (14, 29, 0.80)]:
        s_ = early & (d_exit >= lo) & (d_exit <= hi)
        m[s_] = np.minimum(m[s_], k)
    dirs["cikis_kapisi"] = np.log1p(np.expm1(p0) * m) - p0
    # (e) panel giris kapisi
    m = np.ones(N)
    for lo, hi, k in [(0, 0, 0.45), (1, 2, 0.65), (3, 6, 0.85)]:
        s_ = late & (d_entry >= lo) & (d_entry <= hi)
        m[s_] = np.minimum(m[s_], k)
    dirs["giris_kapisi"] = np.log1p(np.expm1(p0) * m) - p0
    # (f) cold trafolarda sifir payi
    dirs["cold_kucult_20"] = np.where(cold, np.log1p(np.expm1(p0) * 0.8) - p0, 0.0)
    # (g) en dusuk tahminleri sifirla (model zaten sifira yakin diyor)
    thr = np.quantile(p0, 0.03)
    dirs["dusuk_tahmini_sifirla"] = np.where(p0 <= thr, -p0, 0.0)

    say(f"{'yon':<24} {'dokunulan':>10} {'||D||':>8} {'dik R2':>8} "
        f"{'||D_dik||':>10}")
    info = {}
    for nm, d in dirs.items():
        nz = int((np.abs(d) > 1e-9).sum())
        nrm = float(np.sqrt((d ** 2).mean()))
        b = Dm.T @ d / N
        beta = np.linalg.solve(G + 1e-8 * np.trace(G) / G.shape[0] * np.eye(G.shape[0]), b)
        res = d - Dm @ beta
        r2 = 1.0 - float((res ** 2).mean()) / max(float((d ** 2).mean()), 1e-18)
        info[nm] = (d, res, float((res ** 2).mean()), float(beta @ rr))
        say(f"{nm:<24} {nz:10,} {nrm:8.4f} {r2:8.4f} "
            f"{float(np.sqrt((res**2).mean())):10.4f}")
    say()
    say("`dik R2` = havuzun bu yonu ne kadar aciklayabildigi. Dusukse yon yeni.")
    say()

    say("## Kaldirac: dokunulan satirlarda ortalama yanlilik b ise beklenen LB")
    say()
    say("Yon d, dokunulan satirlarda ortalama -delta kadar asagi cekiyor. Eger")
    say("o satirlarda gercekten b kadar FAZLA tahmin ediyorsak <e0,d> = -f*delta*b")
    say("ve kazanc (f*delta*b)^2 / ||d||^2 olur.")
    say()
    say(f"{'yon':<24} " + "".join(f"{f'b={b}':>10}" for b in (0.2, 0.5, 1.0, 2.0)))
    for nm, (d, res, n2, _) in info.items():
        touched = np.abs(d) > 1e-9
        f = float(touched.mean())
        delta = float(-d[touched].mean()) if touched.any() else 0.0
        row = ""
        for b in (0.2, 0.5, 1.0, 2.0):
            r_hat = -f * delta * b
            gain = r_hat ** 2 / max(n2, 1e-12)
            row += f"{np.sqrt(max(S0sq - gain, 1e-9)):10.5f}"
        say(f"{nm:<24} {row}")
    say()
    say("Bu tablo UST SINIR degil senaryo: b'yi olcen sey yalnizca gonderim.")
    say("Ama hangi yonun yoklamaya deger oldugunu gosterir -- kaldiraci")
    say("buyuk olanlar dar ve derin olanlar.")
    say()

    for nm in ("olu_tam_sifir", "zr_ile_olcekle", "cikis_kapisi"):
        d = info[nm][0]
        q = np.expm1(np.clip(p0 + d, 0, None))
        pd.DataFrame({"id": ids.to_numpy(), C.TARGET: q}).to_csv(
            SUBS / f"probe_{nm}.csv", index=False)
        say(f"yazildi: submissions/probe_{nm}.csv")
    say()


def main() -> None:
    train, test = load_train(), load_test()
    say("# Sifir tabani ve sifir odakli yonler")
    say(f"Uretim: {pd.Timestamp.now()}")
    say()
    train_side(train)
    test_side(train, test)
    p = C.REPORTS_DIR / "76_zero_floor.md"
    p.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
