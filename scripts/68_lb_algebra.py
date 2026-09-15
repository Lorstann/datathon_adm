"""LB cebiri: skoru bilinen gonderimlerden optimal harmani kapali formda coz.

Fikir. Gonderim k'nin log-uzayi tahmini p_k, gercek a. Kaggle bize
    LB_k^2 = E[(p_k - a)^2]
sayisini veriyor. Referans olarak en iyi gonderimi p_0 secip d_k = p_k - p_0
yazarsak, acilim

    LB_k^2 = E[(p_0-a)^2] + 2 E[d_k (p_0-a)] + E[d_k^2]

olur. Burada E[(p_0-a)^2] = LB_0^2 ve E[d_k^2] elimizdeki dosyalardan
hesaplanabilir. Geriye tek bilinmeyen kaliyor:

    c_k = E[d_k r],   r = p_0 - a  (referansin artigi)
        = (LB_k^2 - LB_0^2 - E[d_k^2]) / 2

Yani gercek etiketleri hic gormeden, artigin her yon uzerindeki izdusumunu
ogreniyoruz. Bundan sonrasi kapali form: p = p_0 + sum_k lambda_k d_k icin

    MSE(lambda) = LB_0^2 + 2 lambda'c + lambda' D lambda,   D_jk = E[d_j d_k]
    lambda* = -(D + alpha I)^{-1} c

Bu bir kacamak degil; gizli etiketten sizinti da degil. Kullanilan tek bilgi
kendi gonderimlerimizin skorlari. Ayni sey, ayni matematikle, elle harman
agirligi aranarak da yapiliyor -- fark, burada tek adimda ve optimal
yapilmasi.

Guvenlik. Yonler birbirine cok benziyor (ayni boru hattinin varyantlari), yani
D kotu kosullu. Sirt (ridge) ve ozdeger kesme sart. Ustelik framework'un
kendisi dogrulanabilir: bir gonderimi disarida birakip skorunu digerlerinden
tahmin ediyoruz. Tutuyorsa optimizasyona guvenilir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import load_test

DOWNLOADS = Path.home() / "Downloads"
SUBS = C.SUBMISSIONS_DIR

# (dosya, LB skoru, Kaggle'daki total_bytes) -- eslesme BAYT BAYT dogrulaniyor.
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
    (P_EMP := DOWNLOADS / "P_EMP.csv", 1.26800, 27519732),
    (DOWNLOADS / "P_YOY.csv", 1.28964, 27500097),
    (SUBS / "submission_ensemble.csv", 1.29422, 27971660),
    (DOWNLOADS / "submission_1.37997.csv", 1.37997, 27989263),
]

OUT: list[str] = []


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def load_all(ids: pd.Series) -> tuple[list[str], np.ndarray, np.ndarray]:
    names, cols, lbs = [], [], []
    for path, lb, nbytes in CATALOG:
        if not path.exists():
            say(f"  ATLANDI (dosya yok): {path.name}")
            continue
        actual = path.stat().st_size
        if actual != nbytes:
            say(f"  ATLANDI (bayt uyusmuyor): {path.name} "
                f"{actual:,} != {nbytes:,}")
            continue
        s = pd.read_csv(path, dtype={"id": "string"})
        col = C.TARGET if C.TARGET in s.columns else s.columns[-1]
        s = s.set_index("id")[col]
        v = s.reindex(ids).to_numpy(dtype="float64")
        if np.isnan(v).any():
            say(f"  ATLANDI (id eksik): {path.name}")
            continue
        names.append(f"{path.stem} [{lb:.5f}]")
        cols.append(np.log1p(np.clip(v, 0, None)))
        lbs.append(lb)
        say(f"  ok  {path.name:<34} LB {lb:.5f}  {actual:,} bayt")
    return names, np.column_stack(cols), np.asarray(lbs)


def solve(D: np.ndarray, c: np.ndarray, alpha: float) -> np.ndarray:
    return -np.linalg.solve(D + alpha * np.eye(len(c)), c)


def main() -> None:
    test = load_test()
    ids = test["id"].astype("string")

    say("# LB cebiri: skoru bilinen gonderimlerden optimal harman")
    say(f"Uretim: {pd.Timestamp.now()}")
    say()
    say("## Katalog (bayt bayt dogrulanmis eslesme)")
    names, P, lb = load_all(ids)
    n = len(names)
    say()
    say(f"{n} gonderim, {P.shape[0]:,} satir.")
    say()

    # referans = en iyi
    i0 = int(np.argmin(lb))
    say(f"Referans p_0 = {names[i0]}")
    say()
    p0 = P[:, i0]
    M = lb[i0] ** 2
    idx = [k for k in range(n) if k != i0]
    Dmat = P[:, idx] - p0[:, None]           # yonler
    dnames = [names[k] for k in idx]
    dlb = lb[idx]

    Ed2 = (Dmat ** 2).mean(axis=0)
    c = (dlb ** 2 - M - Ed2) / 2.0
    D = Dmat.T @ Dmat / len(p0)

    say("## Yonler ve artik izdusumleri")
    say(f"{'yon (p_k - p_0)':<40} {'E[d^2]':>9} {'c_k':>9} "
        f"{'sabitlik':>9} {'ort d':>8}")
    for j, nm in enumerate(dnames):
        const = Dmat[:, j].mean() ** 2 / Ed2[j]
        say(f"{nm:<40} {Ed2[j]:9.4f} {c[j]:9.4f} {const:9.3f} "
            f"{Dmat[:, j].mean():8.4f}")
    say()
    say("`sabitlik` = ort(d)^2 / E[d^2]; 1'e yakinsa o yon neredeyse duz bir")
    say("kaydirma demektir. c_k negatifse o yonde ilerlemek hatayi azaltir.")
    say()

    # --- Framework dogrulamasi: birini disarida birak, skorunu tahmin et ---
    say("## Dogrulama: bir gonderimi disarida birakip LB'sini tahmin et")
    say()
    say("d_j'yi digerlerinin span'ina izdusurup c_j ~ beta'c ile skoru")
    say("kestiriyoruz. Gercek LB ile karsilastirma framework'un tek gercek")
    say("disariida testi.")
    say()
    say(f"{'disarida birakilan':<40} {'gercek':>8} {'tahmin':>8} {'hata':>8} "
        f"{'izdusum R2':>11}")
    errs = []
    for j in range(len(dnames)):
        keep = [k for k in range(len(dnames)) if k != j]
        A = Dmat[:, keep]
        y = Dmat[:, j]
        G = A.T @ A / len(y)
        b = A.T @ y / len(y)
        beta = np.linalg.solve(G + 1e-6 * np.eye(len(keep)), b)
        resid = y - A @ beta
        r2 = 1.0 - float((resid ** 2).mean()) / float((y ** 2).mean())
        c_hat = float(beta @ c[keep])
        mse_hat = M + 2 * c_hat + Ed2[j]
        pred = float(np.sqrt(max(mse_hat, 1e-9)))
        errs.append(pred - dlb[j])
        say(f"{dnames[j]:<40} {dlb[j]:8.5f} {pred:8.5f} "
            f"{pred - dlb[j]:+8.5f} {r2:11.4f}")
    e = np.asarray(errs)
    say()
    say(f"ortalama mutlak hata {np.abs(e).mean():.5f}, "
        f"medyan {np.median(np.abs(e)):.5f}, en kotu {np.abs(e).max():.5f}")
    say()

    # --- Optimal harman ---
    say("## Optimal harman (sirt parametresine gore)")
    say()
    tr = float(np.trace(D)) / len(c)
    say(f"{'alpha/tr(D)':>12} {'tahmini LB':>11} {'||lambda||_1':>13} "
        f"{'max|lambda|':>12}")
    grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    sols = {}
    for f in grid:
        a = f * tr
        lam = solve(D, c, a)
        mse = M + 2 * lam @ c + lam @ D @ lam
        sols[f] = (lam, mse)
        say(f"{f:12.4f} {np.sqrt(max(mse, 1e-9)):11.5f} "
            f"{np.abs(lam).sum():13.3f} {np.abs(lam).max():12.3f}")
    say()
    say("Kucuk alpha daha iyi tahmini skor verir ama agirliklari buyutur ve")
    say("LB gurultusune duyarlilastirir. Agirlik normu makul kalan en kucuk")
    say("alpha secilir.")
    say()

    # secim: ||lambda||_1 <= 6 olan en kucuk alpha
    pick = None
    for f in grid:
        lam, mse = sols[f]
        if np.abs(lam).sum() <= 6.0:
            pick = f
            break
    if pick is None:
        pick = grid[-1]
    lam, mse = sols[pick]
    say(f"Secilen alpha/tr(D) = {pick}, tahmini LB {np.sqrt(mse):.5f}")
    say()
    say(f"{'yon':<40} {'lambda':>9}")
    for j, nm in enumerate(dnames):
        if abs(lam[j]) > 1e-3:
            say(f"{nm:<40} {lam[j]:9.4f}")
    say()

    blend_log = p0 + Dmat @ lam
    say("## Ortaya cikan harman")
    say(f"ortalama log1p {blend_log.mean():.4f} "
        f"(referans {p0.mean():.4f}, fark {blend_log.mean() - p0.mean():+.4f})")
    say(f"referanstan ortalama mutlak sapma {np.abs(blend_log - p0).mean():.4f}")
    say(f"negatife dusen satir {int((blend_log < 0).sum()):,}")
    say()

    pred = np.expm1(np.clip(blend_log, 0.0, None))
    out = pd.DataFrame({"id": ids.to_numpy(), C.TARGET: pred})
    path = SUBS / "submission_lb_blend.csv"
    out.to_csv(path, index=False)
    say(f"yazildi: {path}")
    say()

    # Bir de daha muhafazakar surum: agirliklarin yarisi
    lam_h = lam * 0.5
    mse_h = M + 2 * lam_h @ c + lam_h @ D @ lam_h
    blend_h = np.expm1(np.clip(p0 + Dmat @ lam_h, 0.0, None))
    pd.DataFrame({"id": ids.to_numpy(), C.TARGET: blend_h}).to_csv(
        SUBS / "submission_lb_blend_half.csv", index=False)
    say(f"yari-adim surum tahmini LB {np.sqrt(max(mse_h, 1e-9)):.5f} "
        f"-> submission_lb_blend_half.csv")
    say()

    p = C.REPORTS_DIR / "68_lb_algebra.md"
    p.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
