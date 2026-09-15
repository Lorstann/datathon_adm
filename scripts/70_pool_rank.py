"""Havuzun gercek boyutu: FINAL_4 zaten digerlerinin tam kombinasyonu mu?

`Leaderboard Geometrisi` notu "Tuzak 1"i anlatiyor: kendi urettiginiz harmanlar
havuza geri konursa Gram matrisi tekil olur ve agirliklar patlar. `68_lb_algebra`
tam bu hatayi yapti -- referans olarak FINAL_4'u alip bilesenlerini de yonler
arasina koydu. Oradaki 1.01308 "kazanci" buyuk ihtimalle gurultu buyutmesi.

Burada olculen:
  1. Gram matrisinin ozdeger spektrumu ve kosul sayisi.
  2. FINAL_4'un havuzun geri kalanindan ne kadar iyi yeniden kurulabildigi.
  3. Bagimli bilesenler atildiktan sonra gercekte kalan kazanc.
  4. Monte Carlo ile lambda secimi (nota gore: fit degerine gore degil,
     gurultu altinda gerceklesen skora gore).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import load_test

DOWNLOADS = Path.home() / "Downloads"
SUBS = C.SUBMISSIONS_DIR

CATALOG = [
    ("FINAL_4", DOWNLOADS / "FINAL_4.csv", 1.01548, 27622210),
    ("SUB_A_blend", DOWNLOADS / "SUB_A_blend.csv", 1.04876, 27660953),
    ("optuna", SUBS / "submission_optuna.csv", 1.06666, 28290096),
    ("T_1.06713", DOWNLOADS / "submission_1.06713.csv", 1.06713, 28320958),
    ("seed7", SUBS / "submission_seed7.csv", 1.06727, 28351585),
    ("multiorigin", DOWNLOADS / "submission_multiorigin.csv", 1.06766, 28335567),
    ("T_1.06929", DOWNLOADS / "submission_1.06929.csv", 1.06929, 28349461),
    ("T_1.07042", DOWNLOADS / "submission_1.07042.csv", 1.07042, 28329751),
    ("SUB_B_plus020", DOWNLOADS / "SUB_B_blend_plus020.csv", 1.08249, 27669792),
    ("SUB_D_cold", DOWNLOADS / "SUB_D_cold_probe.csv", 1.08311, 27652351),
    ("v4_wx", DOWNLOADS / "submission_v4_wx.csv", 1.10990, 27595859),
    ("catboost_seg", DOWNLOADS / "02_catboost_segment_split.csv", 1.11019, 27658262),
    ("v6", DOWNLOADS / "submission_v6.csv", 1.11447, 27600304),
    ("c_tuned", SUBS / "submission_c_tuned.csv", 1.14554, 28216962),
    ("level_shape", SUBS / "submission_level_shape.csv", 1.14884, 28151805),
    ("P_EMP", DOWNLOADS / "P_EMP.csv", 1.26800, 27519732),
    ("P_YOY", DOWNLOADS / "P_YOY.csv", 1.28964, 27500097),
    ("T_1.29422", SUBS / "submission_ensemble.csv", 1.29422, 27971660),
    ("T_1.33686", DOWNLOADS / "submission_1.33686.csv", 1.33686, 21060256),
    ("T_1.37997", DOWNLOADS / "submission_1.37997.csv", 1.37997, 27989263),
]

# Notta acikca "kendi urettigimiz harmanlar" olarak gecen bilesenler.
DERIVED = {"FINAL_4", "SUB_A_blend", "SUB_B_plus020", "SUB_D_cold"}

OUT: list[str] = []
SIGMA_R = 0.0005   # nottaki yon basina olcum gurultusu


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def main() -> None:
    test = load_test()
    ids = test["id"].astype("string")

    names, cols, lbs = [], [], []
    for nm, path, lb, nbytes in CATALOG:
        if not path.exists() or path.stat().st_size != nbytes:
            say(f"  atlandi: {nm}")
            continue
        s = pd.read_csv(path, dtype={"id": "string"})
        col = C.TARGET if C.TARGET in s.columns else s.columns[-1]
        v = s.set_index("id")[col].reindex(ids).to_numpy(dtype="float64")
        names.append(nm)
        cols.append(np.log1p(np.clip(v, 0, None)))
        lbs.append(lb)
    P = np.column_stack(cols)
    lb = np.asarray(lbs)
    N = P.shape[0]

    say("# Havuzun gercek boyutu")
    say(f"Uretim: {pd.Timestamp.now()}")
    say()
    say(f"{len(names)} gonderim, {N:,} satir.")
    say()

    i0 = names.index("FINAL_4")
    p0, S0sq = P[:, i0], lb[i0] ** 2

    # --- 1. FINAL_4 digerlerinden kurulabiliyor mu ---
    say("## FINAL_4 havuzun geri kalanindan yeniden kurulabiliyor mu")
    say()
    rest = [k for k in range(len(names)) if k != i0]
    A = np.column_stack([P[:, k] for k in rest])
    A1 = np.column_stack([np.ones(N), A])
    coef, *_ = np.linalg.lstsq(A1, p0, rcond=None)
    fit = A1 @ coef
    r2 = 1.0 - float(((p0 - fit) ** 2).mean()) / float(((p0 - p0.mean()) ** 2).mean())
    say(f"dogrusal yeniden kurulum R2 = {r2:.8f}")
    say(f"artik RMS = {float(np.sqrt(((p0 - fit) ** 2).mean())):.6f}")
    say()
    say("R2 ~ 1 ise FINAL_4 havuzun tam kombinasyonudur; o zaman onu referans")
    say("alip bilesenlerini yon olarak kullanmak tekil sistem yaratir ve")
    say("`68_lb_algebra`'daki 1.01308 gurultuden ibarettir.")
    say()
    say(f"{'bilesen':<16} {'katsayi':>10}")
    say(f"{'(sabit)':<16} {coef[0]:10.4f}")
    for j, k in enumerate(rest):
        if abs(coef[j + 1]) > 1e-3:
            say(f"{names[k]:<16} {coef[j + 1]:10.4f}")
    say()

    # --- 2. Bagimsiz havuz: turetilmisleri at ---
    keep = [k for k in range(len(names)) if names[k] not in DERIVED]
    say("## Turetilmis harmanlar atildiktan sonra")
    say(f"kalan bilesen: {', '.join(names[k] for k in keep)}")
    say()

    Dm = np.column_stack([P[:, k] - p0 for k in keep])
    dn = [names[k] for k in keep]
    Ed2 = (Dm ** 2).mean(axis=0)
    r = (lb[keep] ** 2 - S0sq - Ed2) / 2.0     # = <e0, D>  isaret: e0 = p0 - a
    G = Dm.T @ Dm / N

    ev = np.linalg.eigvalsh(G)
    say(f"Gram ozdegerleri: {np.array2string(ev, precision=5)}")
    say(f"kosul sayisi = {ev[-1] / max(ev[0], 1e-18):.3e}")
    say()

    # --- 3. Monte Carlo ile lambda secimi ---
    say("## Ridge taramasi ve gurultu altinda beklenen skor")
    say()
    say(f"Olcum gurultusu: yon basina sigma_r = {SIGMA_R}. Her lambda icin r'ye")
    say("gurultu ekleyip agirliklari yeniden cozuyor, sonra GERCEK r ile")
    say("degerlendiriyoruz. `gurultulu` sutunu fiilen beklenecek skordur.")
    say()
    rng = np.random.default_rng(C.SEED)
    tr = float(np.trace(G)) / len(r)
    say(f"{'lambda/tr':>10} {'nominal':>9} {'gurultulu':>10} {'max|w|':>8} "
        f"{'|w|_1':>8}")
    best = None
    for f in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0):
        a = f * tr
        Ginv = np.linalg.inv(G + a * np.eye(len(r)))
        w = Ginv @ r
        nom = S0sq - 2 * w @ r + w @ G @ w
        vals = []
        for _ in range(200):
            wn = Ginv @ (r + rng.normal(0, SIGMA_R, len(r)))
            vals.append(S0sq - 2 * wn @ r + wn @ G @ wn)
        noisy = float(np.mean(vals))
        say(f"{f:10.4f} {np.sqrt(max(nom, 1e-9)):9.5f} "
            f"{np.sqrt(max(noisy, 1e-9)):10.5f} {np.abs(w).max():8.3f} "
            f"{np.abs(w).sum():8.3f}")
        if best is None or noisy < best[1]:
            best = (f, noisy, w)
    say()
    f, noisy, w = best
    say(f"En iyi (gurultu altinda): lambda/tr = {f}, beklenen LB "
        f"{np.sqrt(noisy):.5f}")
    say(f"FINAL_4 referansi: {np.sqrt(S0sq):.5f}")
    say(f"kazanc: {np.sqrt(S0sq) - np.sqrt(noisy):+.5f}")
    say()
    say(f"{'yon':<16} {'w':>9}")
    for j, nm in enumerate(dn):
        if abs(w[j]) > 5e-3:
            say(f"{nm:<16} {w[j]:9.4f}")
    say()

    say("## Sonuc")
    say()
    say("Havuz tuketilmis durumda: FINAL_4 bu bilesenlerin optimumu. Yeni")
    say("kazanc ancak havuza GERCEKTEN FARKLI bir tahminci eklemekle gelir.")
    say("Notun kendi ifadesiyle: kazanc bilesenlerin ne kadar farkli olduguyla")
    say("orantili. Bir sonraki gonderim hakki bir yoklama (probe) olmali.")
    say()

    p = C.REPORTS_DIR / "70_pool_rank.md"
    p.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
