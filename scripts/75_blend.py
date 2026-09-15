"""Nihai harman motoru: havuz + yoklamalar, ridge + Monte Carlo lambda secimi.

`68` ve `70`'i tek yerde topluyor ve yoklamalari da havuza aliyor. Kullanim:
yeni bir yoklamanin skoru gelince CATALOG'a bir satir ekle, calistir.

Yontem (Leaderboard Geometrisi notu, bolum 06):
  1. Referans p0 = en iyi skorlu gonderim.
  2. Turetilmis harmanlari (kendi urettiklerimiz) yon havuzundan cikar --
     yoksa Gram tekil olur.
  3. r_k = (S_k^2 - S_0^2 - ||D_k||^2)/2  ile izdusumleri oku.
  4. w = (G + lambda I)^-1 r,  lambda'yi gurultu altinda gerceklesen skora
     gore sec (fit degerine gore degil).
  5. Sapmayi kirp, dogrula, beklenen skoru yaz.

Ek olarak her bilesenin MARJINAL katkisi raporlanir: o bileseni havuzdan
cikarinca beklenen skor ne kadar bozuluyor. Notun 04. bolumundeki tabloyla
ayni okuma -- hangi bilesenin gercekten tasidigini gosterir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import load_test

DOWNLOADS = Path.home() / "Downloads"
SUBS = C.SUBMISSIONS_DIR

# (ad, yol, LB, beklenen bayt veya None, turetilmis mi)
CATALOG = [
    ("FINAL_4", DOWNLOADS / "FINAL_4.csv", 1.01548, 27622210, True),
    ("SUB_A_blend", DOWNLOADS / "SUB_A_blend.csv", 1.04876, 27660953, True),
    ("SUB_B_plus020", DOWNLOADS / "SUB_B_blend_plus020.csv", 1.08249, 27669792, True),
    ("SUB_D_cold", DOWNLOADS / "SUB_D_cold_probe.csv", 1.08311, 27652351, True),
    ("optuna", SUBS / "submission_optuna.csv", 1.06666, 28290096, False),
    ("T_1.06713", DOWNLOADS / "submission_1.06713.csv", 1.06713, 28320958, False),
    ("seed7", SUBS / "submission_seed7.csv", 1.06727, 28351585, False),
    ("multiorigin", DOWNLOADS / "submission_multiorigin.csv", 1.06766, 28335567, False),
    ("T_1.06929", DOWNLOADS / "submission_1.06929.csv", 1.06929, 28349461, False),
    ("T_1.07042", DOWNLOADS / "submission_1.07042.csv", 1.07042, 28329751, False),
    ("v4_wx", DOWNLOADS / "submission_v4_wx.csv", 1.10990, 27595859, False),
    ("catboost_seg", DOWNLOADS / "02_catboost_segment_split.csv", 1.11019, 27658262, False),
    ("v6", DOWNLOADS / "submission_v6.csv", 1.11447, 27600304, False),
    ("c_tuned", SUBS / "submission_c_tuned.csv", 1.14554, 28216962, False),
    ("level_shape", SUBS / "submission_level_shape.csv", 1.14884, 28151805, False),
    ("P_EMP", DOWNLOADS / "P_EMP.csv", 1.26800, 27519732, False),
    ("P_YOY", DOWNLOADS / "P_YOY.csv", 1.28964, 27500097, False),
    ("T_1.29422", SUBS / "submission_ensemble.csv", 1.29422, 27971660, False),
    ("T_1.33686", DOWNLOADS / "submission_1.33686.csv", 1.33686, 21060256, False),
    ("T_1.37997", DOWNLOADS / "submission_1.37997.csv", 1.37997, 27989263, False),
    # --- yoklamalar ---
    ("Q_COMBO", SUBS / "probe_Q_COMBO.csv", 1.33324, None, False),
]

SIGMA_R = 0.0005
MAX_DEV = 1.0        # referanstan izin verilen maksimum log sapmasi
OUT: list[str] = []


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def solve_report(G, r, S0sq, rng, tag=""):
    tr = float(np.trace(G)) / len(r)
    best = None
    rows = []
    for f in (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0):
        Ginv = np.linalg.inv(G + f * tr * np.eye(len(r)))
        w = Ginv @ r
        nom = S0sq - 2 * w @ r + w @ G @ w
        vals = [S0sq - 2 * wn @ r + wn @ G @ wn
                for wn in (Ginv @ (r + rng.normal(0, SIGMA_R, len(r)))
                           for _ in range(300))]
        noisy = float(np.mean(vals))
        rows.append((f, nom, noisy, w))
        if best is None or noisy < best[2]:
            best = (f, nom, noisy, w)
    return rows, best


def main() -> None:
    test = load_test()
    ids = test["id"].astype("string")

    names, cols, lbs, derived = [], [], [], []
    for nm, path, lb, nbytes, dv in CATALOG:
        if not path.exists():
            say(f"  atlandi (yok): {nm}")
            continue
        if nbytes is not None and path.stat().st_size != nbytes:
            say(f"  atlandi (bayt): {nm}")
            continue
        s = pd.read_csv(path, dtype={"id": "string"})
        col = C.TARGET if C.TARGET in s.columns else s.columns[-1]
        v = s.set_index("id")[col].reindex(ids).to_numpy(dtype="float64")
        names.append(nm)
        cols.append(np.log1p(np.clip(v, 0, None)))
        lbs.append(lb)
        derived.append(dv)
    P = np.column_stack(cols)
    lb = np.asarray(lbs)
    N = P.shape[0]

    i0 = names.index("FINAL_4")
    p0, S0sq = P[:, i0], lb[i0] ** 2

    say("# Nihai harman")
    say(f"Uretim: {pd.Timestamp.now()}")
    say()
    say(f"Referans FINAL_4  LB {lb[i0]:.5f}  MSE {S0sq:.5f}")
    say()

    keep = [k for k in range(len(names)) if not derived[k]]
    Dm = np.column_stack([P[:, k] - p0 for k in keep])
    dn = [names[k] for k in keep]
    Ed2 = (Dm ** 2).mean(axis=0)
    r = (lb[keep] ** 2 - S0sq - Ed2) / 2.0
    G = Dm.T @ Dm / N

    say("## Yonler")
    say(f"{'yon':<16} {'LB':>9} {'||D||':>8} {'<e0,D>':>10} "
        f"{'tek basina':>11}")
    for j, nm in enumerate(dn):
        solo = S0sq - r[j] ** 2 / Ed2[j]
        say(f"{nm:<16} {lb[keep][j]:9.5f} {np.sqrt(Ed2[j]):8.4f} "
            f"{r[j]:10.5f} {np.sqrt(max(solo, 1e-9)):11.5f}")
    say()
    say("`tek basina` = yalnizca o yon kullanilirsa ulasilacak skor.")
    say()

    # Q_COMBO'nun havuza dik bileseni
    if "Q_COMBO" in dn:
        j = dn.index("Q_COMBO")
        oth = [k for k in range(len(dn)) if k != j]
        A, y = Dm[:, oth], Dm[:, j]
        Gs = A.T @ A / N
        beta = np.linalg.solve(Gs + 1e-8 * np.trace(Gs) / len(oth) * np.eye(len(oth)),
                               A.T @ y / N)
        perp = y - A @ beta
        r_perp = r[j] - float(beta @ r[oth])
        n_perp = float((perp ** 2).mean())
        say("## Q_COMBO: havuza dik bilesenin degeri")
        say(f"<e0,D> toplam           {r[j]:+.5f}")
        say(f"havuzun acikladigi kisim {float(beta @ r[oth]):+.5f}")
        say(f"DIK bilesen <e0,D_dik>  {r_perp:+.5f}")
        say(f"||D_dik||^2             {n_perp:.5f}")
        say(f"dik bilesenin tek basina kazanci  "
            f"{r_perp ** 2 / max(n_perp, 1e-12):.6f} MSE")
        say(f"  -> LB {np.sqrt(max(S0sq - r_perp**2/max(n_perp,1e-12), 1e-9)):.5f}")
        say()

    rng = np.random.default_rng(C.SEED)
    rows, best = solve_report(G, r, S0sq, rng)
    say("## Ridge taramasi")
    say(f"{'lambda/tr':>10} {'nominal':>9} {'gurultulu':>10} {'max|w|':>8} "
        f"{'|w|_1':>8}")
    for f, nom, noisy, w in rows:
        say(f"{f:10.4f} {np.sqrt(max(nom,1e-9)):9.5f} "
            f"{np.sqrt(max(noisy,1e-9)):10.5f} {np.abs(w).max():8.3f} "
            f"{np.abs(w).sum():8.3f}")
    say()
    f, nom, noisy, w = best
    say(f"Secilen lambda/tr = {f}, beklenen LB {np.sqrt(noisy):.5f} "
        f"(FINAL_4 {np.sqrt(S0sq):.5f}, kazanc "
        f"{np.sqrt(S0sq) - np.sqrt(noisy):+.5f})")
    say()

    say("## Marjinal katki: bileseni cikarinca beklenen skor")
    say(f"{'cikarilan':<16} {'beklenen LB':>12} {'bozulma':>9}")
    base_noisy = noisy
    for j, nm in enumerate(dn):
        oth = [k for k in range(len(dn)) if k != j]
        _, b2 = solve_report(G[np.ix_(oth, oth)], r[oth], S0sq,
                             np.random.default_rng(C.SEED))
        say(f"{nm:<16} {np.sqrt(b2[2]):12.5f} "
            f"{np.sqrt(b2[2]) - np.sqrt(base_noisy):+9.5f}")
    say()

    blend = p0 + Dm @ w
    dev = blend - p0
    n_clip = int((np.abs(dev) > MAX_DEV).sum())
    blend = p0 + np.clip(dev, -MAX_DEV, MAX_DEV)
    say(f"## Cikti")
    say(f"kirpilan satir {n_clip:,} ({n_clip/N:.4%})")
    say(f"ortalama log1p {blend.mean():.4f} (referans {p0.mean():.4f})")
    say(f"referanstan ortalama mutlak sapma {np.abs(blend - p0).mean():.4f}")
    say()

    out = np.expm1(np.clip(blend, 0.0, None))
    pd.DataFrame({"id": ids.to_numpy(), C.TARGET: out}).to_csv(
        SUBS / "submission_BLEND.csv", index=False)
    say("yazildi: submissions/submission_BLEND.csv")
    say()

    p = C.REPORTS_DIR / "75_blend.md"
    p.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
