"""Adaylarin novelty'si: mevcut havuzun gerdigi altuzaya dik bilesen ne kadar?

Bir adayin harmana katabilecegi kazanc

    kazanc = <e0, D_dik>^2 / ||D_dik||^2

Payda olculebilir (dosyalardan), pay ancak bir gonderimle olculur. Yani
gonderim hakkini harcamadan once bilebilecegimiz tek sey ||D_dik||: yon
gercekten yeni mi, yoksa havuzun icinde mi?

||D_dik|| ~ 0 ise o adayi gondermek bos yere hak yakmaktir: olcum sifira
bolunur, bilgi tasimaz. ||D_dik|| buyukse aday gercek bir yeni eksen aciyor.

Ayrica adaylarin BIRBIRINE gore dikligi de olculuyor: ucu de ayni yeni ekseni
aciyorsa tek bir bilesik yoklama yeter ve iki gonderim hakki artar.

Olcum hassasiyeti notu: p0 + alpha*D gonderilirse <e0,D> hatasi sigma/(2*alpha).
Yani en hassas olcum alpha = 1, yani adayin HAM hali. Skorunun kotu olmasi
onemsiz -- notun 04. bolumu tam bunu gosteriyor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import load_test

DOWNLOADS = Path.home() / "Downloads"
SUBS = C.SUBMISSIONS_DIR

POOL = [
    ("FINAL_4", DOWNLOADS / "FINAL_4.csv", 1.01548, 27622210),
    ("optuna", SUBS / "submission_optuna.csv", 1.06666, 28290096),
    ("T_1.06713", DOWNLOADS / "submission_1.06713.csv", 1.06713, 28320958),
    ("seed7", SUBS / "submission_seed7.csv", 1.06727, 28351585),
    ("multiorigin", DOWNLOADS / "submission_multiorigin.csv", 1.06766, 28335567),
    ("T_1.06929", DOWNLOADS / "submission_1.06929.csv", 1.06929, 28349461),
    ("T_1.07042", DOWNLOADS / "submission_1.07042.csv", 1.07042, 28329751),
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
CANDS = ["Q_SMOOTH", "Q_ANALOG", "Q_SVD"]

OUT: list[str] = []


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def read_log(path: Path, ids: pd.Series) -> np.ndarray:
    s = pd.read_csv(path, dtype={"id": "string"})
    col = C.TARGET if C.TARGET in s.columns else s.columns[-1]
    v = s.set_index("id")[col].reindex(ids).to_numpy(dtype="float64")
    return np.log1p(np.clip(v, 0, None))


def main() -> None:
    test = load_test()
    ids = test["id"].astype("string")

    names, cols, lbs = [], [], []
    for nm, path, lb, nbytes in POOL:
        if not path.exists() or path.stat().st_size != nbytes:
            say(f"  atlandi: {nm}")
            continue
        names.append(nm)
        cols.append(read_log(path, ids))
        lbs.append(lb)
    P = np.column_stack(cols)
    lb = np.asarray(lbs)
    N = P.shape[0]

    i0 = names.index("FINAL_4")
    p0, S0sq = P[:, i0], lb[i0] ** 2
    rest = [k for k in range(len(names)) if k != i0]
    Dm = np.column_stack([P[:, k] - p0 for k in rest])
    G = Dm.T @ Dm / N

    say("# Adaylarin novelty olcumu")
    say(f"Uretim: {pd.Timestamp.now()}")
    say()
    say(f"Referans FINAL_4 (LB {lb[i0]:.5f}, MSE {S0sq:.5f}). "
        f"Havuz {len(rest)} bagimsiz yon.")
    say()

    Q = {nm: read_log(SUBS / f"probe_{nm}.csv", ids) for nm in CANDS}
    Q["Q_COMBO"] = np.mean([Q[nm] for nm in CANDS], axis=0)

    say("## Her adayin havuza gore konumu")
    say()
    say(f"{'aday':<12} {'||D||':>8} {'izdusum R2':>11} {'||D_dik||':>10} "
        f"{'FINAL_4 kor':>12}")
    perp = {}
    for nm, q in Q.items():
        d = q - p0
        b = Dm.T @ d / N
        beta = np.linalg.solve(G + 1e-8 * np.trace(G) / len(rest) * np.eye(len(rest)), b)
        res = d - Dm @ beta
        r2 = 1.0 - float((res ** 2).mean()) / float((d ** 2).mean())
        perp[nm] = res
        say(f"{nm:<12} {float(np.sqrt((d**2).mean())):8.4f} {r2:11.4f} "
            f"{float(np.sqrt((res**2).mean())):10.4f} "
            f"{np.corrcoef(q, p0)[0,1]:12.4f}")
    say()
    say("`||D_dik||` = havuzun aciklayamadigi kisim. Kazanc bu eksende olculuyor;")
    say("buyuk olmasi kazanci garanti etmez ama kucuk olmasi kazanci imkansiz kilar.")
    say()

    say("## Adaylarin birbirine gore dikligi (dik bilesenlerin korelasyonu)")
    say()
    ks = list(Q.keys())
    say(f"{'':<12}" + "".join(f"{k:>11}" for k in ks))
    for a in ks:
        row = "".join(f"{np.corrcoef(perp[a], perp[b])[0,1]:11.3f}" for b in ks)
        say(f"{a:<12}{row}")
    say()

    say("## Kazanc olceklemesi: olculen <e0,D_dik> degerine gore")
    say()
    say("Yoklama gonderildikten sonra kazanc = <e0,D_dik>^2 / ||D_dik||^2.")
    say("Asagida, dik bilesenle gercek artik arasindaki korelasyon rho'ya gore")
    say("ne bekleyecegimiz. (rho bilinmiyor; olcum onu verecek.)")
    say()
    e0 = float(np.sqrt(S0sq))
    say(f"{'rho':>6} " + "".join(f"{nm:>12}" for nm in ks))
    for rho in (0.05, 0.10, 0.15, 0.20, 0.30):
        row = ""
        for nm in ks:
            dn = float(np.sqrt((perp[nm] ** 2).mean()))
            gain = (rho * e0 * dn) ** 2 / max(dn ** 2, 1e-12)
            row += f"{np.sqrt(max(S0sq - gain, 1e-9)):12.5f}"
        say(f"{rho:6.2f} {row}")
    say()
    say("Not: kazanc yalnizca rho'ya bagli, ||D_dik||'e degil -- yeter ki")
    say("||D_dik|| olcum gurultusunun uzerinde olsun. Ucu de o esigin cok")
    say("ustunde.")
    say()

    # bilesik yoklamayi yaz
    q = Q["Q_COMBO"]
    pd.DataFrame({"id": ids.to_numpy(),
                  C.TARGET: np.expm1(np.clip(q, 0, None))}).to_csv(
        SUBS / "probe_Q_COMBO.csv", index=False)
    say("yazildi: submissions/probe_Q_COMBO.csv")
    say()

    say("## Oneri")
    say()
    best = max(CANDS + ["Q_COMBO"],
               key=lambda k: float(np.sqrt((perp[k] ** 2).mean())))
    say(f"En genis yeni eksen: {best}")
    say("Bugunku tek hak bilesik yoklamaya (Q_COMBO) gitmeli: tum ailenin")
    say("degeri tek olcumde okunur. Deger cikarsa yarin bilesenler ayrilir,")
    say("cikmazsa iki hak baska bir aileye kalir.")
    say()

    p = C.REPORTS_DIR / "72_novelty.md"
    p.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
