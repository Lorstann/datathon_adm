"""FINAL_4 (LB 1.01548) ile depodaki en iyi (LB 1.06713) arasindaki farki ac.

Depo raporlari 1.06713'te duruyor ama Kaggle gonderim gecmisi bugun
FINAL_4.csv ile 1.01548 gosteriyor. O dosya depoda uretilmemis (Downloads'ta,
tarayiciyla indirilmis). Bu betik dosyayi tersine okuyup ne yaptigini
cikariyor: hangi dilimde nasil farklilasiyor, hangi capayi kullaniyor,
sifirlari nasil ele aliyor.

Amac: kalan gonderim hakkini (bugun 1, yarin 3) neyin uzerine harcayacagimizi
bilmek.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import load_test, load_train
from src.validation.folds import guc_band

DOWNLOADS = Path.home() / "Downloads"
FILES = {
    "FINAL_4 (1.01548)": DOWNLOADS / "FINAL_4.csv",
    "FINAL_1 (1.02571)": DOWNLOADS / "FINAL_1.csv",
    "SUB_A_blend (1.04876)": DOWNLOADS / "SUB_A_blend.csv",
    "SUB_D_cold (1.08311)": DOWNLOADS / "SUB_D_cold_probe.csv",
    "P_YOY (1.28964)": DOWNLOADS / "P_YOY.csv",
    "P_EMP (1.26800)": DOWNLOADS / "P_EMP.csv",
    "repo BEST (1.06713)": C.SUBMISSIONS_DIR / "submission_BEST_1.06713.csv",
    "repo optuna (1.06666)": C.SUBMISSIONS_DIR / "submission_optuna.csv",
}
OUT = []


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def main() -> None:
    train, test = load_train(), load_test()
    seen = set(train[C.ENTITY].unique())

    # Mart 2026'da canli/olu durumu
    h = train.loc[train[C.DATE] > C.TRAIN_END - pd.Timedelta(days=28)]
    g = h.groupby(C.ENTITY)[C.TARGET]
    lvl28 = g.apply(lambda s: float(np.log1p(s.clip(lower=0)).mean()))
    zero28 = g.apply(lambda s: float((s <= 0).mean()))

    base = test[["id", C.ENTITY, C.DATE, C.POWER, C.LOCATION]].copy()
    base["cold"] = ~base[C.ENTITY].isin(seen)
    base["lvl28"] = base[C.ENTITY].map(lvl28)
    base["z28"] = base[C.ENTITY].map(zero28)
    base["olu"] = (base["z28"] > 0.9).fillna(False)
    base["ay"] = base[C.DATE].dt.month
    base["band"] = guc_band(base[C.POWER]).astype("string")
    base["dilim"] = np.where(base["cold"], "cold",
                             np.where(base["olu"], "warm_olu", "warm_canli"))

    preds = {}
    for name, p in FILES.items():
        if not p.exists():
            say(f"(yok: {p})")
            continue
        s = pd.read_csv(p, dtype={"id": "string"})
        col = C.TARGET if C.TARGET in s.columns else s.columns[-1]
        m = base.merge(s[["id", col]].rename(columns={col: "pred"}), on="id", how="left")
        preds[name] = np.log1p(m["pred"].clip(lower=0).to_numpy())

    say("# FINAL_4 vs depo: gonderimlerin anatomisi")
    say(f"Uretim: {pd.Timestamp.now()}")
    say()
    say("## Genel")
    say(f"{'gonderim':<24} {'ort log1p':>10} {'medyan':>8} {'std':>7} "
        f"{'p<0.5 orani':>12} {'tam 0':>8}")
    for name, lp in preds.items():
        raw = np.expm1(lp)
        say(f"{name:<24} {lp.mean():10.4f} {np.median(lp):8.4f} {lp.std():7.4f} "
            f"{float((raw <= 0.5).mean()):12.4f} {int((raw <= 1e-9).sum()):8d}")
    say()

    say("## Dilim bazinda ortalama log1p tahmini")
    dil = base["dilim"].to_numpy()
    say(f"{'gonderim':<24} " + "".join(f"{d:>13}" for d in
                                       ["cold", "warm_canli", "warm_olu"]))
    for name, lp in preds.items():
        row = "".join(f"{lp[dil == d].mean():13.4f}" for d in
                      ["cold", "warm_canli", "warm_olu"])
        say(f"{name:<24} {row}")
    say()

    say("## Ay bazinda ortalama log1p tahmini (mevsimsel rampa)")
    say(f"{'gonderim':<24} " + "".join(f"{a:>9}" for a in [4, 5, 6, 7]) + f"{'Nis->Tem':>10}")
    ay = base["ay"].to_numpy()
    for name, lp in preds.items():
        vals = [lp[ay == a].mean() for a in (4, 5, 6, 7)]
        say(f"{name:<24} " + "".join(f"{v:9.4f}" for v in vals)
            + f"{vals[-1] - vals[0]:10.4f}")
    say()

    say("## Warm canli satirlarda: tahmin eksi Mart son-28g seviyesi")
    sel = (base["dilim"] == "warm_canli").to_numpy() & base["lvl28"].notna().to_numpy()
    lv = base["lvl28"].to_numpy()
    for name, lp in preds.items():
        d = lp[sel] - lv[sel]
        say(f"{name:<24} ortalama {d.mean():+.4f}  medyan {np.median(d):+.4f}")
    say()

    say("## Cold satirlarda: tahmin eksi log(guc)  (yani ima edilen z)")
    lg = np.log(base[C.POWER].clip(lower=1).astype("float64")).to_numpy()
    selc = (base["dilim"] == "cold").to_numpy()
    for name, lp in preds.items():
        z = lp[selc] - lg[selc]
        say(f"{name:<24} ortalama z {z.mean():+.4f}  medyan {np.median(z):+.4f}")
    say()

    if "FINAL_4 (1.01548)" in preds and "repo BEST (1.06713)" in preds:
        a, b = preds["FINAL_4 (1.01548)"], preds["repo BEST (1.06713)"]
        say("## FINAL_4 - repo BEST farkinin dagilimi")
        d = a - b
        say(f"korelasyon {np.corrcoef(a, b)[0,1]:.4f}, ortalama fark {d.mean():+.4f}, "
            f"std {d.std():.4f}")
        say()
        say(f"{'dilim':<12} {'n':>9} {'ort fark':>10} {'|fark| medyan':>14}")
        for dd in ["cold", "warm_canli", "warm_olu"]:
            s = dil == dd
            say(f"{dd:<12} {int(s.sum()):9d} {d[s].mean():10.4f} "
                f"{np.median(np.abs(d[s])):14.4f}")
        say()
        say(f"{'band':<12} {'n':>9} {'ort fark':>10}")
        bands = base["band"].to_numpy()
        for bd in pd.unique(bands):
            s = bands == bd
            say(f"{str(bd):<12} {int(s.sum()):9d} {d[s].mean():10.4f}")
        say()

        # iki gonderimin harmani ne verir (log uzayinda)
        say("## Harman aritmetigi (log uzayinda w*FINAL_4 + (1-w)*repo)")
        say("Gercek bilinmiyor; yalnizca ne kadar ayrildiklarini gosterir.")
        for w in (0.5, 0.7, 0.85):
            bl = w * a + (1 - w) * b
            say(f"w={w:.2f}  ortalama {bl.mean():.4f}  "
                f"FINAL_4'ten ortalama sapma {np.abs(bl - a).mean():.4f}")
        say()

    p = C.REPORTS_DIR / "67_compare_best.md"
    p.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
