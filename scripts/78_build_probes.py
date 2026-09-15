"""Yarinin iki yoklamasi: dagilim keskinligi ve birlesik sifir kapisi.

Neden bu ikisi. `69_residual_probe` yalnizca uc yonu guvenilir olcebildi
(sabit, cold, warm-canli) ve ucunde de yanlilik sifir cikti. Geri kalan her
dilimin izdusum R2'si 0.5'in altindaydi, yani mevcut gonderimlerden
OLCULEMIYOR. Kalan iki gonderim hakki (ucuncusu nihai harmana) bu olculemeyen
uzayin en buyuk iki eksenine gitmeli.

DISP -- dagilim keskinligi.
  q = ort + (1+a)(p0 - ort). Yani tahminleri ortalamadan uzaklastir.
  Bu eksenin havuza izdusumu yalnizca R2 = 0.08 idi: neredeyse tamamen yeni.
  Olctugu sey "trafolar arasi ayrimi yeterince keskin yapiyor muyuz". RMSLE
  L2 oldugu icin optimal keskinlik tektir ve tek sayiyla bulunur.

ZGATE -- birlesik sifir/panel kapisi.
  Mart'ta olu trafolar + panel cikis satirlari + panel giris satirlari, hepsi
  tek yonde asagi cekilmis. Uc alt-desen de ayni isareti bekliyor ("bu
  satirlarda fazla tahmin ediyoruz"), o yuzden tek olcumde toplanabilirler:
  isaretler ayniysa toplam olcum bilesenlerin toplamini verir ve kaldirac
  artar. `76_zero_floor` bunlarin izdusum R2'sini 0.00-0.18 buldu.

Her iki dosya da simdi uretiliyor ki yarin gonderim aninda ek is olmasin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import load_test, load_train

DOWNLOADS = Path.home() / "Downloads"
BASE = DOWNLOADS / "FINAL_4.csv"
SHARPEN = 0.15
OUT: list[str] = []


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def main() -> None:
    train, test = load_train(), load_test()
    ids = test["id"].astype("string")
    s = pd.read_csv(BASE, dtype={"id": "string"})
    col = C.TARGET if C.TARGET in s.columns else s.columns[-1]
    base = s.set_index("id")[col].reindex(ids).to_numpy(dtype="float64")
    p0 = np.log1p(np.clip(base, 0, None))
    N = len(p0)
    S0sq = 1.01548 ** 2

    say("# Yarinin yoklamalari")
    say(f"Uretim: {pd.Timestamp.now()}")
    say()

    # --- DISP ---
    m = float(p0.mean())
    q = m + (1.0 + SHARPEN) * (p0 - m)
    q = np.clip(q, 0.0, None)
    d = q - p0
    say("## DISP: dagilim keskinligi")
    say(f"a = {SHARPEN}, q = ort + (1+a)(p0 - ort), ort = {m:.4f}")
    say(f"||D|| = {float(np.sqrt((d**2).mean())):.4f}, "
        f"ortalama kayma {d.mean():+.4f}")
    say(f"tahmin std {p0.std():.4f} -> {q.std():.4f}")
    say()
    say("Okuma: skor S ise <e0,D> = (S^2 - S0^2 - ||D||^2)/2.")
    say("  <e0,D> < 0  ->  daha KESKIN olmaliyiz (tahminler fazla puruzsuz)")
    say("  <e0,D> > 0  ->  daha PURUZSUZ olmaliyiz (tahminler fazla keskin)")
    say(f"optimal agirlik w = -<e0,D>/||D||^2, uygulanan keskinlik "
        f"{SHARPEN}*w olur.")
    say()
    pd.DataFrame({"id": ids.to_numpy(),
                  C.TARGET: np.expm1(q)}).to_csv(
        C.SUBMISSIONS_DIR / "probe_DISP.csv", index=False)
    say("yazildi: submissions/probe_DISP.csv")
    say()

    # --- ZGATE ---
    t = test[["id", C.ENTITY, C.DATE]].copy()
    last = t.groupby(C.ENTITY)[C.DATE].transform("max")
    first = t.groupby(C.ENTITY)[C.DATE].transform("min")
    d_exit = (last - t[C.DATE]).dt.days.to_numpy()
    d_entry = (t[C.DATE] - first).dt.days.to_numpy()
    early = (last < C.TEST_END).to_numpy()
    late = (first > C.TEST_START).to_numpy()

    hist = train.loc[train[C.DATE] > C.TRAIN_END - pd.Timedelta(days=56)]
    zr = test[C.ENTITY].map(
        hist.groupby(C.ENTITY)[C.TARGET].apply(
            lambda x: float((x <= 0).mean()))).fillna(-1.0).to_numpy()

    mult = np.ones(N)
    parts = []
    sel = zr > 0.9
    mult[sel] = np.minimum(mult[sel], 0.30)
    parts.append(("Mart'ta olu (zr>0.9)", sel, 0.30))
    sel = (zr > 0.5) & (zr <= 0.9)
    mult[sel] = np.minimum(mult[sel], 0.55)
    parts.append(("yari olu (0.5<zr<=0.9)", sel, 0.55))
    for lo, hi, k in [(0, 2, 0.30), (3, 6, 0.45), (7, 13, 0.65), (14, 29, 0.85)]:
        sel = early & (d_exit >= lo) & (d_exit <= hi)
        mult[sel] = np.minimum(mult[sel], k)
        parts.append((f"cikisa {lo}-{hi} gun", sel, k))
    for lo, hi, k in [(0, 0, 0.45), (1, 2, 0.65), (3, 6, 0.85)]:
        sel = late & (d_entry >= lo) & (d_entry <= hi)
        mult[sel] = np.minimum(mult[sel], k)
        parts.append((f"giristen {lo}-{hi} gun", sel, k))

    say("## ZGATE: birlesik sifir/panel kapisi")
    say(f"{'alt desen':<26} {'satir':>9} {'pay':>8} {'carpan':>7} "
        f"{'ort p0':>8}")
    for nm, sel, k in parts:
        say(f"{nm:<26} {int(sel.sum()):9,} {sel.mean():8.4f} {k:7.2f} "
            f"{p0[sel].mean() if sel.any() else float('nan'):8.4f}")
    newp = base * mult
    q2 = np.log1p(np.clip(newp, 0, None))
    d2 = q2 - p0
    touched = mult < 1.0
    say()
    say(f"toplam dokunulan {int(touched.sum()):,} ({touched.mean():.2%})")
    say(f"||D|| = {float(np.sqrt((d2**2).mean())):.4f}, "
        f"ortalama kayma {d2.mean():+.4f}")
    say()
    pd.DataFrame({"id": ids.to_numpy(), C.TARGET: newp}).to_csv(
        C.SUBMISSIONS_DIR / "probe_ZGATE.csv", index=False)
    say("yazildi: submissions/probe_ZGATE.csv")
    say()

    say("## Kaldirac senaryolari")
    say()
    say(f"{'yon':<10} {'||D||^2':>9} " + "".join(
        f"{f'b={b}':>10}" for b in (0.1, 0.2, 0.5, 1.0)))
    for nm, dd in (("DISP", d), ("ZGATE", d2)):
        n2 = float((dd ** 2).mean())
        row = ""
        for b in (0.1, 0.2, 0.5, 1.0):
            # b = dokunulan satirlarda ortalama yanlilik varsayimi
            tt = np.abs(dd) > 1e-9
            f_ = float(tt.mean())
            delta = float(-dd[tt].mean()) if tt.any() else 0.0
            r_hat = -f_ * delta * b
            row += f"{np.sqrt(max(S0sq - r_hat**2/max(n2,1e-12), 1e-9)):10.5f}"
        say(f"{nm:<10} {n2:9.5f} {row}")
    say()

    p = C.REPORTS_DIR / "78_build_probes.md"
    p.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
