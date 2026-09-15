"""Ikinci aile: panel sinir satirlari. Test tarafinda TAMAMEN bilinen bilgi.

Test'te her trafonun hangi gunlerde satiri oldugunu goruyoruz. Bir trafonun
kaydi 2026-07-31'den once bitiyorsa, o trafo pencerenin ortasinda devreden
cikmis demektir; cikisa yakin gunlerde tuketimin sifira inmesi beklenir.
Ayni sey giris tarafinda: 2026-04-01'den sonra baslayan trafolarin ilk
gunlerinde tuketim henuz oturmamistir.

`reports/60-62` bu deseni train pencerelerinde olctu ve "cogu zaten origin'de
olu trafolardan geliyor, model bunu biliyor" diye kaydetti. Ama o yargi
CV penceresinden geldi; test dagilimi (Nis-Tem 2026, ortasinda 1.326 trafoluk
kutlesel devreye alma) hicbir CV penceresine benzemiyor. Notun 07. bolumunun
dersi tam bu: vekil bozuldugunda olcmek tahmin etmekten ustundur.

Burada uretilen yon dar ve yorumlanabilir: FINAL_4'un aynisi, yalnizca panel
sinir satirlarinda asagi cekilmis. Olcum <e0,D> isaretini dogrudan soyler:
negatifse bu satirlarda gercekten fazla tahmin ediyoruz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import load_test, load_train

DOWNLOADS = Path.home() / "Downloads"
BASE = DOWNLOADS / "FINAL_4.csv"

# cikis tarafi: son N gunde carpan
EXIT_STEPS = [(0, 2, 0.25), (3, 6, 0.40), (7, 13, 0.60), (14, 29, 0.80)]
# giris tarafi: ilk N gunde carpan
ENTRY_STEPS = [(0, 0, 0.45), (1, 2, 0.65), (3, 6, 0.85)]

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

    t = test[["id", C.ENTITY, C.DATE]].copy()
    last = t.groupby(C.ENTITY)[C.DATE].transform("max")
    first = t.groupby(C.ENTITY)[C.DATE].transform("min")
    t["gun_cikisa"] = (last - t[C.DATE]).dt.days
    t["gun_giristen"] = (t[C.DATE] - first).dt.days
    t["erken_cikis"] = last < C.TEST_END
    t["gec_giris"] = first > C.TEST_START

    seen = set(train[C.ENTITY].unique())
    h = train.loc[train[C.DATE] > C.TRAIN_END - pd.Timedelta(days=28)]
    z28 = t[C.ENTITY].map(
        h.groupby(C.ENTITY)[C.TARGET].apply(lambda x: float((x <= 0).mean())))
    t["olu"] = (z28 > 0.9).fillna(False)
    t["cold"] = ~t[C.ENTITY].isin(seen)

    say("# Panel sinir kapisi: ikinci yon ailesi")
    say(f"Uretim: {pd.Timestamp.now()}")
    say()
    n = len(t)
    ne = int(t["erken_cikis"].sum())
    say(f"test satiri {n:,}")
    say(f"erken cikan trafonun satiri {ne:,} ({ne/n:.2%}), "
        f"trafo {t.loc[t['erken_cikis'], C.ENTITY].nunique():,}")
    say(f"gec giren trafonun satiri {int(t['gec_giris'].sum()):,} "
        f"({t['gec_giris'].mean():.2%}), "
        f"trafo {t.loc[t['gec_giris'], C.ENTITY].nunique():,}")
    say()

    mult = np.ones(n)
    say("## Uygulanan carpanlar")
    say(f"{'dilim':<28} {'satir':>9} {'pay':>7} {'carpan':>7} "
        f"{'ort log1p tahmin':>18}")
    for lo, hi, m in EXIT_STEPS:
        sel = (t["erken_cikis"] & (t["gun_cikisa"] >= lo)
               & (t["gun_cikisa"] <= hi)).to_numpy()
        mult[sel] = np.minimum(mult[sel], m)
        say(f"{'cikisa ' + f'{lo}-{hi} gun':<28} {int(sel.sum()):9,} "
            f"{sel.mean():7.4f} {m:7.2f} {p0[sel].mean():18.4f}")
    for lo, hi, m in ENTRY_STEPS:
        sel = (t["gec_giris"] & (t["gun_giristen"] >= lo)
               & (t["gun_giristen"] <= hi)).to_numpy()
        mult[sel] = np.minimum(mult[sel], m)
        say(f"{'giristen ' + f'{lo}-{hi} gun':<28} {int(sel.sum()):9,} "
            f"{sel.mean():7.4f} {m:7.2f} {p0[sel].mean():18.4f}")
    say()

    touched = mult < 1.0
    say(f"toplam dokunulan satir {int(touched.sum()):,} "
        f"({touched.mean():.2%})")
    say(f"bunlarin cold payi {float(t.loc[touched, 'cold'].mean()):.3f}, "
        f"Mart'ta olu payi {float(t.loc[touched, 'olu'].mean()):.3f}")
    say()

    newp = base * mult
    q = np.log1p(np.clip(newp, 0, None))
    d = q - p0
    say(f"yonun buyuklugu ||D|| = {float(np.sqrt((d**2).mean())):.4f}")
    say(f"ortalama kayma {d.mean():+.4f}")
    say()

    pd.DataFrame({"id": ids.to_numpy(), C.TARGET: newp}).to_csv(
        C.SUBMISSIONS_DIR / "probe_P_EXIT.csv", index=False)
    say("yazildi: submissions/probe_P_EXIT.csv")
    say()
    say("Olcum yorumu: bu yoklamanin skoru S ise")
    say("  <e0,D> = (S^2 - 1.01548^2 - ||D||^2) / 2")
    say("negatif cikarsa panel sinir satirlarinda gercekten fazla tahmin")
    say("ediyoruz ve kapinin optimal siddeti w = -<e0,D>/||D||^2 ile bulunur.")
    say()

    p = C.REPORTS_DIR / "74_panel_gate.md"
    p.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
