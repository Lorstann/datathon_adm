"""Gonderim dosyasi on kontrolu: id butunlugu, NaN, isaret, dagilim.

Gonderim hakki kit (bugun 1, yarin 3). Bozuk bir dosya yalnizca skoru degil
olcumu de yakar -- LB cebiri bozuk dosyadan gelen sayiyi sessizce yanlis
yorumlar. Bu yuzden her gonderimden once mekanik kontrol.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import load_sample_submission, load_test


def main() -> None:
    path = Path(sys.argv[1])
    test = load_test()
    sample = load_sample_submission()
    sub = pd.read_csv(path, dtype={"id": "string"})

    print(f"dosya: {path}")
    print(f"satir: {len(sub):,}  (test {len(test):,}, sample {len(sample):,})")
    print(f"kolonlar: {list(sub.columns)}  (sample {list(sample.columns)})")

    ok = True
    if list(sub.columns) != list(sample.columns):
        print("  HATA: kolon adlari/sirasi sample ile ayni degil")
        ok = False
    if len(sub) != len(sample):
        print("  HATA: satir sayisi sample ile ayni degil")
        ok = False

    ids_sub = set(sub["id"])
    ids_sample = set(sample["id"])
    if ids_sub != ids_sample:
        print(f"  HATA: id kumesi farkli "
              f"(eksik {len(ids_sample - ids_sub):,}, "
              f"fazla {len(ids_sub - ids_sample):,})")
        ok = False
    if sub["id"].duplicated().any():
        print(f"  HATA: {int(sub['id'].duplicated().sum()):,} tekrar eden id")
        ok = False

    v = sub[C.TARGET]
    n_nan = int(v.isna().sum())
    n_neg = int((v < 0).sum())
    n_inf = int(np.isinf(v.to_numpy()).sum())
    if n_nan or n_neg or n_inf:
        print(f"  HATA: NaN {n_nan}, negatif {n_neg}, inf {n_inf}")
        ok = False

    lp = np.log1p(v.clip(lower=0))
    print(f"tuketim: min {v.min():,.2f}  medyan {v.median():,.2f}  "
          f"maks {v.max():,.2f}  ort {v.mean():,.2f}")
    print(f"log1p:   ort {lp.mean():.4f}  std {lp.std():.4f}  "
          f"tam sifir {int((v <= 0).sum()):,}")
    print("SONUC:", "GECERLI" if ok else "GECERSIZ")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
