"""Ozellik hijyeni: `year`, origin-sabiti akran kolonlari ve tek-kolon DOW ofseti.

`reports/54_adversarial_validation.md` train/test ayirt etme AUC'sini 1.0000
buldu ve en cok ayirt eden ozellikleri listeledi -- ama sonuc bolumu yazilmadi
ve hicbir kolon cikarilmadi. Listenin basi:

    peer_global_z   48.080.342   (sonrakinin 14 kati)
    cos_month        3.470.525
    year             3.082.500
    month              507.005
    peer_zero_guc      215.328

Iki yapisal sorun:

1. `year` bir ozellik. Egitimde 2026 yalnizca Oca-Mar ile geliyor, test'in
   tamami 2026 Nis-Tem. `year >= 2026` esigi test satirlarinin HEPSINI
   yalnizca kis verisiyle egitilmis bir dala yolluyor.
2. `peer_global_z` origin basina TEK bir sayi. Yani ozellik degil, origin
   kimligi. Agac origin bazli artik ofsetlerini ezberleyebilir; test'te deger
   yeni bir sabit ve en yakin origin'e ekstrapole olur. `peer_zero_guc`,
   `peer_guc_z`, `peer_guc_wknd_z` de origin basina bir avuc deger.

Ucuncu deneme: `hist_dow_rel_0..6` yedi ayri kolon; agacin once `dow`'a sonra
dogru kolona bolmesi gerekiyor. Satirin kendi haftagunune karsilik gelen
degeri TEK kolon olarak vermek ayni bilgiyi tek bolmede erisilir kiliyor.
Ham olcum (reports/65_shape_headroom.md) dort pencerede de -0.005..-0.007.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import build_panel, entity_static, load_test, load_train
from src.external.weather import CACHE_PATH, load_weather, weather_features
from src.features.build import model_matrix
from src.models.level_shape import fit_shape_model, predict_level_shape
from src.models.multiorigin import LEVEL_COL, build_training_frame, mask_cold
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report
from src.validation.split import make_fold

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module

_mo = import_module("20_multiorigin_cv")
valid_frame = _mo.valid_frame

MAX_ROWS = 1_200_000
N_TREES = 600
SEED = C.SEED + 3
PEER_COLS = ["peer_global_z", "peer_guc_z", "peer_guc_log", "peer_lok_z",
             "peer_guc_wknd_z", "peer_zero_guc"]

# Fold A disarida: origin 2025-03-31'de yalnizca 3 ay gecmis var, cok-origin
# cercevesi tek origin'e dusuyor ve egitim 88k satira iniyor. reports/22 bunu
# "A'nin veri acligi, yontemin degil" diye kaydetti; gonderim rejimini temsil
# etmiyor ve tur suresini ikiye katliyor.
FOLDS = [f for f in C.FOLDS if f.name != "A_mevsim"]


def add_dow_off(f: pd.DataFrame) -> pd.DataFrame:
    """Satirin kendi haftagunune karsilik gelen trafo DOW sapmasi, tek kolon."""
    cols = [f"hist_dow_rel_{k}" for k in range(7)]
    if not all(c in f.columns for c in cols):
        f["dow_off"] = np.nan
        return f
    vals = f[cols].to_numpy(dtype="float64")
    dow = f["dow"].to_numpy(dtype="int64")
    f["dow_off"] = vals[np.arange(len(f)), dow]
    return f


VARIANTS = {
    "baseline": [],
    "year_yok": ["year"],
    "year+peer_global_yok": ["year", "peer_global_z"],
    "year+tum_peer_yok": ["year", *PEER_COLS],
    "dow_off": [],  # kolon eklenir, cikarilmaz
    "year_yok+dow_off": ["year"],
}
WITH_DOW = {"dow_off", "year_yok+dow_off"}


def run_fold(fold, train, static, weather, profile) -> dict[str, dict]:
    t0 = time.time()
    split = make_fold(train, fold, profile, seed=C.SEED)
    tr_f = build_training_frame(
        split.train, end_cap=fold.origin, static=static, weather=weather,
        origins=None, max_rows=MAX_ROWS,
    )
    tr_f = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED)
    va_f = valid_frame(split, static, weather)
    va_f = va_f.loc[va_f[C.TARGET].notna()]
    tr_f, va_f = add_dow_off(tr_f), add_dow_off(va_f)
    print(f"  {fold.name}: egitim {len(tr_f):,} dogrulama {len(va_f):,} "
          f"({time.time()-t0:.0f}s)")

    ytr = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
    lvl_tr = tr_f[LEVEL_COL].to_numpy()
    lvl_va = va_f[LEVEL_COL].to_numpy()
    Xtr_all, cat = model_matrix(tr_f)
    Xva_all, _ = model_matrix(va_f)
    Xtr_all, Xva_all = Xtr_all.align(Xva_all, join="outer", axis=1, fill_value=np.nan)
    Xva_all = Xva_all[Xtr_all.columns]

    seg = split.segment.reindex(va_f.index)
    cold = split.is_cold.reindex(va_f.index)

    out = {}
    for name, drop in VARIANTS.items():
        cols = [c for c in Xtr_all.columns if c not in drop]
        if name not in WITH_DOW:
            cols = [c for c in cols if c != "dow_off"]
        Xtr, Xva = Xtr_all[cols], Xva_all[cols]
        cats = [c for c in cat if c in cols]
        hist_cols = [c for c in cols if c.startswith("hist_")]
        m = fit_shape_model(Xtr, ytr, lvl_tr, cats, hist_cols,
                            seed=SEED, n_trees=N_TREES)
        pred = predict_level_shape(m, Xva, lvl_va)
        rep = segment_report(va_f[C.TARGET], pred, seg, is_cold=cold)
        out[name] = rep
        print(f"    {name:<24} blend {rep['rmsle_blend']:.4f} "
              f"warm {rep['rmsle_warm']:.4f} cold {rep['rmsle_cold']:.4f} "
              f"({time.time()-t0:.0f}s)")
    return out


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    recs = []
    for fold in FOLDS:
        res = run_fold(fold, train, static, weather, profile)
        for name, rep in res.items():
            recs.append({"variant": name, "fold": fold.name, **rep})

    df = pd.DataFrame(recs)
    piv = df.pivot(index="variant", columns="fold", values="rmsle_blend")
    piv["mean"] = piv.mean(axis=1)
    base = piv.loc["baseline", "mean"]
    piv["delta"] = piv["mean"] - base

    lines = ["# Ozellik hijyeni: year, akran sabitleri, tek-kolon DOW ofseti\n",
             f"\nUretim: {pd.Timestamp.now()}\n",
             f"\nFold A disarida (veri acligi). {N_TREES} agac, tek tohum, "
             f"{MAX_ROWS:,} satir.\n\n## rmsle_blend\n\n"]
    lines.append("```\n" + piv.round(4).to_string() + "\n```\n")
    for metric in ("rmsle_warm", "rmsle_cold"):
        p = df.pivot(index="variant", columns="fold", values=metric)
        p["mean"] = p.mean(axis=1)
        p["delta"] = p["mean"] - p.loc["baseline", "mean"]
        lines.append(f"\n## {metric}\n\n```\n" + p.round(4).to_string() + "\n```\n")

    path = C.REPORTS_DIR / "66_feature_hygiene.md"
    path.write_text("".join(lines), encoding="utf-8")
    print("\n" + piv.round(4).to_string())
    print(f"\nyazildi: {path}")


if __name__ == "__main__":
    main()
