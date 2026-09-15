"""31 Mart kurali uyumlu hava verisi: maliyet olcumu.

Host (2026-08-25): "tahminlerin 31 Mart 2026 itibariyla uretildigi kabul
edileceginden, Nisan-Temmuz 2026 donemine ait GERCEKLESMIS hava durumu
verilerinin kullanilmasi uygun olmayacaktir." Bizim boru hattimiz su ana
kadar Open-Meteo'dan test donemi (2026-04-01 - 2026-07-31) GERCEK sicakligi
cekip kullaniyordu -- forum'da "ilk 100'un neredeyse tamami" ayni seyi
yapiyor deniyor (~0.02 avantaj, rank 11 OzanM'in olcumu).

Bu script fold A'da (mevsim fold, origin 2025-03-31, valid 2025-04-01/07-31 --
gercek Nis-Tem donemi) iki hava kaynagini karsilastirir:
  - realized: mevcut yol, gercek gozlenmis sicaklik (kural ihlali)
  - compliant: origin'e kadarki veriden yil-gunu iklim ortalamasi
    (`weather_features(..., normals_cutoff=origin)`), test donemi icin
    gercek deger hic kullanilmiyor.

Kullanim: python scripts/56_compliant_weather.py
"""

from __future__ import annotations

import sys
import time
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import build_panel, entity_static, load_test, load_train
from src.external.weather import CACHE_PATH, load_normals, load_weather, weather_features
from src.features.build import model_matrix
from src.models.level_shape import fit_shape_model, predict_level_shape
from src.models.multiorigin import LEVEL_COL, build_training_frame, mask_cold
from src.models.zero_gate import dead_probability, shrink_level
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report
from src.validation.split import make_fold

_cv = import_module("20_multiorigin_cv")
GATE_K = 0.75
N_TREES = 400
MAX_ROWS = 500_000
SEED = C.SEED + 3


def run_fold(fold, train, static, profile, weather_raw, normals_table, *, compliant: bool) -> dict:
    origin = fold.origin
    weather = weather_features(
        weather_raw, normals=compliant, normals_cutoff=origin if compliant else None,
        normals_table=normals_table if compliant else None,
    )
    split = make_fold(train, fold, profile, seed=C.SEED)
    p_table = dead_probability(split.train, origin)
    tr_f = build_training_frame(
        split.train, end_cap=origin, static=static, weather=weather,
        max_rows=MAX_ROWS,
    )
    tr_f = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED, entry_offsets=profile.entry_offsets)
    va_f = _cv.valid_frame(split, static, weather)
    va_f = va_f.loc[va_f[C.TARGET].notna()]

    y = va_f[C.TARGET].to_numpy()
    seg = split.segment.reindex(va_f.index)
    cold_s = split.is_cold.reindex(va_f.index)
    p_row = va_f[C.LOCATION].map(p_table["by_lokasyon"]).fillna(p_table["global"])
    p_row = np.where(cold_s.to_numpy(), p_row.to_numpy(), 0.0)

    ytr = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
    lvl_tr, lvl_va = tr_f[LEVEL_COL].to_numpy(), va_f[LEVEL_COL].to_numpy()
    Xtr, cat = model_matrix(tr_f)
    Xva, _ = model_matrix(va_f)
    Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
    Xva = Xva[Xtr.columns]
    hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]

    m = fit_shape_model(Xtr, ytr, lvl_tr, cat, hist_cols, seed=SEED, n_trees=N_TREES)
    pred = predict_level_shape(m, Xva, lvl_va)
    pred = shrink_level(pred, p_row, strength=GATE_K)
    return segment_report(y, pred, seg, is_cold=cold_s)


def main() -> None:
    C.ensure_dirs()
    t0 = time.time()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather_raw = load_weather()
    normals_table = load_normals()
    print(f"iklim normali tablosu: {len(normals_table):,} satir", flush=True)

    fold_a = next(f for f in C.FOLDS if "A" in f.name or "mevsim" in f.name.lower())
    print(f"fold: {fold_a.name}, origin {fold_a.origin.date()}", flush=True)

    m_real = run_fold(fold_a, train, static, profile, weather_raw, normals_table, compliant=False)
    print(f"realized  (kural ihlali): warm={m_real['rmsle_warm']:.4f} "
          f"cold={m_real['rmsle_cold']:.4f} blend={m_real['rmsle_blend']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    m_comp = run_fold(fold_a, train, static, profile, weather_raw, normals_table, compliant=True)
    print(f"compliant (31 Mart)    : warm={m_comp['rmsle_warm']:.4f} "
          f"cold={m_comp['rmsle_cold']:.4f} blend={m_comp['rmsle_blend']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    delta = m_comp["rmsle_blend"] - m_real["rmsle_blend"]
    print(f"maliyet (compliant - realized): {delta:+.4f}", flush=True)

    lines = [
        "# 31 Mart kurali uyumu: maliyet olcumu\n",
        f"Uretim: {pd.Timestamp.now()}\n\n",
        f"Fold {fold_a.name}, origin {fold_a.origin.date()}, "
        f"n_trees={N_TREES}, max_rows={MAX_ROWS:,}\n\n",
        "| kaynak | warm | cold | blend |\n| --- | --- | --- | --- |\n",
        f"| realized (kural ihlali) | {m_real['rmsle_warm']:.4f} | "
        f"{m_real['rmsle_cold']:.4f} | {m_real['rmsle_blend']:.4f} |\n",
        f"| compliant (31 Mart) | {m_comp['rmsle_warm']:.4f} | "
        f"{m_comp['rmsle_cold']:.4f} | {m_comp['rmsle_blend']:.4f} |\n",
        f"\nMaliyet (compliant - realized): **{delta:+.4f}**\n",
    ]
    (C.REPORTS_DIR / "56_compliant_weather.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
