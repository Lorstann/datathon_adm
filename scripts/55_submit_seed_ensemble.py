"""Genis tohum ortalamasi gonderimi. python scripts/55_submit_seed_ensemble.py

`scripts/22_submit_multiorigin.py`'nin BIREBIR ayni boru hatti; tek fark
SEEDS 3 -> 7. Gerekce, yarisma forumunda 2.'nin yazdigi gozlem: birden fazla
tahminin ortalamasi, anlayista hicbir iyilesme olmadan guvenilir bicimde
0.005-0.015 kazandiriyor -- cunku SEED ortalamasi saf varyans azaltma, yonu
onceden belli (asla kotulesmez, beklenen deger degismez).

`reports/47_seeds_trees.md` s1/s3/s7 arasinda CV'de ±0.001 fark bulmustu,
yani 3 tohum CV'de zaten doymus GORUNUYORDU. Ama round5 (`reports/51`) ayni
donemde CV'nin buyuk yapisal degisiklikler icin LB'yi guvenilir tahmin
etmedigini gosterdi (deney 8: CV -0.0041, LB +0.0138). Tohum sayisi YAPISAL
bir degisiklik degil -- ayni model, ayni ozellik seti, sadece daha fazla
bagimsiz cekirdegin ortalamasi -- yani CV'nin bu eksende yaniltici olma
riski yapisal degisikliklerdeki kadar yuksek degil. Asagi yonlu riski yok:
en kotu ihtimalle CV'nin gordugu ±0.001'lik fark kadar degisir.

Cikti mevcut en iyiyi (1.06713) EZMEZ, ayri dosyaya yazar. Kaggle'a
gonderilip gonderilmeyecegine kullanici karar verir.
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
from src.data.load import (
    build_panel,
    entity_static,
    load_sample_submission,
    load_test,
    load_train,
)
from src.external.weather import CACHE_PATH, load_weather, weather_features
from src.features.build import assemble, model_matrix
from src.features.history import entity_history_features
from src.features.peer import peer_tables
from src.models.level_shape import (
    WARM_LEVEL_CHAIN,
    entity_level,
    fit_shape_model,
    predict_level_shape,
)
from src.models.multiorigin import (
    LEVEL_COL,
    build_training_frame,
    frozen_history,
    mask_cold,
    training_origins,
)
from src.models.zero_gate import dead_probability, shrink_level
from src.validation.folds import build_cold_profile

N_TREES = 600
MAX_ROWS = 1_800_000
SEEDS = tuple(C.SEED + 3 + 10 * i for i in range(7))  # 3 -> 7 tohum
GATE_K = 0.75
OUT = C.SUBMISSIONS_DIR / "submission_seed7.csv"


def main() -> None:
    C.ensure_dirs()
    t0 = time.time()
    train, test, sample = load_train(), load_test(), load_sample_submission()
    static = entity_static(build_panel(train, test))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None
    origin = C.TRAIN_END

    origins = training_origins(origin, max_origins=8)
    print("origins:", [str(o.date()) for o in origins], flush=True)
    from src.models.multiorigin import origin_frame
    frames = []
    for o in origins:
        to = time.time()
        f = origin_frame(train, o, static=static, weather=weather, end_cap=origin)
        if f is not None and len(f):
            f["_origin"] = o
            frames.append(f)
        print(f"  origin {o.date()} -> {len(f) if f is not None else 0:,} satir "
              f"({time.time()-to:.0f}s, toplam {time.time()-t0:.0f}s)", flush=True)
    tr_f = pd.concat(frames, ignore_index=True)
    if len(tr_f) > MAX_ROWS:
        rng = np.random.default_rng(C.SEED)
        keep = rng.choice(len(tr_f), size=MAX_ROWS, replace=False)
        tr_f = tr_f.iloc[np.sort(keep)].reset_index(drop=True)
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    tr_f = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED, entry_offsets=profile.entry_offsets)
    print(f"egitim satiri: {len(tr_f):,}  ({time.time()-t0:.0f}s)", flush=True)

    hist = frozen_history(train, origin, weather)
    peer = peer_tables(train, origin)
    te_f = assemble(
        test, origin=origin, history=hist, weather=weather, static=static,
        peer=peer, include_history=True,
    )
    te_f[LEVEL_COL] = entity_level(
        hist, te_f[C.ENTITY], te_f["log_guc"].to_numpy(), warm_col=WARM_LEVEL_CHAIN
    )

    y = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
    lvl_tr = tr_f[LEVEL_COL].to_numpy()
    lvl_te = te_f[LEVEL_COL].to_numpy()
    Xtr, cat = model_matrix(tr_f)
    Xte, _ = model_matrix(te_f)
    Xtr, Xte = Xtr.align(Xte, join="outer", axis=1, fill_value=np.nan)
    Xte = Xte[Xtr.columns]
    hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]

    acc = np.zeros(len(Xte))
    for s in SEEDS:
        m = fit_shape_model(Xtr, y, lvl_tr, cat, hist_cols, seed=s, n_trees=N_TREES)
        acc += np.log1p(predict_level_shape(m, Xte, lvl_te))
        print(f"  seed {s} bitti ({time.time()-t0:.0f}s)", flush=True)
    pred = np.clip(np.expm1(acc / len(SEEDS)), 0.0, None)

    is_cold = ~te_f[C.ENTITY].isin(set(train[C.ENTITY].unique())).to_numpy()
    p_table = dead_probability(train, origin)
    p_row = te_f[C.LOCATION].map(p_table["by_lokasyon"]).fillna(p_table["global"])
    p_row = np.where(is_cold, p_row.to_numpy(), 0.0)
    pred = shrink_level(pred, p_row, strength=GATE_K)

    out = sample.copy()
    out[C.TARGET] = pred
    assert out["id"].equals(test["id"]), "id sirasi sample_submission ile uyusmuyor"
    out.to_csv(OUT, index=False)
    print(f"yazildi {OUT}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
