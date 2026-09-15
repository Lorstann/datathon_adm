"""Iki asamali mimari: sifirsiz regresor + ayri sifir olasiligi.

`reports/02_cold_start_ceiling.md` karar 3 aynen sunu diyordu:

    "regresor yalnizca sifir olmayan satirlarda egitilecek, sifir olasiligi
     ayri bir siniflandiriciyla modellenecek"

Bu hic uygulanmadi. Model bugun hala sifirlar DAHIL tum satirlarda egitiliyor;
sifirlar sadece cold tarafinda `shrink_level` ile ele aliniyor. Oysa sifirlar
kare hatanin %55-60'i (reports/23_error_decomp.md).

Mimari:
    tahmin_log = (1 - k*p) * (pozitif_seviye + artik)

  - `pozitif_seviye`: yalniz sifir olmayan gunlerin ortalamasi
    (`hist_pos_mean_*`). Sifirli seviye kullanilirsa sifir kutlesi iki kez
    sayilir, cunku `hist_mean` zaten ~ (1-p_gecmis) * pozitif_seviye.
    Olculdu: sifiri olan trafolarda fark +1.444 log birimi.
  - `artik`: sekil modeli, YALNIZ sifir olmayan satirlarda egitilir.
  - `p`: warm icin son 28 gunun sifir orani (EDA: sifir durumu kalici,
    P(sifir|sifir)=0.976, ufuk oncesi/sonrasi korelasyon 0.938),
    cold icin lokasyon bazli olu-trafo olasiligi.

Kullanim: python scripts/50_two_stage.py
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
from src.external.weather import CACHE_PATH, load_weather, weather_features
from src.features.build import model_matrix
from src.features.history import entity_history_features
from src.models.level_shape import (
    POS_LEVEL_CHAIN,
    entity_level,
    fit_shape_model,
    predict_level_shape,
)
from src.models.multiorigin import LEVEL_COL, build_training_frame, mask_cold
from src.models.zero_gate import dead_probability, shrink_level
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

_cv = import_module("20_multiorigin_cv")
GATE_K = 0.75
TWO_STAGE_K = (0.75, 1.0)


def zero_prob(frame, cold_mask, p_dead):
    """Satir bazinda P(ufukta sifir). Warm: son 28 gunun sifir orani."""
    p = frame["hist_zero_28"] if "hist_zero_28" in frame.columns else None
    if p is None:
        p = frame.get("hist_zero_rate")
    p = pd.Series(np.asarray(p, dtype="float64"), index=frame.index)
    p = p.fillna(0.0).to_numpy()
    return np.clip(np.where(cold_mask, p_dead, p), 0.0, 1.0)


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    records = []
    lines = ["# Iki asamali mimari (reports/02 karar 3)\n", f"Uretim: {pd.Timestamp.now()}\n"]

    for fold in C.FOLDS:
        split = make_fold(train, fold, profile, seed=C.SEED)
        p_table = dead_probability(split.train, fold.origin)
        tr_f = build_training_frame(
            split.train, end_cap=fold.origin, static=static, weather=weather,
            max_rows=_cv.MAX_ROWS,
        )
        tr_f = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED,
                         entry_offsets=profile.entry_offsets)
        va_f = _cv.valid_frame(split, static, weather)
        va_f = va_f.loc[va_f[C.TARGET].notna()]
        hist_va = entity_history_features(split.train, fold.origin)
        hist_by_origin = {
            o: entity_history_features(split.train, o) for o in tr_f["_origin"].unique()
        }

        y = va_f[C.TARGET].to_numpy()
        seg = split.segment.reindex(va_f.index)
        cold_s = split.is_cold.reindex(va_f.index)
        cold = cold_s.to_numpy()
        p_dead_va = va_f[C.LOCATION].map(p_table["by_lokasyon"]).fillna(
            p_table["global"]
        ).to_numpy()
        p_gate = np.where(cold, p_dead_va, 0.0)

        ytr = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
        Xtr, cat = model_matrix(tr_f)
        Xva, _ = model_matrix(va_f)
        Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
        Xva = Xva[Xtr.columns]
        hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]

        lines.append(f"\n## Fold {fold.name}\n\n| variant | warm | cold | blend |\n")
        lines.append("| --- | --- | --- | --- |\n")

        def report(name, pred):
            met = segment_report(y, pred, seg, is_cold=cold_s)
            records.append({"model": name, "fold": fold.name, **met})
            lines.append("| {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                name, met["rmsle_warm"], met["rmsle_cold"], met["rmsle_blend"]))
            print(fold.name, name, round(met["rmsle_blend"], 4),
                  "cold", round(met["rmsle_cold"], 4), flush=True)

        # --- mevcut: tum satirlar, sifirli seviye, yalniz cold kapisi
        t0 = time.time()
        m = fit_shape_model(Xtr, ytr, tr_f[LEVEL_COL].to_numpy(), cat, hist_cols,
                            seed=C.SEED + 3, n_trees=600)
        base = shrink_level(
            predict_level_shape(m, Xva, va_f[LEVEL_COL].to_numpy()), p_gate,
            strength=GATE_K,
        )
        report("mevcut", base)
        print(f"  {time.time()-t0:.0f}s", flush=True)

        # --- iki asamali: sifirsiz seviye + sifirsiz egitim + (1-k*p)
        is_masked = tr_f["hist_n"].isna().to_numpy()
        parts = [
            pd.Series(
                entity_level(hist_by_origin[o], g[C.ENTITY],
                             g["log_guc"].to_numpy(), warm_col=POS_LEVEL_CHAIN),
                index=g.index,
            )
            for o, g in tr_f.groupby("_origin", sort=False)
        ]
        lvl_pos_tr = pd.concat(parts).reindex(tr_f.index).to_numpy()
        lvl_pos_tr = np.where(is_masked, tr_f["_cold_level"].to_numpy(), lvl_pos_tr)
        lvl_pos_va = entity_level(hist_va, va_f[C.ENTITY],
                                  va_f["log_guc"].to_numpy(), warm_col=POS_LEVEL_CHAIN)

        nz = (tr_f[C.TARGET].to_numpy() > 0)
        t0 = time.time()
        m2 = fit_shape_model(Xtr[nz], ytr[nz], lvl_pos_tr[nz], cat, hist_cols,
                             seed=C.SEED + 3, n_trees=600)
        lp2 = np.log1p(np.clip(predict_level_shape(m2, Xva, lvl_pos_va), 0, None))
        print(f"  sifirsiz egitim {int(nz.sum()):,} satir, {time.time()-t0:.0f}s", flush=True)

        p_row = zero_prob(va_f, cold, p_dead_va)
        two = {}
        for k in TWO_STAGE_K:
            pred = np.clip(np.expm1(lp2 * np.clip(1.0 - k * p_row, 0.0, 1.0)), 0.0, None)
            two[k] = pred
            report(f"iki_asamali_k{k}", pred)

        # Melez: warm satirlarda iki asamali (p gercek gecmisten), cold
        # satirlarda mevcut (orada p yalnizca zayif bir lokasyon onseli).
        for k in TWO_STAGE_K:
            report(f"hibrit_k{k}", np.where(cold, base, two[k]))

        del tr_f, va_f, Xtr, Xva

    df = pd.DataFrame(records)
    lines.append("\n## Ozet\n\n| variant | mean | std | warm | cold |\n")
    lines.append("| --- | --- | --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            model, s["mean"], s["std"], g["rmsle_warm"].mean(), g["rmsle_cold"].mean()))
    piv = df.pivot(index="fold", columns="model", values="rmsle_blend")
    for c in piv.columns:
        lines.append(f"\n- {c}: {(piv[c] - piv['mevcut']).round(4).to_dict()}")
    (C.REPORTS_DIR / "50_two_stage.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().to_string())


if __name__ == "__main__":
    main()
