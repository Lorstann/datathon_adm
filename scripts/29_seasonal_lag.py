"""Satir duzeyinde gecen-yil lag'i (364 gun) ozellik olarak.

`hist_yoy_mean` varlik duzeyinde ve ufkun tamaminin gecen yilki ORTALAMASI.
Bu deney ufkun ICINDEKI sekli ekliyor: gecen yil ayni hafta ne oldu.

KAPSAMA ASIMETRISI (karari okurken sart): aranan tarih = ufuk gunu - 364.
  fold A (ufuk Nis-Tem 2025)  -> Nis-Tem 2024, panel oncesi, kapsama ~0
  fold C (ufuk Eki-Oca)       -> Eki 2024-Oca 2025, cogu panel oncesi
  fold B (ufuk Ara-Mar)       -> Ara 2024-Mar 2025, kismi
  GONDERIM (ufuk Nis-Tem 2026) -> Nis-Tem 2025, TAM
Yani CV bu ozelligi hak ettiginden dusuk gosterir; karar fold B'ye ve kapsama
sayilarina bakarak verilmeli, uc fold ortalamasina degil.

Kullanim: python scripts/29_seasonal_lag.py
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
from src.models.level_shape import fit_shape_model, predict_level_shape
from src.models.multiorigin import LEVEL_COL, build_training_frame, mask_cold
from src.models.zero_gate import dead_probability, shrink_level
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

_cv = import_module("20_multiorigin_cv")

GATE_K = 0.75
LAG_COLS = ["y_lag364", "y_lag364_win"]


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    records = []
    lines = ["# Satir duzeyinde gecen-yil lag'i\n", f"Uretim: {pd.Timestamp.now()}\n"]
    lines.append(
        "\nKapsama asimetrisi: gonderimde tam, fold A/C'de nerdeyse yok. "
        "Karar fold B'ye gore.\n"
    )

    for fold in C.FOLDS:
        split = make_fold(train, fold, profile, seed=C.SEED)
        p_table = dead_probability(split.train, fold.origin)

        lines.append(f"\n## Fold {fold.name}\n\n")
        lines.append("| variant | kapsama(valid) | warm | cold | blend |\n")
        lines.append("| --- | --- | --- | --- | --- |\n")

        for use_lag in (False, True):
            t0 = time.time()
            tr_f = build_training_frame(
                split.train,
                end_cap=fold.origin,
                static=static,
                weather=weather,
                max_rows=_cv.MAX_ROWS,
                seasonal_lag=use_lag,
            )
            tr_f = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED)
            va_f = _cv.valid_frame(split, static, weather, seasonal_lag=use_lag)
            va_f = va_f.loc[va_f[C.TARGET].notna()]

            cov = float(va_f["y_lag364_win"].notna().mean()) if use_lag else float("nan")
            cov_tr = float(tr_f["y_lag364_win"].notna().mean()) if use_lag else float("nan")

            y = va_f[C.TARGET].to_numpy()
            seg = split.segment.reindex(va_f.index)
            cold_s = split.is_cold.reindex(va_f.index)
            cold = cold_s.to_numpy()
            p_row = va_f[C.LOCATION].map(p_table["by_lokasyon"]).fillna(p_table["global"])
            p_row = np.where(cold, p_row.to_numpy(), 0.0)

            ytr = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
            lvl_tr = tr_f[LEVEL_COL].to_numpy()
            lvl_va = va_f[LEVEL_COL].to_numpy()

            Xtr, cat = model_matrix(tr_f)
            Xva, _ = model_matrix(va_f)
            Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
            Xva = Xva[Xtr.columns]
            hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]

            m = fit_shape_model(
                Xtr, ytr, lvl_tr, cat, hist_cols, seed=C.SEED + 3, n_trees=600
            )
            pred = shrink_level(
                predict_level_shape(m, Xva, lvl_va), p_row, strength=GATE_K
            )
            met = segment_report(y, pred, seg, is_cold=cold_s)
            name = "lag364" if use_lag else "base"
            records.append({"model": name, "fold": fold.name, **met})
            lines.append(
                "| {} | {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                    name,
                    "-" if not use_lag else f"{cov:.1%} (egitim {cov_tr:.1%})",
                    met["rmsle_warm"],
                    met["rmsle_cold"],
                    met["rmsle_blend"],
                )
            )
            print(
                fold.name, name, round(met["rmsle_blend"], 4),
                f"cov={cov:.3f}" if use_lag else "", f"{time.time()-t0:.0f}s",
                flush=True,
            )
            del tr_f, va_f, Xtr, Xva

    df = pd.DataFrame(records)
    lines.append("\n## Ozet (rmsle_blend)\n\n| variant | mean | std | warm |\n")
    lines.append("| --- | --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                model, s["mean"], s["std"], g["rmsle_warm"].mean()
            )
        )
    piv = df.pivot(index="fold", columns="model", values="rmsle_blend")
    lines.append(
        f"\nFold bazinda fark (lag364 - base): {(piv['lag364']-piv['base']).round(4).to_dict()}\n"
    )
    (C.REPORTS_DIR / "29_seasonal_lag.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().to_string())


if __name__ == "__main__":
    main()
