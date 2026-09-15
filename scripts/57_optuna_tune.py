"""Ilk sistematik hiperparametre taramasi (Optuna).

`experiments.csv`'de round1'den beri LGBM_PARAMS sabit
(learning_rate=0.05, num_leaves=63, min_child_samples=80); tek ablasyon
"deep_leaves" tek seferlik denemeydi. Sistematik arama hic yapilmadi --
bu script onu yapiyor.

Arama fold A'da (mevsim fold, en ilgili) kucultulmus max_rows/n_trees ile
hizli calisir; en iyi 3 aday sonunda 3 fold'un tamaminda tam boyutta
dogrulanir (M5 kurali: ortalama VE std).

Kullanim: python scripts/57_optuna_tune.py
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
import optuna
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
SEARCH_TREES = 350
SEARCH_MAX_ROWS = 500_000
FULL_TREES = 600
FULL_MAX_ROWS = 1_200_000
SEED = C.SEED + 3
N_TRIALS = 25

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _prep_fold(fold, train, static, profile, weather, max_rows):
    origin = fold.origin
    split = make_fold(train, fold, profile, seed=C.SEED)
    p_table = dead_probability(split.train, origin)
    tr_f = build_training_frame(
        split.train, end_cap=origin, static=static, weather=weather, max_rows=max_rows,
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
    return dict(Xtr=Xtr, ytr=ytr, lvl_tr=lvl_tr, cat=cat, hist_cols=hist_cols,
                Xva=Xva, lvl_va=lvl_va, y=y, seg=seg, cold_s=cold_s, p_row=p_row)


def _eval(prepped, params, n_trees, seed=SEED):
    m = fit_shape_model(
        prepped["Xtr"], prepped["ytr"], prepped["lvl_tr"], prepped["cat"],
        prepped["hist_cols"], seed=seed, n_trees=n_trees, params_override=params,
    )
    pred = predict_level_shape(m, prepped["Xva"], prepped["lvl_va"])
    pred = shrink_level(pred, prepped["p_row"], strength=GATE_K)
    return segment_report(prepped["y"], pred, prepped["seg"], is_cold=prepped["cold_s"])


def main() -> None:
    C.ensure_dirs()
    t0 = time.time()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    fold_a = next(f for f in C.FOLDS if "A" in f.name)
    prepped_a = _prep_fold(fold_a, train, static, profile, weather, SEARCH_MAX_ROWS)
    print(f"arama fold {fold_a.name} hazir ({time.time()-t0:.0f}s)", flush=True)

    baseline = _eval(prepped_a, None, SEARCH_TREES)
    print(f"baseline (varsayilan params, {SEARCH_TREES} agac): "
          f"blend={baseline['rmsle_blend']:.4f}", flush=True)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
        }
        met = _eval(prepped_a, params, SEARCH_TREES)
        return met["rmsle_blend"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.enqueue_trial({  # mevcut varsayilanlar ilk deneme olsun
        "num_leaves": 63, "min_child_samples": 80, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
        "reg_alpha": 0.0 + 1e-3, "max_depth": 12,
    })
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
                   callbacks=[lambda s, t: print(
                       f"  trial {t.number}: blend={t.value:.4f} "
                       f"(en iyi {s.best_value:.4f}) params={t.params} "
                       f"({time.time()-t0:.0f}s)", flush=True)])

    print(f"\narama bitti ({time.time()-t0:.0f}s). En iyi 3 aday tum fold'larda "
          f"tam boyutta dogrulaniyor.\n", flush=True)

    top = sorted(study.trials, key=lambda t: t.value)[:3]
    report_path = C.REPORTS_DIR / "57_optuna_tune.md"
    report_path.write_text(
        "# Optuna hiperparametre taramasi (arama fazi, kismi)\n\n"
        f"Uretim: {pd.Timestamp.now()}\n\nEn iyi 3 aday:\n\n"
        + "\n".join(f"- top{i+1} (arama blend={t.value:.4f}): {t.params}"
                     for i, t in enumerate(top))
        + "\n\nTam 3-fold dogrulama asagida devam ediyor; bu dosya her adimda guncellenir.\n",
        encoding="utf-8",
    )
    records = []
    prepped_full = {}
    for fold in C.FOLDS:
        prepped_full[fold.name] = _prep_fold(fold, train, static, profile, weather, FULL_MAX_ROWS)
        print(f"tam fold {fold.name} hazir ({time.time()-t0:.0f}s)", flush=True)

    def _write_partial(records):
        pd.DataFrame(records).to_csv(C.REPORTS_DIR / "57_optuna_partial.csv", index=False)

    # varsayilan taban da tam boyutta olcusun
    for fold in C.FOLDS:
        met = _eval(prepped_full[fold.name], None, FULL_TREES)
        records.append({"config": "baseline_default", "fold": fold.name, **met})
        print(f"baseline_default {fold.name}: blend={met['rmsle_blend']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        _write_partial(records)

    for i, trial in enumerate(top):
        name = f"optuna_top{i+1}"
        for fold in C.FOLDS:
            met = _eval(prepped_full[fold.name], trial.params, FULL_TREES)
            records.append({"config": name, "fold": fold.name, **met})
            print(f"{name} {fold.name}: blend={met['rmsle_blend']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            _write_partial(records)

    df = pd.DataFrame(records)
    lines = [
        "# Optuna hiperparametre taramasi\n",
        f"Uretim: {pd.Timestamp.now()}\n\n",
        f"Arama: fold A, {N_TRIALS} deneme, {SEARCH_TREES} agac, "
        f"{SEARCH_MAX_ROWS:,} satir. Dogrulama: 3 fold, {FULL_TREES} agac, "
        f"{FULL_MAX_ROWS:,} satir.\n\n",
        "## Ozet (rmsle_blend, 3 fold ortalama/std)\n\n",
        "| config | mean | std |\n| --- | --- | --- |\n",
    ]
    for cfg, g in df.groupby("config"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append(f"| {cfg} | {s['mean']:.4f} | {s['std']:.4f} |\n")
    lines.append("\n## En iyi 3 adayin parametreleri\n\n")
    for i, trial in enumerate(top):
        lines.append(f"- optuna_top{i+1} (arama blend={trial.value:.4f}): {trial.params}\n")
    (C.REPORTS_DIR / "57_optuna_tune.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("config")["rmsle_blend"].mean().to_string())
    print(f"yazildi: reports/57_optuna_tune.md ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
