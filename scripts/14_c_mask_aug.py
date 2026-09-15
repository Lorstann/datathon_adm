"""C cold iyilestirme: yuksek hist-mask + no-hist augment.

Varyantlar (hep level_shape C iskeleti):
  baseline     — mask 0.30 (mevcut)
  mask50       — mask 0.50
  mask70       — mask 0.70
  augment      — mask 0.30 + egitime hist-dusurulmus kopya (cold seviye)
  mask50_aug   — mask 0.50 + augment
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import config as C
from src.data.load import build_panel, entity_static, load_test, load_train
from src.external.weather import CACHE_PATH, load_weather, weather_features
from src.models.boosting import fit_lightgbm
from src.models.level_shape import entity_level, predict_level_shape
from src.models.pipeline import fold_frames, mask_history_entities, xy
from src.models.router import drop_history_cols
from src.tracking import log_experiment
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold


def cold_level_vec(hist: pd.DataFrame, log_guc: np.ndarray) -> np.ndarray:
    z = (
        float(hist["hist_mean_z"].median())
        if "hist_mean_z" in hist.columns and hist["hist_mean_z"].notna().any()
        else 0.0
    )
    return z + np.asarray(log_guc, dtype="float64")


def fit_c_variant(
    Xtr,
    ytr,
    logg_tr,
    entities,
    hist,
    cat,
    *,
    mask_rate: float,
    augment: bool,
    seed: int,
    n_trees: int = 300,
):
    X_m = mask_history_entities(Xtr, entities, mask_rate, seed)
    level = entity_level(hist, entities, logg_tr)
    resid = ytr - level

    if augment:
        X_cold = drop_history_cols(Xtr)
        # hist kolonlarini NaN ile geri ekle (ayni sema)
        for c in X_m.columns:
            if c.startswith("hist_") and c not in X_cold.columns:
                X_cold[c] = np.nan
        X_cold = X_cold.reindex(columns=X_m.columns)
        level_c = cold_level_vec(hist, logg_tr)
        resid_c = ytr - level_c
        X_fit = pd.concat([X_m, X_cold], axis=0, ignore_index=True)
        y_fit = np.concatenate([resid, resid_c])
    else:
        X_fit = X_m
        y_fit = resid

    hist_cols = [c for c in X_fit.columns if c.startswith("hist_")]
    # fit_lightgbm maskelemesin
    model = fit_lightgbm(
        X_fit,
        y_fit,
        cat,
        seed=seed,
        target="log1p",
        mask_history_rate=0.0,
        history_cols=hist_cols,
        n_trees=n_trees,
    )
    return model


def eval_variant(split, static, weather, *, mask_rate: float, augment: bool, tag: str):
    train_f, valid_f, hist = fold_frames(split, static, weather)
    Xtr, ytr, cat, logg_tr, tr_use = xy(train_f)
    Xva, _, _, logg_va, va_use = xy(valid_f, dropna_target=True)
    Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
    Xva = Xva[Xtr.columns]

    model = fit_c_variant(
        Xtr,
        ytr,
        logg_tr,
        tr_use[C.ENTITY],
        hist,
        cat,
        mask_rate=mask_rate,
        augment=augment,
        seed=C.SEED + 3,
    )
    level_va = entity_level(hist, va_use[C.ENTITY], logg_va)
    pred = predict_level_shape(model, Xva, level_va)
    m = segment_report(va_use[C.TARGET], pred, split.segment, is_cold=split.is_cold)
    return m


def main() -> None:
    C.ensure_dirs()
    assert CACHE_PATH.exists()
    weather = weather_features(load_weather())
    train = load_train()
    test = load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))

    variants = [
        ("baseline", 0.30, False),
        ("mask50", 0.50, False),
        ("mask70", 0.70, False),
        ("augment", 0.30, True),
        ("mask50_aug", 0.50, True),
    ]

    rows = []
    lines = [
        "# C hist-mask / augment cold tune\n",
        f"Uretim: {pd.Timestamp.now()}\n",
    ]

    for fold in C.FOLDS:
        print(f"=== {fold.name} ===")
        split = make_fold(train, fold, profile, seed=C.SEED)
        lines.append(f"\n## Fold {fold.name}\n")
        lines.append("| variant | warm | cold | blend |\n| --- | --- | --- | --- |\n")
        for tag, rate, aug in variants:
            m = eval_variant(split, static, weather, mask_rate=rate, augment=aug, tag=tag)
            rows.append({"model": tag, "fold": fold.name, **m})
            log_experiment(
                exp_id=f"cmask_{tag}",
                model=tag,
                target="log1p",
                feature_set="full_weather",
                fold=fold.name,
                metrics=m,
                notes=f"mask={rate} aug={aug}",
            )
            lines.append(
                f"| {tag} | {m.get('rmsle_warm', float('nan')):.4f} | "
                f"{m.get('rmsle_cold', float('nan')):.4f} | {m.get('rmsle_blend', float('nan')):.4f} |\n"
            )
            print(
                f"  {tag}: blend={m['rmsle_blend']:.4f} warm={m['rmsle_warm']:.4f} cold={m['rmsle_cold']:.4f}"
            )

    df = pd.DataFrame(rows)
    lines.append("\n## Ozet\n| variant | blend | warm | cold |\n| --- | --- | --- | --- |\n")
    summary = {}
    for tag, g in df.groupby("model"):
        b = summarize_folds(g, "rmsle_blend")["mean"]
        w = summarize_folds(g, "rmsle_warm")["mean"]
        c = summarize_folds(g, "rmsle_cold")["mean"]
        summary[tag] = {"blend": float(b), "warm": float(w), "cold": float(c)}
        lines.append(f"| {tag} | {b:.4f} | {w:.4f} | {c:.4f} |\n")

    base = summary["baseline"]
    best = "baseline"
    best_score = base["blend"]
    for tag, s in summary.items():
        if tag == "baseline":
            continue
        # kabul: blend <= base-0.01 ve cold kotulesmesin, VEYA cold-0.02 ve blend ~same
        ok = (s["blend"] <= base["blend"] - 0.01 and s["cold"] <= base["cold"] + 0.01) or (
            s["cold"] <= base["cold"] - 0.02 and s["blend"] <= base["blend"] + 0.005
        )
        if ok and s["blend"] < best_score:
            best, best_score = tag, s["blend"]

    submit_ok = best != "baseline"
    decision = {"summary": summary, "best": best, "submit_ok": submit_ok}
    lines.append(f"\nBest under gate: **{best}**; submit_ok={submit_ok}\n")
    (C.REPORTS_DIR / "13_c_mask_aug.md").write_text("".join(lines), encoding="utf-8")
    (C.REPORTS_DIR / "13_c_mask_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print("DECISION", decision)


if __name__ == "__main__":
    main()
