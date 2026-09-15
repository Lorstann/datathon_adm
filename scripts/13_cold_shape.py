"""Cold iyilestirme turu 2: cold-only shape (seviye+sekil, hist yok).

Warm: mevcut C (hist seviyesi + sekil).
Cold: zorunlu seviye = global_z + log(guc); sekil modeli hist kolonlari
dusurulmus X uzerinde, hedef = y_log - cold_level (tum egitim).

Karsilastirma: C, cold_shape_alone, router_C_coldshape.
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
from src.data.load import (
    build_panel,
    entity_static,
    load_sample_submission,
    load_test,
    load_train,
)
from src.external.weather import CACHE_PATH, load_weather, weather_features
from src.features.build import assemble, model_matrix
from src.features.history import add_lagged_history, entity_history_features
from src.features.peer import peer_tables
from src.models.boosting import fit_lightgbm
from src.models.ensemble import cold_warm_blend
from src.models.level_shape import entity_level, fit_shape_model, predict_level_shape
from src.models.pipeline import fold_frames, mask_history_entities, xy
from src.models.router import drop_history_cols, is_cold_from_hist
from src.tracking import log_experiment
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold


def cold_level(hist: pd.DataFrame, log_guc: np.ndarray) -> np.ndarray:
    if "hist_mean_z" in hist.columns and hist["hist_mean_z"].notna().any():
        z = float(hist["hist_mean_z"].median())
    else:
        z = 0.0
    return z + np.asarray(log_guc, dtype="float64")


def fit_cold_shape(X, y_log, log_guc, hist, cat_cols, *, seed, n_trees=350):
    Xc = drop_history_cols(X)
    cats = [c for c in cat_cols if c in Xc.columns]
    level = cold_level(hist, log_guc)
    resid = y_log - level
    model = fit_lightgbm(
        Xc, resid, cats, seed=seed, target="log1p", mask_history_rate=0.0, n_trees=n_trees
    )
    return model, level


def predict_cold_shape(model, X, hist, log_guc):
    Xc = drop_history_cols(X).reindex(columns=model.feature_names)
    resid = model.predict_log1p(Xc, log_guc)
    level = cold_level(hist, log_guc)
    return np.clip(np.expm1(level + resid), 0.0, None)


def month_bias_correct(y_true, pred, month, is_cold, apply_month, apply_cold):
    """Cold x month medyan artik (log1p); ayni fold icinde fit+apply (iyimser ust sinir).

    Asil CV icin ayri holdout gerekir; burada hizli tavan olcumu.
    """
    a = np.log1p(np.clip(y_true, 0, None))
    p = np.log1p(np.clip(pred, 0, None))
    df = pd.DataFrame({"r": a - p, "m": month, "c": np.asarray(is_cold, dtype=bool)})
    shifts = df.loc[df["c"]].groupby("m")["r"].median().to_dict()
    out = p.copy()
    mask = np.asarray(apply_cold, dtype=bool)
    for m, d in shifts.items():
        sel = mask & (np.asarray(apply_month) == m)
        out[sel] = out[sel] + d
    return np.clip(np.expm1(out), 0.0, None)


def eval_fold(split, static, weather):
    train_f, valid_f, hist = fold_frames(split, static, weather)
    Xtr, ytr, cat, logg_tr, tr_use = xy(train_f)
    Xva, _, _, logg_va, va_use = xy(valid_f, dropna_target=True)
    Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
    Xva = Xva[Xtr.columns]
    Xtr_m = mask_history_entities(Xtr, tr_use[C.ENTITY], 0.30, C.SEED)

    # C warm path
    level_tr = entity_level(hist, tr_use[C.ENTITY], logg_tr)
    level_va = entity_level(hist, va_use[C.ENTITY], logg_va)
    hist_cols = [c for c in Xtr_m.columns if c.startswith("hist_")]
    shape = fit_shape_model(
        Xtr_m, ytr, level_tr, cat, hist_cols, seed=C.SEED + 3, n_trees=300
    )
    pred_c = predict_level_shape(shape, Xva, level_va)

    # cold-only shape
    cold_m, _ = fit_cold_shape(
        Xtr, ytr, logg_tr, hist, cat, seed=C.SEED + 7, n_trees=300
    )
    pred_cs = predict_cold_shape(cold_m, Xva, hist, logg_va)

    pred_r = cold_warm_blend(pred_c, pred_cs, split.is_cold)

    # tavan: cold x month bias (ayni fold — ust sinir, secim icin degil)
    pred_bias = month_bias_correct(
        va_use[C.TARGET].to_numpy(),
        pred_r,
        va_use["month"].to_numpy(),
        split.is_cold,
        va_use["month"].to_numpy(),
        split.is_cold,
    )

    y = va_use[C.TARGET]
    return {
        "level_shape_C": segment_report(y, pred_c, split.segment, is_cold=split.is_cold),
        "cold_shape": segment_report(y, pred_cs, split.segment, is_cold=split.is_cold),
        "router_C_cs": segment_report(y, pred_r, split.segment, is_cold=split.is_cold),
        "router_C_cs_monthbias_oracle": segment_report(
            y, pred_bias, split.segment, is_cold=split.is_cold
        ),
    }


def build_submission(weather, train, test, sample, static) -> Path:
    origin = C.TRAIN_END
    hist = entity_history_features(train, origin)
    peer = peer_tables(train, origin)
    tr = add_lagged_history(train.sort_values([C.ENTITY, C.DATE], ignore_index=True))
    train_f = assemble(
        tr, origin=origin, history=hist, weather=weather, static=static, peer=peer, include_history=False
    )
    test_f = assemble(
        test, origin=origin, history=hist, weather=weather, static=static, peer=peer, include_history=True
    )
    y = np.log1p(train_f[C.TARGET].clip(lower=0).to_numpy())
    Xtr, cat = model_matrix(train_f)
    Xte, _ = model_matrix(test_f)
    Xtr, Xte = Xtr.align(Xte, join="outer", axis=1, fill_value=np.nan)
    Xte = Xte[Xtr.columns]
    logg_tr = train_f["log_guc"].to_numpy()
    logg_te = test_f["log_guc"].to_numpy()
    Xtr_m = mask_history_entities(Xtr, train_f[C.ENTITY], 0.30, C.SEED)
    level_tr = entity_level(hist, train_f[C.ENTITY], logg_tr)
    level_te = entity_level(hist, test_f[C.ENTITY], logg_te)
    hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]
    print("egitim C...")
    shape = fit_shape_model(Xtr_m, y, level_tr, cat, hist_cols, seed=C.SEED + 3, n_trees=400)
    print("egitim cold shape...")
    cold_m, _ = fit_cold_shape(Xtr, y, logg_tr, hist, cat, seed=C.SEED + 7, n_trees=400)
    pred_c = predict_level_shape(shape, Xte, level_te)
    pred_cs = predict_cold_shape(cold_m, Xte, hist, logg_te)
    is_cold = is_cold_from_hist(test_f[C.ENTITY], hist)
    pred = cold_warm_blend(pred_c, pred_cs, is_cold)
    out = sample.copy()
    out[C.TARGET] = np.clip(pred, 0.0, None)
    path = C.SUBMISSIONS_DIR / "submission_router_ccs.csv"
    out.to_csv(path, index=False)
    print(f"yazildi {path} mean={out[C.TARGET].mean():.1f} cold_frac={is_cold.mean():.3f}")
    return path


def main() -> None:
    C.ensure_dirs()
    assert CACHE_PATH.exists()
    weather = weather_features(load_weather())
    train = load_train()
    test = load_test()
    sample = load_sample_submission()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))

    rows = []
    lines = [
        "# Cold-only shape router\n",
        f"Uretim: {pd.Timestamp.now()}\n",
        "Cold = (global_z+log_guc) + shape(no hist). Warm = C.\n",
    ]
    for fold in C.FOLDS:
        print(f"=== {fold.name} ===")
        split = make_fold(train, fold, profile, seed=C.SEED)
        metrics = eval_fold(split, static, weather)
        lines.append(f"\n## Fold {fold.name}\n")
        lines.append("| Model | all | warm | cold | blend |\n| --- | --- | --- | --- | --- |\n")
        for name, m in metrics.items():
            rows.append({"model": name, "fold": fold.name, **m})
            log_experiment(
                exp_id=f"ccs_{name}",
                model=name,
                target="log1p",
                feature_set="full_weather",
                fold=fold.name,
                metrics=m,
                notes="cold shape router",
            )
            lines.append(
                f"| {name} | {m['rmsle_all']:.4f} | {m.get('rmsle_warm', float('nan')):.4f} | "
                f"{m.get('rmsle_cold', float('nan')):.4f} | {m.get('rmsle_blend', float('nan')):.4f} |\n"
            )
            print(
                f"  {name}: blend={m['rmsle_blend']:.4f} cold={m.get('rmsle_cold', float('nan')):.4f}"
            )

    df = pd.DataFrame(rows)
    lines.append("\n## Ozet\n| Model | blend | cold |\n| --- | --- | --- |\n")
    summary = {}
    for name, g in df.groupby("model"):
        b = summarize_folds(g, "rmsle_blend")["mean"]
        c = summarize_folds(g, "rmsle_cold")["mean"]
        summary[name] = {"blend": float(b), "cold": float(c)}
        lines.append(f"| {name} | {b:.4f} | {c:.4f} |\n")

    base_b = summary["level_shape_C"]["blend"]
    base_c = summary["level_shape_C"]["cold"]
    r_b = summary["router_C_cs"]["blend"]
    r_c = summary["router_C_cs"]["cold"]
    submit_ok = ((r_b <= base_b - 0.01) and (r_c <= base_c + 0.01)) or (
        (r_c <= base_c - 0.02) and (r_b <= base_b + 0.005)
    )
    decision = {
        "summary": summary,
        "delta_blend": float(r_b - base_b),
        "delta_cold": float(r_c - base_c),
        "submit_ok": bool(submit_ok),
    }
    lines.append(
        f"\nDelta router_C_cs vs C: blend {r_b - base_b:+.4f}, cold {r_c - base_c:+.4f}\n"
        f"submit_ok: **{submit_ok}**\n"
        "Not: `monthbias_oracle` ayni fold artik medyani — tavan; secim kurali degil.\n"
    )
    (C.REPORTS_DIR / "12_cold_shape.md").write_text("".join(lines), encoding="utf-8")
    (C.REPORTS_DIR / "12_cold_shape_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print("DECISION", decision)

    if submit_ok:
        path = build_submission(weather, train, test, sample, static)
        decision["submission"] = str(path)
        (C.REPORTS_DIR / "12_cold_shape_decision.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )
    else:
        print("Esik tutulmadi — submission yok.")


if __name__ == "__main__":
    main()
