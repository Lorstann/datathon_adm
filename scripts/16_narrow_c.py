"""NO-GO sonrasi dar C: YoY cold seviye + sequential cold bias + mild soft-zero.

Kullanim: python scripts/16_narrow_c.py
Kabul: blend >=0.01 iyi VE cold kotulesmesin (vs havayla C ref).
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
from src.models.level_shape import (
    apply_cold_log_bias,
    entity_level,
    fit_shape_model,
    learn_cold_bias,
    map_yoy_z,
    predict_level_shape,
    soft_zero_factor,
    yoy_guc_month_z_table,
)
from src.models.pipeline import fold_frames, mask_history_entities, xy
from src.models.router import is_cold_from_hist
from src.tracking import log_experiment
from src.validation.folds import build_cold_profile, guc_band
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

C_BLEND_REF = 1.3359
C_COLD_REF = 1.8746


def _level(hist, frame, log_guc, z_table, *, use_yoy: bool):
    if use_yoy and z_table is not None and len(z_table):
        if "guc_band" not in frame.columns:
            frame = frame.copy()
            frame["guc_band"] = guc_band(frame[C.POWER]).astype("string")
        if "month" not in frame.columns:
            frame = frame.copy()
            frame["month"] = pd.to_datetime(frame[C.DATE]).dt.month.astype("int8")
        peer = map_yoy_z(frame, z_table)
        return entity_level(hist, frame[C.ENTITY], log_guc, peer_guc_z=peer)
    return entity_level(hist, frame[C.ENTITY], log_guc)


def run_variant(
    split,
    static,
    weather,
    raw_train,
    *,
    use_yoy: bool,
    bias_month: dict[int, float] | None,
    bias_hol: float,
    use_soft: bool,
    n_trees: int = 300,
):
    train_f, valid_f, hist = fold_frames(split, static, weather)
    Xtr, ytr, cat, logg_tr, tr_use = xy(train_f)
    Xva, _, _, logg_va, va_use = xy(valid_f, dropna_target=True)
    Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
    Xva = Xva[Xtr.columns]
    Xtr_m = mask_history_entities(Xtr, tr_use[C.ENTITY], 0.30, C.SEED)

    z_table = yoy_guc_month_z_table(raw_train.loc[raw_train[C.DATE] <= split.fold.origin], split.fold.origin) if use_yoy else None
    # ensure month/guc_band on frames
    for df in (tr_use, va_use):
        if "guc_band" not in df.columns:
            df["guc_band"] = guc_band(df[C.POWER]).astype("string")
        if "month" not in df.columns:
            df["month"] = pd.to_datetime(df[C.DATE]).dt.month.astype("int8")

    level_tr = _level(hist, tr_use, logg_tr, z_table, use_yoy=use_yoy)
    level_va = _level(hist, va_use, logg_va, z_table, use_yoy=use_yoy)
    hist_cols = [c for c in Xtr_m.columns if c.startswith("hist_")]
    model = fit_shape_model(
        Xtr_m, ytr, level_tr, cat, hist_cols, seed=C.SEED + 3, n_trees=n_trees
    )
    soft = None
    if use_soft:
        soft = soft_zero_factor(
            va_use["peer_zero_guc"] if "peer_zero_guc" in va_use.columns else None,
            va_use["devreye_alinma_yasi"] if "devreye_alinma_yasi" in va_use.columns else None,
            strength=0.25,
        )
        if soft is not None:
            soft = np.where(np.asarray(split.is_cold, dtype=bool), soft, 1.0)

    pred = predict_level_shape(model, Xva, level_va, soft_factor=soft)
    if bias_month is not None:
        hol = va_use["is_holiday"].to_numpy() if "is_holiday" in va_use.columns else None
        pred = apply_cold_log_bias(
            pred,
            np.asarray(split.is_cold, dtype=bool),
            va_use["month"].to_numpy(),
            hol,
            bias_month,
            bias_hol,
        )
    metrics = segment_report(va_use[C.TARGET], pred, split.segment, is_cold=split.is_cold)
    # residuals for learning bias (always from this fold's raw C-like pred before external bias)
    return metrics, pred, va_use, model


def main() -> None:
    C.ensure_dirs()
    assert CACHE_PATH.exists()
    weather = weather_features(load_weather())
    train = load_train()
    test = load_test()
    sample = load_sample_submission()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))

    variants = ("baseline", "yoy", "yoy_bias", "yoy_bias_soft")
    rows = []
    lines = [
        "# Narrow C (NO-GO hierarchy path)\n",
        f"Uretim: {pd.Timestamp.now()}\n",
        f"Ref C: blend={C_BLEND_REF:.4f} cold={C_COLD_REF:.4f}\n",
    ]

    # sequential bias store from prior folds (learned on yoy predictions without bias)
    accumulated_month: dict[int, list[float]] = {}
    accumulated_hol: list[float] = []

    fold_yoy_preds = []  # for building test bias from fold A only (seasonal)

    for fold in C.FOLDS:
        print(f"=== {fold.name} ===")
        split = make_fold(train, fold, profile, seed=C.SEED)
        # bias from prior folds only
        if accumulated_month:
            bias_m = {k: float(np.median(v)) for k, v in accumulated_month.items()}
            bias_h = float(np.median(accumulated_hol)) if accumulated_hol else 0.0
        else:
            bias_m, bias_h = {}, 0.0

        configs = {
            "baseline": dict(use_yoy=False, bias_month=None, bias_hol=0.0, use_soft=False),
            "yoy": dict(use_yoy=True, bias_month=None, bias_hol=0.0, use_soft=False),
            "yoy_bias": dict(use_yoy=True, bias_month=bias_m or None, bias_hol=bias_h, use_soft=False),
            "yoy_bias_soft": dict(
                use_yoy=True, bias_month=bias_m or None, bias_hol=bias_h, use_soft=True
            ),
        }

        lines.append(f"\n## Fold {fold.name}\n")
        lines.append("| variant | warm | cold | blend |\n| --- | --- | --- | --- |\n")

        # run yoy without bias first to learn residuals for next folds
        m_yoy, pred_yoy, va_yoy, _ = run_variant(
            split, static, weather, train, use_yoy=True, bias_month=None, bias_hol=0.0, use_soft=False
        )
        sh_m, sh_h = learn_cold_bias(
            va_yoy[C.TARGET].to_numpy(),
            pred_yoy,
            np.asarray(split.is_cold, dtype=bool),
            va_yoy["month"].to_numpy(),
            va_yoy["is_holiday"].to_numpy() if "is_holiday" in va_yoy.columns else None,
        )
        for k, v in sh_m.items():
            accumulated_month.setdefault(k, []).append(v)
        if sh_h != 0.0:
            accumulated_hol.append(sh_h)
        if fold.name == "A_mevsim":
            fold_yoy_preds.append((sh_m, sh_h))

        for name, cfg in configs.items():
            if name == "yoy":
                metrics = m_yoy
            else:
                metrics, _, _, _ = run_variant(split, static, weather, train, **cfg)
            rows.append({"model": name, "fold": fold.name, **metrics})
            log_experiment(
                exp_id=f"narrow_{name}",
                model=name,
                target="log1p",
                feature_set="narrow_c",
                fold=fold.name,
                metrics=metrics,
                notes="post hierarchy NO-GO",
            )
            lines.append(
                f"| {name} | {metrics.get('rmsle_warm', float('nan')):.4f} | "
                f"{metrics.get('rmsle_cold', float('nan')):.4f} | {metrics.get('rmsle_blend', float('nan')):.4f} |\n"
            )
            print(f"  {name}: blend={metrics['rmsle_blend']:.4f} cold={metrics['rmsle_cold']:.4f}")

    df = pd.DataFrame(rows)
    lines.append("\n## Ozet\n| variant | blend | cold |\n| --- | --- | --- |\n")
    summary = {}
    for name, g in df.groupby("model"):
        b = summarize_folds(g, "rmsle_blend")["mean"]
        c = summarize_folds(g, "rmsle_cold")["mean"]
        summary[name] = {"blend": float(b), "cold": float(c)}
        lines.append(f"| {name} | {b:.4f} | {c:.4f} |\n")

    base_b, base_c = summary["baseline"]["blend"], summary["baseline"]["cold"]
    best = "baseline"
    for name, s in summary.items():
        if name == "baseline":
            continue
        ok = (s["blend"] <= base_b - 0.01 and s["cold"] <= base_c + 0.01) or (
            s["cold"] <= base_c - 0.02 and s["blend"] <= base_b + 0.005
        )
        # also vs ref C
        ok_ref = (s["blend"] <= C_BLEND_REF - 0.01 and s["cold"] <= C_COLD_REF + 0.01)
        if (ok or ok_ref) and s["blend"] < summary[best]["blend"]:
            best = name

    submit_ok = best != "baseline" and (
        summary[best]["blend"] <= C_BLEND_REF - 0.01
        and summary[best]["cold"] <= C_COLD_REF + 0.01
    )
    # allow submit if beats this run's baseline by 0.01 and cold ok
    if best != "baseline" and summary[best]["blend"] <= base_b - 0.01 and summary[best]["cold"] <= base_c + 0.01:
        submit_ok = True

    decision = {
        "summary": summary,
        "best": best,
        "submit_ok": submit_ok,
        "path": "narrow_C",
    }
    lines.append(f"\nBest: **{best}**; submit_ok={submit_ok}\n")
    (C.REPORTS_DIR / "16_narrow_c.md").write_text("".join(lines), encoding="utf-8")
    (C.REPORTS_DIR / "16_narrow_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print("DECISION", decision)

    if not submit_ok:
        print("Esik tutulmadi — submission/LB yok.")
        return

    # build submission with best config
    use_yoy = best in ("yoy", "yoy_bias", "yoy_bias_soft")
    use_soft = best == "yoy_bias_soft"
    use_bias = best in ("yoy_bias", "yoy_bias_soft")
    # seasonal bias from fold A
    bias_m = fold_yoy_preds[0][0] if (use_bias and fold_yoy_preds) else {}
    bias_h = fold_yoy_preds[0][1] if (use_bias and fold_yoy_preds) else 0.0

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
    for df in (train_f, test_f):
        if "guc_band" not in df.columns:
            df["guc_band"] = guc_band(df[C.POWER]).astype("string")
        if "month" not in df.columns:
            df["month"] = pd.to_datetime(df[C.DATE]).dt.month.astype("int8")
    logg_tr = train_f["log_guc"].to_numpy()
    logg_te = test_f["log_guc"].to_numpy()
    Xtr_m = mask_history_entities(Xtr, train_f[C.ENTITY], 0.30, C.SEED)
    z_table = yoy_guc_month_z_table(train, origin) if use_yoy else None
    level_tr = _level(hist, train_f, logg_tr, z_table, use_yoy=use_yoy)
    level_te = _level(hist, test_f, logg_te, z_table, use_yoy=use_yoy)
    hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]
    print("egitim final C...", best)
    model = fit_shape_model(Xtr_m, y, level_tr, cat, hist_cols, seed=C.SEED + 3, n_trees=400)
    soft = None
    if use_soft:
        soft = soft_zero_factor(
            test_f.get("peer_zero_guc"),
            test_f.get("devreye_alinma_yasi"),
            strength=0.25,
        )
        is_cold = is_cold_from_hist(test_f[C.ENTITY], hist)
        if soft is not None:
            soft = np.where(is_cold, soft, 1.0)
    else:
        is_cold = is_cold_from_hist(test_f[C.ENTITY], hist)
    pred = predict_level_shape(model, Xte, level_te, soft_factor=soft)
    if use_bias and bias_m:
        hol = test_f["is_holiday"].to_numpy() if "is_holiday" in test_f.columns else None
        pred = apply_cold_log_bias(pred, is_cold, test_f["month"].to_numpy(), hol, bias_m, bias_h)
    out = sample.copy()
    out[C.TARGET] = np.clip(pred, 0.0, None)
    path = C.SUBMISSIONS_DIR / "submission_narrow_c_v3.csv"
    out.to_csv(path, index=False)
    decision["submission"] = str(path)
    (C.REPORTS_DIR / "16_narrow_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(f"yazildi {path} mean={out[C.TARGET].mean():.1f}")


if __name__ == "__main__":
    main()
