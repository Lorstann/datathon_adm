"""Warm C + cold B router CV ve (esik tutarsa) submission.

Kullanim: python scripts/12_router_cb.py
Kabul: blend, havayli C baseline'dan >=0.01 iyi VE cold kotulesmesin.
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
from src.models.pipeline import fold_frames, mask_history_entities, xy
from src.models.router import (
    drop_history_cols,
    fit_cold_z,
    is_cold_from_hist,
    predict_router_cb,
)
from src.models.level_shape import entity_level, fit_shape_model, predict_level_shape
from src.models.ensemble import cold_warm_blend
from src.tracking import log_experiment
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold


def _weather() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(CACHE_PATH)
    return weather_features(load_weather())


def eval_fold(split, static, weather, *, n_trees_shape=300, n_trees_cold=300):
    train_f, valid_f, hist = fold_frames(split, static, weather)
    Xtr, ytr, cat, logg_tr, tr_use = xy(train_f)
    Xva, _, _, logg_va, va_use = xy(valid_f, dropna_target=True)
    Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
    Xva = Xva[Xtr.columns]
    Xtr_m = mask_history_entities(Xtr, tr_use[C.ENTITY], 0.30, C.SEED)

    # C
    level_tr = entity_level(hist, tr_use[C.ENTITY], logg_tr)
    level_va = entity_level(hist, va_use[C.ENTITY], logg_va)
    hist_cols = [c for c in Xtr_m.columns if c.startswith("hist_")]
    shape = fit_shape_model(
        Xtr_m, ytr, level_tr, cat, hist_cols, seed=C.SEED + 3, n_trees=n_trees_shape
    )
    pred_c = predict_level_shape(shape, Xva, level_va)

    # B cold-only (hist dusur)
    cold = fit_cold_z(Xtr, ytr, logg_tr, cat, seed=C.SEED + 2, n_trees=n_trees_cold)
    Xva_c = drop_history_cols(Xva).reindex(columns=cold.feature_names)
    pred_b = cold.predict(Xva_c, logg_va)

    # router: CV is_cold
    pred_r = cold_warm_blend(pred_c, pred_b, split.is_cold)

    # ekstra: cold satirda C seviyesi yerine B (ayni)
    y = va_use[C.TARGET]
    out = {
        "level_shape_C": segment_report(y, pred_c, split.segment, is_cold=split.is_cold),
        "cold_B_alone": segment_report(y, pred_b, split.segment, is_cold=split.is_cold),
        "router_CB": segment_report(y, pred_r, split.segment, is_cold=split.is_cold),
    }
    return out, shape, cold, hist


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
    print("egitim B cold...")
    cold = fit_cold_z(Xtr, y, logg_tr, cat, seed=C.SEED + 2, n_trees=400)

    pred_c = predict_level_shape(shape, Xte, level_te)
    Xte_c = drop_history_cols(Xte).reindex(columns=cold.feature_names)
    pred_b = cold.predict(Xte_c, logg_te)
    is_cold = is_cold_from_hist(test_f[C.ENTITY], hist)
    print(f"test cold satir orani: {is_cold.mean():.3f}")
    pred = cold_warm_blend(pred_c, pred_b, is_cold)

    out = sample.copy()
    out[C.TARGET] = np.clip(pred, 0.0, None)
    path = C.SUBMISSIONS_DIR / "submission_router_cb.csv"
    out.to_csv(path, index=False)
    print(f"yazildi {path}  mean={out[C.TARGET].mean():.1f}  cold_frac={is_cold.mean():.3f}")
    return path


def main() -> None:
    C.ensure_dirs()
    weather = _weather()
    train = load_train()
    test = load_test()
    sample = load_sample_submission()
    panel = build_panel(train, test)
    static = entity_static(panel)
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))

    rows = []
    lines = [
        "# Warm C + cold B router\n",
        f"Uretim: {pd.Timestamp.now()}\n",
        "Warm = level_shape_C; cold = hist-dusurulmus z-LGBM; router = is_cold.\n",
    ]

    for fold in C.FOLDS:
        print(f"=== {fold.name} ===")
        split = make_fold(train, fold, profile, seed=C.SEED)
        metrics, _, _, _ = eval_fold(split, static, weather)
        lines.append(f"\n## Fold {fold.name}\n")
        lines.append("| Model | all | warm | cold | blend |\n| --- | --- | --- | --- | --- |\n")
        for name, m in metrics.items():
            rows.append({"model": name, "fold": fold.name, **m})
            log_experiment(
                exp_id=f"router_{name}",
                model=name,
                target="log1p",
                feature_set="full_weather",
                fold=fold.name,
                metrics=m,
                notes="C+B router",
            )
            lines.append(
                f"| {name} | {m['rmsle_all']:.4f} | {m.get('rmsle_warm', float('nan')):.4f} | "
                f"{m.get('rmsle_cold', float('nan')):.4f} | {m.get('rmsle_blend', float('nan')):.4f} |\n"
            )
            print(
                f"  {name}: blend={m['rmsle_blend']:.4f} warm={m.get('rmsle_warm', float('nan')):.4f} "
                f"cold={m.get('rmsle_cold', float('nan')):.4f}"
            )

    df = pd.DataFrame(rows)
    lines.append("\n## Ozet (rmsle_blend / rmsle_cold)\n")
    lines.append("| Model | blend | cold |\n| --- | --- | --- |\n")
    summary = {}
    for name, g in df.groupby("model"):
        b = summarize_folds(g, "rmsle_blend")["mean"]
        c = summarize_folds(g, "rmsle_cold")["mean"]
        summary[name] = {"blend": float(b), "cold": float(c)}
        lines.append(f"| {name} | {b:.4f} | {c:.4f} |\n")

    base_b = summary["level_shape_C"]["blend"]
    base_c = summary["level_shape_C"]["cold"]
    r_b = summary["router_CB"]["blend"]
    r_c = summary["router_CB"]["cold"]
    # warm korunmali: router warm ~= C warm (otomatik ayni warm tahmin)
    accept = (r_b <= base_b - 0.01) and (r_c <= base_c + 0.01)
    # cold iyilesmesi ana hedef: en az 0.02 cold VE blend kotulesmesin
    accept_alt = (r_c <= base_c - 0.02) and (r_b <= base_b + 0.005)
    submit_ok = accept or accept_alt

    decision = {
        "summary": summary,
        "delta_blend": float(r_b - base_b),
        "delta_cold": float(r_c - base_c),
        "submit_ok": bool(submit_ok),
        "reason": (
            f"router blend {r_b:.4f} (delta {r_b - base_b:+.4f}), "
            f"cold {r_c:.4f} (delta {r_c - base_c:+.4f})"
        ),
    }
    lines.append(
        f"\nDelta vs C: blend {r_b - base_b:+.4f}, cold {r_c - base_c:+.4f}\n"
        f"submit_ok: **{submit_ok}**\n"
    )
    (C.REPORTS_DIR / "11_router_cb.md").write_text("".join(lines), encoding="utf-8")
    (C.REPORTS_DIR / "11_router_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print("DECISION", decision)

    if submit_ok:
        path = build_submission(weather, train, test, sample, static)
        decision["submission"] = str(path)
        (C.REPORTS_DIR / "11_router_decision.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )
    else:
        print("Esik tutulmadi — submission uretilmedi.")


if __name__ == "__main__":
    main()
