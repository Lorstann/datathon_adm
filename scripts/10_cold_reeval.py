"""Cold seviye varyantlarini hizli karsilastir (hava cache ile).

Varyantlar:
  baseline   — global_z + log_guc (eski C)
  peer       — peer_guc_z + log_guc
  peer_soft  — peer + soft sifir post-gate (yalniz cold satirlar)
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
from src.models.level_shape import (
    entity_level,
    fit_shape_model,
    predict_level_shape,
    soft_zero_factor,
)
from src.models.pipeline import fold_frames, mask_history_entities, xy
from src.tracking import log_experiment
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold


def eval_variant(split, static, weather, *, variant: str, n_trees: int = 300):
    train_f, valid_f, hist = fold_frames(split, static, weather)
    Xtr, ytr, cat, logg_tr, tr_use = xy(train_f)
    Xva, _, _, logg_va, va_use = xy(valid_f, dropna_target=True)
    Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
    Xva = Xva[Xtr.columns]
    Xtr_m = mask_history_entities(Xtr, tr_use[C.ENTITY], 0.30, C.SEED)

    use_peer = variant in ("peer", "peer_soft")
    peer_tr = tr_use["peer_guc_z"] if use_peer and "peer_guc_z" in tr_use.columns else None
    peer_va = va_use["peer_guc_z"] if use_peer and "peer_guc_z" in va_use.columns else None

    level_tr = entity_level(hist, tr_use[C.ENTITY], logg_tr, peer_guc_z=peer_tr)
    level_va = entity_level(hist, va_use[C.ENTITY], logg_va, peer_guc_z=peer_va)

    hist_cols = [c for c in Xtr_m.columns if c.startswith("hist_")]
    model = fit_shape_model(
        Xtr_m, ytr, level_tr, cat, hist_cols, seed=C.SEED + 3, n_trees=n_trees
    )

    soft = None
    if variant == "peer_soft":
        soft = soft_zero_factor(
            va_use["peer_zero_guc"] if "peer_zero_guc" in va_use.columns else None,
            va_use["devreye_alinma_yasi"] if "devreye_alinma_yasi" in va_use.columns else None,
            strength=0.35,
        )
        # yalniz cold satirlar
        cold = split.is_cold.to_numpy()
        if soft is not None:
            soft = np.where(cold, soft, 1.0)

    pred = predict_level_shape(model, Xva, level_va, soft_factor=soft)
    return segment_report(va_use[C.TARGET], pred, split.segment, is_cold=split.is_cold)


def main() -> None:
    C.ensure_dirs()
    assert CACHE_PATH.exists(), CACHE_PATH
    weather = weather_features(load_weather())
    train = load_train()
    test = load_test()
    panel = build_panel(train, test)
    static = entity_static(panel)
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))

    rows = []
    lines = ["# Cold level patch re-eval\n", f"Uretim: {pd.Timestamp.now()}\n"]
    for fold in C.FOLDS:
        print(f"=== {fold.name} ===")
        split = make_fold(train, fold, profile, seed=C.SEED)
        lines.append(f"\n## Fold {fold.name}\n")
        lines.append("| variant | all | warm | cold | blend |\n| --- | --- | --- | --- | --- |\n")
        for var in ("baseline", "peer", "peer_soft"):
            m = eval_variant(split, static, weather, variant=var)
            rows.append({"model": var, "fold": fold.name, **m})
            log_experiment(
                exp_id=f"cold_{var}",
                model=f"level_shape_{var}",
                target="log1p",
                feature_set="full_weather",
                fold=fold.name,
                metrics=m,
                notes="cold patch re-eval",
            )
            lines.append(
                f"| {var} | {m['rmsle_all']:.4f} | {m.get('rmsle_warm', float('nan')):.4f} | "
                f"{m.get('rmsle_cold', float('nan')):.4f} | {m.get('rmsle_blend', float('nan')):.4f} |\n"
            )
            print(f"  {var}: blend={m['rmsle_blend']:.4f} cold={m['rmsle_cold']:.4f}")

    df = pd.DataFrame(rows)
    lines.append("\n## Ozet\n| variant | blend | cold |\n| --- | --- | --- |\n")
    summary = {}
    for var, g in df.groupby("model"):
        b = summarize_folds(g, "rmsle_blend")["mean"]
        c = summarize_folds(g, "rmsle_cold")["mean"]
        summary[var] = {"blend": float(b), "cold": float(c)}
        lines.append(f"| {var} | {b:.4f} | {c:.4f} |\n")

    base_b, base_c = summary["baseline"]["blend"], summary["baseline"]["cold"]
    # en iyi adayi sec: blend dusuk, cold kotulesmesin
    best = "baseline"
    best_gain = 0.0
    for var in ("peer", "peer_soft"):
        if summary[var]["cold"] <= base_c + 0.01 and summary[var]["blend"] < base_b - 0.005:
            gain = base_b - summary[var]["blend"]
            if gain > best_gain:
                best, best_gain = var, gain

    accept = best != "baseline" and best_gain >= 0.01
    # plan: en az 0.01 blend VE cold kotulesmesin
    if best != "baseline":
        accept = (base_b - summary[best]["blend"]) >= 0.01 and summary[best]["cold"] <= base_c + 0.01

    decision = {
        "summary": summary,
        "best_variant": best,
        "accept_patch": best != "baseline",
        "submit_ok": bool(accept),
        "reason": (
            f"best={best} blend_gain={base_b - summary.get(best, summary['baseline'])['blend']:+.4f}"
        ),
    }
    lines.append(f"\nBest: **{best}**; submit_ok={accept}\n")
    (C.REPORTS_DIR / "06b_cold_patch.md").write_text("".join(lines), encoding="utf-8")
    (C.REPORTS_DIR / "09_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print("DECISION", decision)


if __name__ == "__main__":
    main()
