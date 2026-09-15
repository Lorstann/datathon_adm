"""C sikilastirma: daha fazla agac, hiper, seed ortalamasi. Bugun LB.

Kullanim: python scripts/18_c_tune.py
Harici TÜİK/EPİAŞ kapali (LB 1.15 C ile ayni iskelet).
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
from src.models.level_shape import entity_level, fit_shape_model, predict_level_shape
from src.models.pipeline import fold_frames, mask_history_entities, xy
from src.tracking import log_experiment
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

C_BLEND_REF = 1.3359
C_COLD_REF = 1.8746

# fold_frames assemble'i cagiriyor — use_external icin pipeline'i monkey patch yerine
# burada ozel fold_frames kopyasi
from src.features.history import add_lagged_history as _alh
from src.features.history import entity_history_features as _ehf
from src.features.peer import peer_tables as _pt


def fold_frames_wx(split, static, weather):
    origin = split.fold.origin
    hist = _ehf(split.train, origin)
    peer = _pt(split.train, origin)
    tr = _alh(split.train)
    train_f = assemble(
        tr,
        origin=origin,
        history=hist,
        weather=weather,
        static=static,
        peer=peer,
        include_history=False,
        use_external=False,
    )
    valid_f = assemble(
        split.valid,
        origin=origin,
        history=hist,
        weather=weather,
        static=static,
        peer=peer,
        include_history=True,
        use_external=False,
    )
    return train_f, valid_f, hist


VARIANTS = {
    "baseline_400": dict(n_trees=400, seeds=(C.SEED + 3,), params=None),
    "trees_700": dict(n_trees=700, seeds=(C.SEED + 3,), params=None),
    "deep_leaves": dict(
        n_trees=600,
        seeds=(C.SEED + 3,),
        params={"num_leaves": 127, "min_child_samples": 40, "learning_rate": 0.03},
    ),
    "seed_avg3_500": dict(
        n_trees=500,
        seeds=(C.SEED + 3, C.SEED + 17, C.SEED + 99),
        params=None,
    ),
    "seed_avg3_deep": dict(
        n_trees=600,
        seeds=(C.SEED + 3, C.SEED + 17, C.SEED + 99),
        params={"num_leaves": 95, "min_child_samples": 50, "learning_rate": 0.04},
    ),
}


def predict_variant(Xtr_m, ytr, level_tr, Xva, level_va, cat, hist_cols, cfg):
    preds = []
    for s in cfg["seeds"]:
        m = fit_shape_model(
            Xtr_m,
            ytr,
            level_tr,
            cat,
            hist_cols,
            seed=s,
            n_trees=cfg["n_trees"],
            params_override=cfg["params"],
        )
        preds.append(predict_level_shape(m, Xva, level_va))
    return np.mean(np.stack(preds, axis=0), axis=0)


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
        "# C tune (seed / trees / hyper)\n",
        f"Uretim: {pd.Timestamp.now()}\n",
        f"Ref: blend={C_BLEND_REF:.4f} cold={C_COLD_REF:.4f}\n",
        "use_external=False (LB C iskeleti).\n",
    ]

    for fold in C.FOLDS:
        print(f"=== {fold.name} ===")
        split = make_fold(train, fold, profile, seed=C.SEED)
        train_f, valid_f, hist = fold_frames_wx(split, static, weather)
        Xtr, ytr, cat, logg_tr, tr_use = xy(train_f)
        Xva, _, _, logg_va, va_use = xy(valid_f, dropna_target=True)
        Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
        Xva = Xva[Xtr.columns]
        Xtr_m = mask_history_entities(Xtr, tr_use[C.ENTITY], 0.30, C.SEED)
        level_tr = entity_level(hist, tr_use[C.ENTITY], logg_tr)
        level_va = entity_level(hist, va_use[C.ENTITY], logg_va)
        hist_cols = [c for c in Xtr_m.columns if c.startswith("hist_")]

        lines.append(f"\n## Fold {fold.name}\n")
        lines.append("| variant | warm | cold | blend |\n| --- | --- | --- | --- |\n")
        for name, cfg in VARIANTS.items():
            pred = predict_variant(
                Xtr_m, ytr, level_tr, Xva, level_va, cat, hist_cols, cfg
            )
            m = segment_report(va_use[C.TARGET], pred, split.segment, is_cold=split.is_cold)
            rows.append({"model": name, "fold": fold.name, **m})
            log_experiment(
                exp_id=f"ctune_{name}",
                model=name,
                target="log1p",
                feature_set="weather_no_external",
                fold=fold.name,
                metrics=m,
                notes=str(cfg),
            )
            lines.append(
                f"| {name} | {m.get('rmsle_warm', float('nan')):.4f} | "
                f"{m.get('rmsle_cold', float('nan')):.4f} | {m.get('rmsle_blend', float('nan')):.4f} |\n"
            )
            print(f"  {name}: blend={m['rmsle_blend']:.4f} cold={m['rmsle_cold']:.4f}")

    df = pd.DataFrame(rows)
    lines.append("\n## Ozet\n| variant | blend | cold |\n| --- | --- | --- |\n")
    summary = {}
    for name, g in df.groupby("model"):
        b = summarize_folds(g, "rmsle_blend")["mean"]
        c = summarize_folds(g, "rmsle_cold")["mean"]
        summary[name] = {"blend": float(b), "cold": float(c)}
        lines.append(f"| {name} | {b:.4f} | {c:.4f} |\n")

    base_b = summary["baseline_400"]["blend"]
    base_c = summary["baseline_400"]["cold"]
    best = "baseline_400"
    for name, s in summary.items():
        if name == "baseline_400":
            continue
        # LB odaklı: blend iyi veya esit + cold kotulesmesin; esik biraz yumusak
        ok = s["blend"] <= base_b - 0.005 and s["cold"] <= base_c + 0.01
        ok2 = s["blend"] <= base_b + 0.002 and s["cold"] <= base_c - 0.01
        if (ok or ok2) and s["blend"] <= summary[best]["blend"]:
            best = name

    # submit if best beats baseline meaningfully OR seed avg is not worse and we want LB diversity
    submit_ok = best != "baseline_400" and (
        summary[best]["blend"] <= base_b - 0.005
        and summary[best]["cold"] <= base_c + 0.01
    )
    # also allow seed_avg if blend <= baseline (tie-break for LB variance reduction)
    if not submit_ok:
        for cand in ("seed_avg3_deep", "seed_avg3_500", "trees_700", "deep_leaves"):
            s = summary[cand]
            if s["blend"] <= base_b and s["cold"] <= base_c + 0.005:
                best = cand
                submit_ok = True
                break

    decision = {"summary": summary, "best": best, "submit_ok": submit_ok}
    lines.append(f"\nBest: **{best}**; submit_ok={submit_ok}\n")
    (C.REPORTS_DIR / "19_c_tune.md").write_text("".join(lines), encoding="utf-8")
    (C.REPORTS_DIR / "19_c_tune_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print("DECISION", decision)

    if not submit_ok:
        print("Esik zayif — yine de en iyi seed_avg submission uretiliyor (LB denemesi).")
        # Bugun yapacagiz: lokal kotülese de seed avg LB'de yardimci olabilir
        # Kullanici istedi — en az kotu olmayani gonder
        best = min(summary.items(), key=lambda kv: (kv[1]["blend"], kv[1]["cold"]))[0]
        if summary[best]["cold"] > base_c + 0.02:
            print("Cold cok kotu, gonderim yok.")
            decision["submit_ok"] = False
            decision["best"] = best
            (C.REPORTS_DIR / "19_c_tune_decision.json").write_text(
                json.dumps(decision, indent=2), encoding="utf-8"
            )
            return
        submit_ok = True
        decision["submit_ok"] = True
        decision["best"] = best
        decision["note"] = "LB denemesi: lokal esik gevsek, cold kontrolu var"

    cfg = VARIANTS[best]
    origin = C.TRAIN_END
    hist = entity_history_features(train, origin)
    peer = peer_tables(train, origin)
    tr = add_lagged_history(train.sort_values([C.ENTITY, C.DATE], ignore_index=True))
    train_f = assemble(
        tr,
        origin=origin,
        history=hist,
        weather=weather,
        static=static,
        peer=peer,
        include_history=False,
        use_external=False,
    )
    test_f = assemble(
        test,
        origin=origin,
        history=hist,
        weather=weather,
        static=static,
        peer=peer,
        include_history=True,
        use_external=False,
    )
    y = np.log1p(train_f[C.TARGET].clip(lower=0).to_numpy())
    Xtr, cat = model_matrix(train_f)
    Xte, _ = model_matrix(test_f)
    Xtr, Xte = Xtr.align(Xte, join="outer", axis=1, fill_value=np.nan)
    Xte = Xte[Xtr.columns]
    Xtr_m = mask_history_entities(Xtr, train_f[C.ENTITY], 0.30, C.SEED)
    level_tr = entity_level(hist, train_f[C.ENTITY], train_f["log_guc"].to_numpy())
    level_te = entity_level(hist, test_f[C.ENTITY], test_f["log_guc"].to_numpy())
    hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]
    print(f"final egitim {best}...")
    pred = predict_variant(
        Xtr_m, y, level_tr, Xte, level_te, cat, hist_cols, cfg
    )
    out = sample.copy()
    out[C.TARGET] = np.clip(pred, 0.0, None)
    path = C.SUBMISSIONS_DIR / "submission_c_tuned.csv"
    out.to_csv(path, index=False)
    decision["submission"] = str(path)
    (C.REPORTS_DIR / "19_c_tune_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(f"yazildi {path} mean={out[C.TARGET].mean():.1f}")


if __name__ == "__main__":
    main()
