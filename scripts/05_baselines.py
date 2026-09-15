"""Naif tabanlari 3 fold'da skorlar.

Kullanim: python scripts/05_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src import config as C
from src.data.load import build_panel, entity_static, load_test, load_train
from src.models.baselines import run_baselines
from src.tracking import log_experiment
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold


def main() -> None:
    C.ensure_dirs()
    train = load_train()
    test = load_test()
    panel = build_panel(train, test)
    static = entity_static(panel)
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))

    rows = []
    lines = ["# Naif tabanlar\n", f"Uretim: {pd.Timestamp.now()}\n"]
    for fold in C.FOLDS:
        split = make_fold(train, fold, profile, seed=C.SEED)
        preds = run_baselines(split.train, split.valid, fold.origin)
        lines.append(f"\n## Fold {fold.name} (sizinti {split.checks})\n")
        lines.append(f"train {len(split.train):,}  valid {len(split.valid):,}  cold {int(split.is_cold.sum())}\n")
        lines.append("| Model | rmsle_all | rmsle_warm | rmsle_cold | rmsle_blend |\n| --- | --- | --- | --- | --- |\n")
        for name, pred in preds.items():
            m = segment_report(
                split.valid[C.TARGET],
                pred,
                split.segment,
                is_cold=split.is_cold,
            )
            log_experiment(
                exp_id=f"baseline_{name}",
                model=name,
                target="log1p",
                feature_set="naive",
                fold=fold.name,
                metrics=m,
                notes="naif taban",
            )
            rows.append({"model": name, "fold": fold.name, **m})
            lines.append(
                f"| {name} | {m.get('rmsle_all', float('nan')):.4f} | "
                f"{m.get('rmsle_warm', float('nan')):.4f} | "
                f"{m.get('rmsle_cold', float('nan')):.4f} | "
                f"{m.get('rmsle_blend', float('nan')):.4f} |\n"
            )

    df = pd.DataFrame(rows)
    lines.append("\n## Fold'lar arasi ozet (rmsle_blend)\n")
    lines.append("| Model | mean | std |\n| --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append(f"| {model} | {s['mean']:.4f} | {s['std']:.4f} |\n")

    path = C.REPORTS_DIR / "04_baselines.md"
    path.write_text("".join(lines), encoding="utf-8")
    print(f"yazildi: {path}")
    print(df.groupby("model")["rmsle_blend"].mean().sort_values().to_string())


if __name__ == "__main__":
    main()
