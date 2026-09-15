"""Olu-trafo kapisinin kalibrasyonu: on-bilgi agirligi x guc x anahtar.

Kapi ilk denemede kaba secilmisti (k grid'i, sabit `DEAD_PRIOR_WEIGHT=40`,
yalniz lokasyon anahtari). Cold+sifir dilimi hala kare hatanin %42-51'i ve
kahin tavani -0.34, yani kalibrasyonun kendisi olcmeye deger.

Kapi tahminden SONRA uygulandigi icin fold basina tek bir egitim yetiyor;
butun grid ayni tahminler uzerinde degerlendiriliyor.

Kullanim: python scripts/34_gate_grid.py
"""

from __future__ import annotations

import sys
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
from src.models.zero_gate import (
    DEAD_MIN_ROWS,
    DEAD_ZERO_RATE,
    shrink_level,
)
from src.validation.folds import build_cold_profile, guc_band
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

_cv = import_module("20_multiorigin_cv")

PRIORS = (10.0, 20.0, 40.0, 80.0)
STRENGTHS = (0.6, 0.75, 0.9)
KEYS = ("lokasyon", "lokasyon_guc")


def dead_table(observed, origin, *, key: str, prior: float):
    """Yeni giren trafolarin olu orani; secilen anahtarda, ampirik Bayes."""
    hist = observed.loc[observed[C.DATE] <= origin]
    first = hist.groupby(C.ENTITY)[C.DATE].min()
    new = first[first > hist[C.DATE].min() + pd.Timedelta(days=14)].index
    sub = hist.loc[hist[C.ENTITY].isin(set(new))]
    ent = sub.groupby(C.ENTITY).agg(
        zero_rate=(C.TARGET, lambda s: float((s <= 0).mean())),
        n=(C.TARGET, "size"),
        lokasyon=(C.LOCATION, "first"),
        guc=(C.POWER, "first"),
    )
    ent = ent.loc[ent["n"] >= DEAD_MIN_ROWS]
    ent["dead"] = (ent["zero_rate"] > DEAD_ZERO_RATE).astype("float64")
    glob = float(ent["dead"].mean())
    if key == "lokasyon":
        cols = ["lokasyon"]
    else:
        ent["gb"] = guc_band(ent["guc"]).astype(str)
        cols = ["lokasyon", "gb"]
    g = ent.groupby(cols)["dead"].agg(["sum", "size"])
    tab = (g["sum"] + prior * glob) / (g["size"] + prior)
    return glob, tab, cols


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    records = []
    lines = ["# Olu-trafo kapisi kalibrasyon gridi\n", f"Uretim: {pd.Timestamp.now()}\n"]

    for fold in C.FOLDS:
        split = make_fold(train, fold, profile, seed=C.SEED)
        pred, va, _m, _n = _cv.new_path(split, static, weather)
        y = va[C.TARGET].to_numpy()
        seg = split.segment.reindex(va.index)
        cold_s = split.is_cold.reindex(va.index)
        cold = cold_s.to_numpy()
        va = va.copy()
        va["gb"] = guc_band(va[C.POWER]).astype(str)

        lines.append(f"\n## Fold {fold.name}\n\n| anahtar | prior | k | cold | blend |\n")
        lines.append("| --- | --- | --- | --- | --- |\n")

        for key in KEYS:
            for prior in PRIORS:
                glob, tab, cols = dead_table(
                    split.train, fold.origin, key=key, prior=prior
                )
                if len(cols) == 1:
                    p = va[C.LOCATION].map(tab)
                else:
                    idx = pd.MultiIndex.from_arrays([va[C.LOCATION], va["gb"]])
                    p = pd.Series(tab.reindex(idx).to_numpy(), index=va.index)
                p = p.fillna(glob).to_numpy()
                p = np.where(cold, p, 0.0)
                for k in STRENGTHS:
                    met = segment_report(
                        y, shrink_level(pred, p, strength=k), seg, is_cold=cold_s
                    )
                    name = f"{key}_p{int(prior)}_k{k}"
                    records.append({"model": name, "fold": fold.name, **met})
                    lines.append(
                        "| {} | {:.0f} | {:.2f} | {:.4f} | {:.4f} |\n".format(
                            key, prior, k, met["rmsle_cold"], met["rmsle_blend"]
                        )
                    )
        print(fold.name, "bitti", flush=True)

    df = pd.DataFrame(records)
    lines.append("\n## Ozet (rmsle_blend)\n\n| variant | mean | std | cold |\n")
    lines.append("| --- | --- | --- | --- |\n")
    summ = []
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        summ.append((s["mean"], model, s["std"], g["rmsle_cold"].mean()))
    for mean, model, sd, cold in sorted(summ)[:12]:
        lines.append(f"| {model} | {mean:.4f} | {sd:.4f} | {cold:.4f} |\n")

    piv = df.pivot(index="fold", columns="model", values="rmsle_blend")
    base = piv["lokasyon_p40_k0.75"]
    lines.append("\n### Mevcut yapilandirmadan (lokasyon_p40_k0.75) fark\n\n")
    deltas = {c: (piv[c] - base) for c in piv.columns}
    ok = {
        c: d for c, d in deltas.items() if (d < 0).all()
    }
    for c, d in sorted(ok.items(), key=lambda kv: kv[1].mean())[:10]:
        lines.append(f"- **{c}**: {d.round(4).to_dict()} (3/3 fold'da iyi)\n")
    if not ok:
        lines.append("- Uc fold'un tamaminda iyilestiren varyant YOK.\n")

    (C.REPORTS_DIR / "34_gate_grid.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().sort_values().head(10).to_string())


if __name__ == "__main__":
    main()
