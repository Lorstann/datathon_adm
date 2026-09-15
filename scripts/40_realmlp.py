"""RealMLP (TabArena sinifi sinir agi) vs LightGBM + topluluk.

Neden bu deney: `reports/28_diversity.md` ve `32_level_diversity.md`'de
CatBoost ve XGBoost harmanlari LightGBM'i gecemedi. Ikisi de agac, yani
hatalari LightGBM ile yuksek korelasyonlu; topluluk ancak elemanlar
BAGIMSIZ hata yaptiginda kazandirir. Mimari olarak farkli bir model
(sinir agi) bu varsayimi ilk kez gercekten test ediyor.

Olculen sey tek basina skor DEGIL, harmanin kazandirip kazandirmadigi ve
hata korelasyonu.

NaN: agac dallanir, sinir agi dallanamaz. Bu projede NaN anlamli (bos
`hist_*` = cold trafo), o yuzden `neural_prep` once eksiklik gostergesi
uretip sonra medyanla dolduruyor.

Kullanim: python scripts/40_realmlp.py
"""

from __future__ import annotations

import os
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
from src.models.neural_prep import apply_prep, fit_prep
from src.models.zero_gate import dead_probability, shrink_level
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

_cv = import_module("20_multiorigin_cv")

GATE_K = 0.75
MLP_ROWS = int(os.environ.get("MLP_ROWS", 400_000))
MLP_EPOCHS = int(os.environ.get("MLP_EPOCHS", 48))
# NOT: anahtar bicimi .2f olmali; .1f ile 0.05/0.10/0.15 ayni anahtara
# cokup birbirini eziyordu.
BLEND_W = (0.05, 0.10, 0.15, 0.20, 0.30)
OUT_NAME = os.environ.get("OUT_NAME", "40_realmlp")


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    from pytabkit import RealMLP_TD_Regressor

    records, lines = [], ["# RealMLP vs LightGBM + topluluk\n", f"Uretim: {pd.Timestamp.now()}\n"]

    for fold in C.FOLDS:
        split = make_fold(train, fold, profile, seed=C.SEED)
        p_table = dead_probability(split.train, fold.origin)
        tr_f = build_training_frame(
            split.train, end_cap=fold.origin, static=static, weather=weather,
            max_rows=_cv.MAX_ROWS,
        )
        tr_f = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED,
                         entry_offsets=profile.entry_offsets)
        va_f = _cv.valid_frame(split, static, weather)
        va_f = va_f.loc[va_f[C.TARGET].notna()]

        y = va_f[C.TARGET].to_numpy()
        seg = split.segment.reindex(va_f.index)
        cold_s = split.is_cold.reindex(va_f.index)
        p_row = va_f[C.LOCATION].map(p_table["by_lokasyon"]).fillna(p_table["global"])
        p_row = np.where(cold_s.to_numpy(), p_row.to_numpy(), 0.0)

        ytr = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
        lvl_tr, lvl_va = tr_f[LEVEL_COL].to_numpy(), va_f[LEVEL_COL].to_numpy()
        resid_tr = ytr - lvl_tr
        Xtr, cat = model_matrix(tr_f)
        Xva, _ = model_matrix(va_f)
        Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
        Xva = Xva[Xtr.columns]
        hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]

        t0 = time.time()
        m = fit_shape_model(Xtr, ytr, lvl_tr, cat, hist_cols, seed=C.SEED + 3, n_trees=600)
        lp_lgbm = np.log1p(np.clip(predict_level_shape(m, Xva, lvl_va), 0, None))
        print(f"{fold.name} lgbm {time.time()-t0:.0f}s", flush=True)

        # RealMLP: alt orneklem (CPU), ayni artik hedef
        rng = np.random.default_rng(C.SEED)
        idx = rng.choice(len(Xtr), size=min(MLP_ROWS, len(Xtr)), replace=False)
        prep = fit_prep(Xtr, cat)
        Xn_tr = apply_prep(Xtr.iloc[idx], prep)
        Xn_va = apply_prep(Xva, prep)
        t0 = time.time()
        mlp = RealMLP_TD_Regressor(
            n_epochs=MLP_EPOCHS, device="cpu", verbosity=0, random_state=C.SEED
        )
        mlp.fit(Xn_tr, resid_tr[idx])
        r = np.asarray(mlp.predict(Xn_va), dtype="float64").ravel()
        lp_mlp = np.clip(lvl_va + r, 0.0, None)
        print(f"{fold.name} realmlp {time.time()-t0:.0f}s (n={len(idx):,})", flush=True)

        a = np.log1p(y)
        e1, e2 = lp_lgbm - a, lp_mlp - a
        corr = float(np.corrcoef(e1, e2)[0, 1])

        preds = {"lgbm": lp_lgbm, "realmlp": lp_mlp}
        for w in BLEND_W:
            preds[f"blend_{w:.2f}"] = (1 - w) * lp_lgbm + w * lp_mlp

        lines.append(
            f"\n## Fold {fold.name}  (hata korelasyonu lgbm-realmlp: **{corr:.4f}**)\n\n"
        )
        lines.append("| model | warm | cold | blend |\n| --- | --- | --- | --- |\n")
        for name, lp in preds.items():
            pred = shrink_level(np.expm1(lp), p_row, strength=GATE_K)
            met = segment_report(y, pred, seg, is_cold=cold_s)
            records.append({"model": name, "fold": fold.name, "corr": corr, **met})
            lines.append(
                "| {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                    name, met["rmsle_warm"], met["rmsle_cold"], met["rmsle_blend"]
                )
            )
            print(" ", name, round(met["rmsle_blend"], 4), flush=True)
        del tr_f, va_f, Xtr, Xva, Xn_tr, Xn_va

    df = pd.DataFrame(records)
    lines.append("\n## Ozet (rmsle_blend)\n\n| model | mean | std |\n| --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append(f"| {model} | {s['mean']:.4f} | {s['std']:.4f} |\n")
    piv = df.pivot(index="fold", columns="model", values="rmsle_blend")
    lines.append("\n### Fold bazinda lgbm farki (negatif = iyi)\n\n")
    for c in piv.columns:
        lines.append(f"- {c}: {(piv[c]-piv['lgbm']).round(4).to_dict()}\n")
    (C.REPORTS_DIR / f"{OUT_NAME}.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().sort_values().to_string())


if __name__ == "__main__":
    main()
