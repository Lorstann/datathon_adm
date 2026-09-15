"""Cold uzman modeli: tek model vs warm/cold ayrik modeller.

`reports/11..13` router'lari reddetmisti ama o olcumler kirli `is_cold`
tanimiyla ve tek-adim egitim cercevesiyle yapilmisti. O zamandan beri metrik,
egitim cercevesi, seviye zinciri, cold giris zamanlamasi ve olu kapisi
degisti; karar tasinmaz.

Gerekce: cold satirlar test'in %22'si ama kare hatanin ~%45'i. Tek model
NaN dallanmasiyla ikisini birlikte ogreniyor; kapasitesinin cogunu warm
satirlar aliyor olabilir (egitimde warm ~%71).

Uc varyant:
  tek         : mevcut -- tum satirlar tek modelde
  ayrik       : warm modeli gecmisli satirlarda, cold modeli maskeli satirlarda
  ayrik_karma : cold tahmini = 0.5*(tek) + 0.5*(cold uzmani), log1p uzayinda

Kullanim: python scripts/48_cold_specialist.py
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


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    records = []
    lines = ["# Cold uzman modeli\n", f"Uretim: {pd.Timestamp.now()}\n"]

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
        cold = cold_s.to_numpy()
        p_row = va_f[C.LOCATION].map(p_table["by_lokasyon"]).fillna(p_table["global"])
        p_row = np.where(cold, p_row.to_numpy(), 0.0)

        ytr = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
        lvl_tr, lvl_va = tr_f[LEVEL_COL].to_numpy(), va_f[LEVEL_COL].to_numpy()
        Xtr, cat = model_matrix(tr_f)
        Xva, _ = model_matrix(va_f)
        Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
        Xva = Xva[Xtr.columns]
        hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]

        is_masked = tr_f["hist_n"].isna().to_numpy()
        print(f"{fold.name}: egitim {len(tr_f):,}, maskeli(cold) {is_masked.mean():.1%}, "
              f"valid cold {cold.mean():.1%}", flush=True)

        t0 = time.time()
        m_all = fit_shape_model(Xtr, ytr, lvl_tr, cat, hist_cols, seed=C.SEED + 3, n_trees=600)
        lp_all = np.log1p(np.clip(predict_level_shape(m_all, Xva, lvl_va), 0, None))
        print(f"  tek model {time.time()-t0:.0f}s", flush=True)

        t0 = time.time()
        m_warm = fit_shape_model(Xtr[~is_masked], ytr[~is_masked], lvl_tr[~is_masked],
                                 cat, hist_cols, seed=C.SEED + 3, n_trees=600)
        lp_warm = np.log1p(np.clip(predict_level_shape(m_warm, Xva, lvl_va), 0, None))
        m_cold = fit_shape_model(Xtr[is_masked], ytr[is_masked], lvl_tr[is_masked],
                                 cat, hist_cols, seed=C.SEED + 3, n_trees=600)
        lp_cold = np.log1p(np.clip(predict_level_shape(m_cold, Xva, lvl_va), 0, None))
        print(f"  ayrik modeller {time.time()-t0:.0f}s", flush=True)

        variants = {
            "tek": lp_all,
            "ayrik": np.where(cold, lp_cold, lp_warm),
            "ayrik_karma": np.where(cold, 0.5 * lp_all + 0.5 * lp_cold, lp_all),
            "warm_uzman_only": np.where(cold, lp_all, lp_warm),
        }

        lines.append(f"\n## Fold {fold.name}\n\n| variant | warm | cold | blend |\n")
        lines.append("| --- | --- | --- | --- |\n")
        for name, lp in variants.items():
            pred = shrink_level(np.expm1(lp), p_row, strength=GATE_K)
            met = segment_report(y, pred, seg, is_cold=cold_s)
            records.append({"model": name, "fold": fold.name, **met})
            lines.append("| {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                name, met["rmsle_warm"], met["rmsle_cold"], met["rmsle_blend"]))
            print(" ", name, round(met["rmsle_blend"], 4),
                  "cold", round(met["rmsle_cold"], 4), flush=True)
        del tr_f, va_f, Xtr, Xva

    df = pd.DataFrame(records)
    lines.append("\n## Ozet\n\n| variant | mean | std | warm | cold |\n")
    lines.append("| --- | --- | --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            model, s["mean"], s["std"], g["rmsle_warm"].mean(), g["rmsle_cold"].mean()))
    piv = df.pivot(index="fold", columns="model", values="rmsle_blend")
    pc = df.pivot(index="fold", columns="model", values="rmsle_cold")
    for c in piv.columns:
        lines.append(f"\n- {c} blend: {(piv[c] - piv['tek']).round(4).to_dict()}")
        lines.append(f"  cold: {(pc[c] - pc['tek']).round(4).to_dict()}")
    (C.REPORTS_DIR / "48_cold_specialist.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().to_string())


if __name__ == "__main__":
    main()
