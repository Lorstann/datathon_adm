"""Cold taklidi: gecmisi silmek yetmiyor, GIRIS ZAMANLAMASI da taklit edilmeli.

Olculen uyusmazlik (gercek test cold satirlari / duzeltilmeden onceki egitim
taklidi):

    devreye_alinma_yasi  medyan   40 / 202
    test_gun_sayisi      medyan   82 / 122
    horizon_day          medyan   82 /  51

Gercek cold trafolarin %82'si 2026-05'te panele giriyor, yani ufkun ortasinda.
Egitimde ise maskelenen trafo ufkun basindan beri varmis gibi gorunuyordu:
model "gecmis yok + yasli" ogrenip tahminde "gecmis yok + genc" ile
karsilasiyordu. Egitimde genc satirlarin tamaminin gecmisi vardi, yani bu
bolge agac icin bos.

Duzeltme `mask_cold`'a ampirik giris ofseti ekler: giristen onceki satirlar
dusulur, yas ve test_gun_sayisi giristen itibaren yeniden yazilir. Ayni
duzeltme dogrulama tarafinda da yapilir (maskelenen trafolarin transduktif
kolonlari orada da gercek yasi tasiyordu).

Kullanim: python scripts/37_cold_entry_sim.py
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
    lines = ["# Cold giris zamanlamasi taklidi\n", f"Uretim: {pd.Timestamp.now()}\n"]

    for fold in C.FOLDS:
        split = make_fold(train, fold, profile, seed=C.SEED)
        p_table = dead_probability(split.train, fold.origin)
        tr_base = build_training_frame(
            split.train,
            end_cap=fold.origin,
            static=static,
            weather=weather,
            max_rows=_cv.MAX_ROWS,
        )
        va_f = _cv.valid_frame(split, static, weather)
        va_f = va_f.loc[va_f[C.TARGET].notna()]

        y = va_f[C.TARGET].to_numpy()
        seg = split.segment.reindex(va_f.index)
        cold_s = split.is_cold.reindex(va_f.index)
        cold = cold_s.to_numpy()
        p_row = va_f[C.LOCATION].map(p_table["by_lokasyon"]).fillna(p_table["global"])
        p_row = np.where(cold, p_row.to_numpy(), 0.0)
        Xva_full, _ = model_matrix(va_f)
        lvl_va = va_f[LEVEL_COL].to_numpy()

        lines.append(f"\n## Fold {fold.name}\n\n| variant | egitim satiri | warm | cold | blend |\n")
        lines.append("| --- | --- | --- | --- | --- |\n")

        for name, kw in (
            ("no_entry_sim", {}),
            ("entry_sim", {"entry_offsets": profile.entry_offsets}),
        ):
            t0 = time.time()
            tr_f = mask_cold(tr_base, tr_base[C.ENTITY], seed=C.SEED, **kw)
            ytr = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
            lvl_tr = tr_f[LEVEL_COL].to_numpy()
            Xtr, cat = model_matrix(tr_f)
            Xtr, Xva = Xtr.align(Xva_full, join="outer", axis=1, fill_value=np.nan)
            Xva = Xva[Xtr.columns]
            hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]
            m = fit_shape_model(
                Xtr, ytr, lvl_tr, cat, hist_cols, seed=C.SEED + 3, n_trees=600
            )
            pred = shrink_level(
                predict_level_shape(m, Xva, lvl_va), p_row, strength=GATE_K
            )
            met = segment_report(y, pred, seg, is_cold=cold_s)
            records.append({"model": name, "fold": fold.name, **met})
            lines.append(
                "| {} | {:,} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                    name, len(tr_f), met["rmsle_warm"], met["rmsle_cold"],
                    met["rmsle_blend"],
                )
            )
            print(fold.name, name, round(met["rmsle_blend"], 4),
                  "cold", round(met["rmsle_cold"], 4), f"{time.time()-t0:.0f}s", flush=True)
            del tr_f, Xtr, Xva
        del tr_base, va_f, Xva_full

    df = pd.DataFrame(records)
    lines.append("\n## Ozet (rmsle_blend)\n\n| variant | mean | std | warm | cold |\n")
    lines.append("| --- | --- | --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                model, s["mean"], s["std"], g["rmsle_warm"].mean(), g["rmsle_cold"].mean()
            )
        )
    piv = df.pivot(index="fold", columns="model", values="rmsle_blend")
    pc = df.pivot(index="fold", columns="model", values="rmsle_cold")
    lines.append(
        f"\nblend farki (entry_sim - no): {(piv['entry_sim']-piv['no_entry_sim']).round(4).to_dict()}\n"
    )
    lines.append(
        f"\ncold farki (entry_sim - no): {(pc['entry_sim']-pc['no_entry_sim']).round(4).to_dict()}\n"
    )
    (C.REPORTS_DIR / "37_cold_entry_sim.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().to_string())


if __name__ == "__main__":
    main()
