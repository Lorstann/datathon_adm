"""Cold seviyesine lokasyon ofseti: varyans ayristirmasinin isaret ettigi bosluk.

Varyans ayristirmasi (reports/43_variance.md):
  trafo seviyesi varyansinin R2'si  ->  guc 0.490 | lokasyon 0.279 | ikisi 0.577

Mevcut cold seviyemiz `global_z + log_guc`, yani yalnizca gucu kullaniyor.
Lokasyonun ekledigi 0.087'lik kisim tamamen bosta duruyor. Cold satirlar
test'in %22'si ve kare hatanin ~%45'i, dolayisiyla bu bosluk onemli.

`reports/02_cold_start_ceiling.md` lokasyon grup ortalamalarinin cold'u
KOTULESTIRDIGINI bulmustu, ama o olcum hem kirli `is_cold` tanimindan hem de
duzenlilestirilmemis grup ortalamasindan geliyordu (285-304 grup, dogrudan
ortalama). Burada ampirik Bayes ile globale cekilmis bir OFSET deneniyor ve
tek anahtar lokasyon (47 deger), lokasyon x guc_band (300+) degil.

Kullanim: python scripts/42_cold_lokasyon_level.py
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
from src.features.history import entity_history_features
from src.models.level_shape import fit_shape_model, predict_level_shape
from src.models.multiorigin import LEVEL_COL, build_training_frame, mask_cold
from src.models.zero_gate import dead_probability, shrink_level
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

_cv = import_module("20_multiorigin_cv")
GATE_K = 0.75
PRIORS = (0.0, 20.0, 50.0, 150.0)  # 0.0 = ofset yok (mevcut davranis)


def lok_offset_table(hist: pd.DataFrame, prior: float) -> tuple[float, pd.Series]:
    """Lokasyon bazinda z ofseti, globale ampirik Bayes ile cekilmis.

    Yalnizca CANLI trafolardan (sifir orani dusuk) kurulur: olu trafonun z'si
    ~ -log(guc) ve ofseti bozar.
    """
    z = hist["hist_mean_z"]
    ok = z.notna()
    if "hist_zero_rate" in hist.columns:
        ok &= hist["hist_zero_rate"] < 0.1
    h = hist.loc[ok, ["hist_mean_z", "lokasyon"]].dropna()
    if h.empty:
        return 0.0, pd.Series(dtype="float64")
    glob = float(h["hist_mean_z"].median())
    g = h.groupby("lokasyon")["hist_mean_z"].agg(["median", "size"])
    # n kucukken globale cek
    off = (g["median"] - glob) * g["size"] / (g["size"] + prior)
    return glob, off


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    records, lines = [], ["# Cold seviyesine lokasyon ofseti\n", f"Uretim: {pd.Timestamp.now()}\n"]

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
        hist_va = entity_history_features(split.train, fold.origin)
        hist_by_origin = {
            o: entity_history_features(split.train, o) for o in tr_f["_origin"].unique()
        }

        y = va_f[C.TARGET].to_numpy()
        seg = split.segment.reindex(va_f.index)
        cold_s = split.is_cold.reindex(va_f.index)
        cold = cold_s.to_numpy()
        p_row = va_f[C.LOCATION].map(p_table["by_lokasyon"]).fillna(p_table["global"])
        p_row = np.where(cold, p_row.to_numpy(), 0.0)
        ytr = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
        Xtr, cat = model_matrix(tr_f)
        Xva, _ = model_matrix(va_f)
        Xtr, Xva = Xtr.align(Xva, join="outer", axis=1, fill_value=np.nan)
        Xva = Xva[Xtr.columns]
        hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]

        is_masked = tr_f["hist_n"].isna().to_numpy()
        base_lvl_tr = tr_f[LEVEL_COL].to_numpy()
        base_lvl_va = va_f[LEVEL_COL].to_numpy()

        lines.append(f"\n## Fold {fold.name}\n\n| prior | warm | cold | blend |\n")
        lines.append("| --- | --- | --- | --- |\n")

        for prior in PRIORS:
            t0 = time.time()
            lvl_tr = base_lvl_tr.copy()
            lvl_va = base_lvl_va.copy()
            if prior > 0:
                # egitim: her origin kendi tablosuyla, yalnizca cold satirlara
                add = np.zeros(len(tr_f))
                for o, g in tr_f.groupby("_origin", sort=False):
                    _, off = lok_offset_table(hist_by_origin[o], prior)
                    add[g.index.to_numpy()] = (
                        g[C.LOCATION].map(off).fillna(0.0).to_numpy()
                    )
                lvl_tr = np.where(is_masked, base_lvl_tr + add, base_lvl_tr)
                _, off_va = lok_offset_table(hist_va, prior)
                add_va = va_f[C.LOCATION].map(off_va).fillna(0.0).to_numpy()
                lvl_va = np.where(cold, base_lvl_va + add_va, base_lvl_va)

            m = fit_shape_model(Xtr, ytr, lvl_tr, cat, hist_cols,
                                seed=C.SEED + 3, n_trees=600)
            pred = shrink_level(predict_level_shape(m, Xva, lvl_va), p_row,
                                strength=GATE_K)
            met = segment_report(y, pred, seg, is_cold=cold_s)
            records.append({"model": f"prior{prior:.0f}", "fold": fold.name, **met})
            lines.append("| {:.0f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                prior, met["rmsle_warm"], met["rmsle_cold"], met["rmsle_blend"]))
            print(fold.name, prior, round(met["rmsle_blend"], 4),
                  "cold", round(met["rmsle_cold"], 4), f"{time.time()-t0:.0f}s", flush=True)
        del tr_f, va_f, Xtr, Xva

    df = pd.DataFrame(records)
    lines.append("\n## Ozet\n\n| prior | mean | std | cold |\n| --- | --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append(f"| {model} | {s['mean']:.4f} | {s['std']:.4f} | {g['rmsle_cold'].mean():.4f} |\n")
    piv = df.pivot(index="fold", columns="model", values="rmsle_blend")
    pc = df.pivot(index="fold", columns="model", values="rmsle_cold")
    for c in piv.columns:
        lines.append(f"\n- {c} blend farki: {(piv[c]-piv['prior0']).round(4).to_dict()}")
        lines.append(f"  cold farki: {(pc[c]-pc['prior0']).round(4).to_dict()}")
    (C.REPORTS_DIR / "42_cold_lokasyon.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().to_string())


if __name__ == "__main__":
    main()
