"""Cok-origin seviye+sekil gonderimi. python scripts/22_submit_multiorigin.py

Egitim satirlari test satirlariyla ayni sekilde uretilir: her origin icin
gecmis ozetleri donar, sonraki 122 gun egitim satiri olur. Boylece `hist_*`
kolonlari ve `horizon_day` egitimde ve tahminde ayni anlami tasir.
"""

from __future__ import annotations

import sys
import time
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
from src.features.peer import peer_tables
from src.models.level_shape import (
    POS_LEVEL_CHAIN,
    WARM_LEVEL_CHAIN,
    entity_level,
    fit_shape_model,
    predict_level_shape,
)
from src.features.history import entity_history_features
from src.models.multiorigin import (
    LEVEL_COL,
    build_training_frame,
    frozen_history,
    mask_cold,
    training_origins,
)
from src.models.zero_gate import dead_probability, shrink_level
from src.validation.folds import build_cold_profile

N_TREES = 600  # 1500 agac CV blend'i iyilestirmedi (reports/21_mo_tune.md)
MAX_ROWS = 1_800_000
SEEDS = (C.SEED + 3, C.SEED + 13, C.SEED + 23)
# Seviye cesitliligi (m7 + m28 capalari ortalamasi) LB'de OLCULDU ve
# REDDEDILDI: 1.06713 -> 1.06929. CV'de de fold C isaret degistiriyordu.
LEVEL_CHAINS = {"m7": WARM_LEVEL_CHAIN}

# Mevsim-esli agirlik. Ufkun tamami Nis-Tem ama egitim satirlarinin yalnizca
# %19,6'sinin ufku o aylara denk geliyor. CV bunu OLCEMEZ: fold'larin egitim
# ufuklari tanim geregi dogrulama aylarindan once bitiyor (fold A/C'de
# mevsim-ici oran %0, fold B'de %2,9). Bu yuzden LB ile sinaniyor.
# SONUC: LB'de REDDEDILDI (1.06713 -> 1.07042). Yani modelin daha duz olan
# mevsimsel rampasi dogru; 2026 yazi 2025'ten serin (Tem CDD 5.08 vs 7.54)
# ve model bunu zaten hava degiskenlerinden okuyor. Agirlik 1.0'da birakildi.
SEASON_MONTHS = (4, 5, 6, 7)
SEASON_WEIGHT = 1.0
# Cold olu-trafo kapisi. k=0.75 uc fold'un tamaminda kazandiriyor
# (-0.019 / -0.003 / -0.013, reports/24_zero_shrink.md).
GATE_K = 0.75
# Warm satirlarda iki asamali mimari (reports/02 karar 3, reports/50_two_stage.md).
# LB'de SERT RET: 1.06713 -> 1.08097 (+0.0138), CV -0.0041 demesine ragmen.
# Projedeki en buyuk CV/LB sapmasi. p tahmini suclu degil: test warm
# satirlarinda hist_zero_28 ortalamasi 0.0431, fold'larda 0.0448/0.0425/0.0284
# ve fold'larda gercek sifir oraniyla ortusuyor. CV kazanci fold A (-0.0101) ve
# B (-0.0047) kaynakliydi, fold C zaten (+0.0023) karsi cikiyordu.
TWO_STAGE_WARM = False
OUT = C.SUBMISSIONS_DIR / "submission_multiorigin.csv"


def main() -> None:
    C.ensure_dirs()
    t0 = time.time()
    train, test, sample = load_train(), load_test(), load_sample_submission()
    static = entity_static(build_panel(train, test))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None
    origin = C.TRAIN_END

    origins = training_origins(origin, max_origins=8)
    print("origins:", [str(o.date()) for o in origins])
    tr_f = build_training_frame(
        train,
        end_cap=origin,
        static=static,
        weather=weather,
        origins=origins,
        max_rows=MAX_ROWS,
    )
    # Cold taklidi yalnizca gecmisi silmekle olmuyor: gercek cold trafo ufkun
    # ortasinda devreye giriyor (ofset medyani 40 gun) ve transduktif kolonlar
    # bunu ele veriyor. Ampirik giris ofsetleri gercek test cold profilinden.
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    tr_f = mask_cold(
        tr_f,
        tr_f[C.ENTITY],
        seed=C.SEED,
        entry_offsets=profile.entry_offsets,
    )
    print(f"egitim satiri: {len(tr_f):,}  ({time.time()-t0:.0f}s)")

    # frozen_history, entity_history_features DEGIL: hava duyarliligi kolonlari
    # egitim cercevesinde var; burada da olmali, yoksa align onlari NaN'a
    # cevirir ve egitim/tahmin arasinda yeniden ayrisma olusur.
    hist = frozen_history(train, origin, weather)
    peer = peer_tables(train, origin)
    te_f = assemble(
        test,
        origin=origin,
        history=hist,
        weather=weather,
        static=static,
        peer=peer,
        include_history=True,
    )
    te_f[LEVEL_COL] = entity_level(
        hist, te_f[C.ENTITY], te_f["log_guc"].to_numpy(), warm_col=WARM_LEVEL_CHAIN
    )

    y = np.log1p(tr_f[C.TARGET].clip(lower=0).to_numpy())
    lvl_tr = tr_f[LEVEL_COL].to_numpy()
    lvl_te = te_f[LEVEL_COL].to_numpy()
    Xtr, cat = model_matrix(tr_f)
    Xte, _ = model_matrix(te_f)
    Xtr, Xte = Xtr.align(Xte, join="outer", axis=1, fill_value=np.nan)
    Xte = Xte[Xtr.columns]
    hist_cols = [c for c in Xtr.columns if c.startswith("hist_")]

    # Tohum ve seviye ortalamasi log1p uzayinda: metrik orada L2, dolayisiyla
    # ortalama ham olcekte degil log uzayinda alinmali.
    is_cold_te = ~te_f[C.ENTITY].isin(set(train[C.ENTITY].unique())).to_numpy()
    hist_by_origin = {
        o: entity_history_features(train, o) for o in tr_f["_origin"].unique()
    }
    is_masked_tr = tr_f["hist_n"].isna().to_numpy()
    cold_lvl_tr = tr_f["_cold_level"].to_numpy()

    acc = np.zeros(len(Xte))
    n_models = 0
    for cname, chain in LEVEL_CHAINS.items():
        parts = [
            pd.Series(
                entity_level(
                    hist_by_origin[o],
                    g[C.ENTITY],
                    g["log_guc"].to_numpy(),
                    warm_col=chain,
                ),
                index=g.index,
            )
            for o, g in tr_f.groupby("_origin", sort=False)
        ]
        lvl_tr_c = pd.concat(parts).reindex(tr_f.index).to_numpy()
        lvl_tr_c = np.where(is_masked_tr, cold_lvl_tr, lvl_tr_c)
        lvl_te_c = entity_level(
            hist, te_f[C.ENTITY], te_f["log_guc"].to_numpy(), warm_col=chain
        )
        # Lokasyon ofseti LB'de reddedildi (1.06713 -> 1.06766); uygulanmiyor.
        sw = None
        if SEASON_WEIGHT != 1.0:
            sw = np.where(
                tr_f["month"].isin(SEASON_MONTHS).to_numpy(), SEASON_WEIGHT, 1.0
            )
        for s in SEEDS:
            m = fit_shape_model(
                Xtr, y, lvl_tr_c, cat, hist_cols, seed=s, n_trees=N_TREES,
                weight=sw,
            )
            acc += np.log1p(predict_level_shape(m, Xte, lvl_te_c))
            n_models += 1
            print(f"  {cname} seed {s} bitti ({time.time()-t0:.0f}s)")
    pred = np.clip(np.expm1(acc / n_models), 0.0, None)

    if TWO_STAGE_WARM:
        # Sifirsiz seviye: hem egitimde (origin bazli) hem test'te
        parts = [
            pd.Series(
                entity_level(
                    hist_by_origin[o], g[C.ENTITY], g["log_guc"].to_numpy(),
                    warm_col=POS_LEVEL_CHAIN,
                ),
                index=g.index,
            )
            for o, g in tr_f.groupby("_origin", sort=False)
        ]
        lvl_pos_tr = pd.concat(parts).reindex(tr_f.index).to_numpy()
        lvl_pos_tr = np.where(is_masked_tr, cold_lvl_tr, lvl_pos_tr)
        lvl_pos_te = entity_level(
            hist, te_f[C.ENTITY], te_f["log_guc"].to_numpy(),
            warm_col=POS_LEVEL_CHAIN,
        )
        nz = tr_f[C.TARGET].to_numpy() > 0
        acc2 = np.zeros(len(Xte))
        for s in SEEDS:
            m2 = fit_shape_model(
                Xtr[nz], y[nz], lvl_pos_tr[nz], cat, hist_cols,
                seed=s, n_trees=N_TREES,
            )
            acc2 += np.log1p(predict_level_shape(m2, Xte, lvl_pos_te))
            print(f"  iki-asamali seed {s} bitti ({time.time()-t0:.0f}s)")
        lp2 = acc2 / len(SEEDS)

        # p: warm icin son 28 gunun sifir orani (sifir durumu kalici,
        # ufuk oncesi/sonrasi korelasyon 0.938)
        pz = te_f["hist_zero_28"] if "hist_zero_28" in te_f.columns else te_f["hist_zero_rate"]
        pz = pd.Series(np.asarray(pz, dtype="float64")).fillna(0.0).to_numpy()
        pz = np.clip(pz, 0.0, 1.0)
        warm_pred = np.clip(np.expm1(lp2 * np.clip(1.0 - pz, 0.0, 1.0)), 0.0, None)
        pred = np.where(is_cold_te, pred, warm_pred)
        print(f"  warm satir {int((~is_cold_te).sum()):,} iki-asamali yola gecti")

    # Cold satirlarda seviyeyi olu-trafo olasiligiyla kucult. Warm satirlarda
    # p=0: durum zaten gecmisten okunuyor ve seviye onu tasiyor.
    is_cold = ~te_f[C.ENTITY].isin(set(train[C.ENTITY].unique())).to_numpy()
    p_table = dead_probability(train, origin)
    p_row = te_f[C.LOCATION].map(p_table["by_lokasyon"]).fillna(p_table["global"])
    p_row = np.where(is_cold, p_row.to_numpy(), 0.0)
    pred = shrink_level(pred, p_row, strength=GATE_K)
    print(
        f"cold satir {int(is_cold.sum()):,}  olu olasiligi global "
        f"{p_table['global']:.4f}  aralik {p_row[is_cold].min():.3f}-{p_row[is_cold].max():.3f}"
    )

    out = sample.copy()
    out[C.TARGET] = pred
    assert out["id"].equals(test["id"]), "id sirasi sample_submission ile uyusmuyor"
    out.to_csv(OUT, index=False)

    lp = np.log1p(pred)
    by_month = pd.Series(lp).groupby(test[C.DATE].dt.month.to_numpy()).median()
    print(f"yazildi {OUT}  ({time.time()-t0:.0f}s)")
    print("aylik medyan log1p:", by_month.round(3).to_dict())
    print("sifire yakin tahmin orani:", float((pred <= 0.5).mean()))


if __name__ == "__main__":
    main()
