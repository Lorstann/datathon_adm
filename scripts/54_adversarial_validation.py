"""Adversarial validation: train vs gercek test satirlarini ayirt eden siniflandirici.

Motivasyon: `reports/51_round5_summary.md` CV'nin artik guvenilmez oldugunu
kaydetti (deney 8: CV -0.0041, LB +0.0138) ve sapmanin kaynagini bulamadi.
Yarisma forumunda 201. sira ekibin bulgusu (offline mekanizmanin production'a
tasinabilirligini test eden W9 common-support/domain-classifier analizi,
AUC 0.9996) ayni siniftan bir kontrolu bizim panelde de calistirmayi
gerektiriyor: train (origin <= 2026-03-31) satirlariyla gercek test
satirlarini ayirt eden bir model kurulabiliyor mu, kurulabiliyorsa hangi
ozellik ayirt ediyor.

Yuksek AUC = train/test dagilim farki gercek ve buyuk; CV guvenilmezliginin
kaynagi olabilir. Dusuk AUC (~0.5) = train/test ayni dagilimdan, CV sapmasinin
kaynagi baska yerde.

Kullanim: python scripts/54_adversarial_validation.py
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import config as C
from src.data.load import build_panel, entity_static, load_test, load_train
from src.external.weather import CACHE_PATH, load_weather, weather_features
from src.features.build import assemble, model_matrix
from src.features.peer import peer_tables
from src.models.boosting import fit_lightgbm
from src.models.multiorigin import build_training_frame, frozen_history, mask_cold
from src.validation.folds import build_cold_profile

MAX_ROWS = 400_000  # AV icin tam 1.8M gerekmiyor, hizli kontrol yeterli
N_TREES = 300


def main() -> None:
    C.ensure_dirs()
    t0 = time.time()
    print(f"basladi ({time.time()-t0:.0f}s)", flush=True)
    train, test = load_train(), load_test()
    print(f"train/test yuklendi ({time.time()-t0:.0f}s)", flush=True)
    static = entity_static(build_panel(train, test))
    print(f"static hazir ({time.time()-t0:.0f}s)", flush=True)
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None
    print(f"hava hazir ({time.time()-t0:.0f}s)", flush=True)
    origin = C.TRAIN_END

    tr_f = build_training_frame(
        train, end_cap=origin, static=static, weather=weather,
        origins=None, max_rows=MAX_ROWS,
    )
    print(f"tr_f hazir: {len(tr_f):,} satir ({time.time()-t0:.0f}s)", flush=True)
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    tr_f = mask_cold(tr_f, tr_f[C.ENTITY], seed=C.SEED, entry_offsets=profile.entry_offsets)
    print(f"cold maskeleme bitti ({time.time()-t0:.0f}s)", flush=True)

    hist = frozen_history(train, origin, weather)
    peer = peer_tables(train, origin)
    print(f"hist/peer hazir ({time.time()-t0:.0f}s)", flush=True)
    te_f = assemble(
        test, origin=origin, history=hist, weather=weather, static=static,
        peer=peer, include_history=True,
    )
    print(f"tr_f={len(tr_f):,}  te_f={len(te_f):,}  ({time.time()-t0:.0f}s)", flush=True)

    Xtr, cat = model_matrix(tr_f)
    print(f"Xtr hazir ({time.time()-t0:.0f}s)", flush=True)
    Xte, _ = model_matrix(te_f)
    print(f"Xte hazir ({time.time()-t0:.0f}s)", flush=True)
    Xtr, Xte = Xtr.align(Xte, join="outer", axis=1, fill_value=np.nan)
    Xte = Xte[Xtr.columns]

    X = pd.concat([Xtr, Xte], axis=0, ignore_index=True)
    y = np.concatenate([np.zeros(len(Xtr)), np.ones(len(Xte))])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    oof = np.zeros(len(X))
    gains = np.zeros(len(X.columns))
    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        m = fit_lightgbm(
            X.iloc[tr_idx], y[tr_idx], cat, seed=C.SEED + fold_i, n_trees=N_TREES,
            mask_history_rate=0.0,
            params_override={"objective": "binary", "metric": "auc"},
        )
        oof[va_idx] = m.booster.predict(X.iloc[va_idx][m.feature_names])
        gains += m.booster.feature_importance(importance_type="gain")
        print(f"  fold {fold_i} bitti ({time.time()-t0:.0f}s)")

    auc = roc_auc_score(y, oof)
    gain_df = pd.DataFrame({"feature": X.columns, "gain": gains}).sort_values(
        "gain", ascending=False
    ).head(25)

    lines = [
        "# Adversarial validation: train vs gercek test\n",
        f"Uretim: {pd.Timestamp.now()}\n\n",
        f"tr_f={len(tr_f):,} satir (max_rows={MAX_ROWS:,}), te_f={len(te_f):,} satir. "
        f"5-fold OOF AUC.\n\n",
        f"## AUC: **{auc:.4f}**\n\n",
        "0.5 = train/test ayni dagilimdan, ayirt edilemiyor. "
        "1.0 = mukemmel ayirt ediliyor, train/test dagilim farki buyuk.\n\n",
        "## En cok ayirt eden 25 ozellik (gain)\n\n| feature | gain |\n| --- | --- |\n",
    ]
    for _, r in gain_df.iterrows():
        lines.append(f"| {r['feature']} | {r['gain']:.1f} |\n")
    (C.REPORTS_DIR / "54_adversarial_validation.md").write_text("".join(lines), encoding="utf-8")
    print(f"AUC={auc:.4f}")
    print(gain_df.to_string(index=False))
    print(f"yazildi: reports/54_adversarial_validation.md ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
