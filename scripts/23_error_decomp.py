"""Multiorigin C'nin kalan hatasini nereye harcadigini olcer.

Kare hata (log1p uzayi) kirilimlari: cold/warm x sifir/sifir-degil, varlik
yogunlasmasi, ufuk, ay. Hangi cephenin gercekten buyuk oldugunu tahmin etmek
yerine olcmek icin; sonraki denemenin hedefini bu belirler.

Kullanim: python scripts/23_error_decomp.py
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
from src.validation.folds import build_cold_profile
from src.validation.split import make_fold

_cv = import_module("20_multiorigin_cv")

FOLDS = ("B_guncel", "C_ara")


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    lines = ["# Multiorigin C hata kirilimi\n", f"Uretim: {pd.Timestamp.now()}\n"]
    lines.append(
        "\nButun paylar **kare hata** (log1p uzayi) uzerinden. RMSLE'lerin "
        "kendisi toplanamaz; harcanan hatayi ancak kare uzayda bolusturebiliriz.\n"
    )

    for fold in [f for f in C.FOLDS if f.name in FOLDS]:
        split = make_fold(train, fold, profile, seed=C.SEED)
        pred, va, _model, n_tr = _cv.new_path(split, static, weather)

        y = va[C.TARGET].to_numpy()
        sq = (np.log1p(np.clip(pred, 0, None)) - np.log1p(y)) ** 2
        d = pd.DataFrame(
            {
                "sq": sq,
                "cold": split.is_cold.reindex(va.index).to_numpy(),
                "zero": y <= 0,
                "ent": va[C.ENTITY].to_numpy(),
                "hz": va["horizon_day"].to_numpy(),
                "month": va["month"].to_numpy(),
                "pred": np.clip(pred, 0, None),
                "y": y,
            }
        )
        tot = d["sq"].sum()

        lines.append(
            f"\n## Fold {fold.name}  (egitim satiri {n_tr:,}, dogrulama {len(d):,})\n"
        )
        lines.append(f"\nRMSLE all = {np.sqrt(d['sq'].mean()):.4f}\n")

        lines.append("\n### cold/warm x sifir/sifir-degil\n\n")
        lines.append("| dilim | satir | satir payi | kare hata payi | RMSLE |\n")
        lines.append("| --- | --- | --- | --- | --- |\n")
        for cold in (False, True):
            for zero in (False, True):
                s = d[(d.cold == cold) & (d.zero == zero)]
                if not len(s):
                    continue
                lines.append(
                    "| {} / {} | {:,} | {:.1%} | **{:.1%}** | {:.4f} |\n".format(
                        "cold" if cold else "warm",
                        "sifir" if zero else "sifir degil",
                        len(s),
                        len(s) / len(d),
                        s.sq.sum() / tot,
                        np.sqrt(s.sq.mean()),
                    )
                )

        lines.append("\n### Varlik yogunlasmasi\n\n")
        per = d.groupby("ent")["sq"].sum().sort_values(ascending=False)
        lines.append("| en kotu N varlik | kare hata payi |\n| --- | --- |\n")
        for n in (10, 20, 50, 100, 500):
            if n <= len(per):
                lines.append(f"| {n} / {len(per):,} | {per.head(n).sum()/tot:.1%} |\n")

        lines.append("\n### Ufka gore\n\n| ufuk | RMSLE | kare hata payi |\n| --- | --- | --- |\n")
        for lo, hi in ((1, 30), (31, 60), (61, 90), (91, 130)):
            s = d[d.hz.between(lo, hi)]
            if len(s):
                lines.append(
                    f"| {lo}-{hi} | {np.sqrt(s.sq.mean()):.4f} | {s.sq.sum()/tot:.1%} |\n"
                )

        lines.append("\n### Yanlilik (ortalama log1p artik = gercek - tahmin)\n\n")
        d["res"] = np.log1p(d.y) - np.log1p(d.pred)
        lines.append("| dilim | n | ortalama artik |\n| --- | --- | --- |\n")
        nz = d[~d.zero]
        for name, s in (
            ("warm, sifir degil", nz[~nz.cold]),
            ("cold, sifir degil", nz[nz.cold]),
        ):
            lines.append(f"| {name} | {len(s):,} | {s.res.mean():+.4f} |\n")
        by_m = nz.groupby("month")["res"].agg(["mean", "size"])
        for m, r in by_m.iterrows():
            lines.append(f"| ay {int(m)} (sifir degil) | {int(r['size']):,} | {r['mean']:+.4f} |\n")

        print(fold.name, "done", flush=True)

    path = C.REPORTS_DIR / "23_error_decomp.md"
    path.write_text("".join(lines), encoding="utf-8")
    print("yazildi:", path)


if __name__ == "__main__":
    main()
