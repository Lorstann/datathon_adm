"""Cold sifir kapisi: olu-trafo olasiligiyla seviye kucultme.

`reports/23_error_decomp.md`: cold + sifir satirlar dogrulama satirlarinin
%0,8-1,7'si ama kare hatanin **%42-51'i**. Multiorigin C bu satirlarda
RMSLE 6.7, yani gercek 0 iken ~800 kWh tahmin ediyor.

Sifirlar satir duzeyinde olay degil, varlik duzeyinde durum: panele sonradan
giren 2.383 trafonun 2.130'unda hic sifir yok, 132'sinde satirlarin >%90'i
sifir; ara bolge nerdeyse bos. Yani sorun "bugun sifir mi" degil, "bu trafo
olu mu".

L2 log1p uzayinda oldugu icin, olu olma olasiligi p olan bir satirin optimal
tahmini `(1-p) * seviye`: karisimin beklenen degeri. Cold trafo icin p'yi
gecmisten okuyamayiz, ama train'e sonradan giren trafolarin olu oranindan
lokasyon bazinda kestirebiliriz (0.000 - 0.234 arasi degisiyor, genel 0.055).

Kullanim: python scripts/24_zero_shrink.py
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
from src.models.zero_gate import dead_probability, shrink_level
from src.validation.folds import build_cold_profile
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

_cv = import_module("20_multiorigin_cv")

STRENGTHS = (0.0, 0.5, 0.75, 1.0, 1.25)


def main() -> None:
    C.ensure_dirs()
    train, test = load_train(), load_test()
    static = entity_static(build_panel(train, test))
    profile = build_cold_profile(static, set(train[C.ENTITY].unique()))
    weather = weather_features(load_weather()) if CACHE_PATH.exists() else None

    records = []
    lines = ["# Cold olu-trafo kapisi\n", f"Uretim: {pd.Timestamp.now()}\n"]

    for fold in C.FOLDS:
        split = make_fold(train, fold, profile, seed=C.SEED)
        pred, va, _m, n_tr = _cv.new_path(split, static, weather)
        y = va[C.TARGET].to_numpy()
        seg = split.segment.reindex(va.index)
        cold = split.is_cold.reindex(va.index).to_numpy()

        # p, YALNIZCA origin oncesi veriden: panele sonradan giren trafolarin
        # olu orani. Dogrulama penceresi hic gorulmuyor.
        p_table = dead_probability(split.train, fold.origin)
        p_row = va[C.LOCATION].map(p_table["by_lokasyon"]).fillna(p_table["global"])
        p_row = np.where(cold, p_row.to_numpy(), 0.0)

        lines.append(f"\n## Fold {fold.name}  (cold satir {cold.sum():,})\n\n")
        lines.append(
            f"global olu orani {p_table['global']:.4f}, "
            f"lokasyon araligi {min(p_table['by_lokasyon']):.3f}-"
            f"{max(p_table['by_lokasyon']):.3f}\n\n"
        )
        lines.append("| strength | warm | cold | blend |\n| --- | --- | --- | --- |\n")

        for k in STRENGTHS:
            pk = shrink_level(pred, p_row, strength=k)
            m = segment_report(y, pk, seg, is_cold=split.is_cold.reindex(va.index))
            records.append({"model": f"k{k}", "fold": fold.name, **m})
            lines.append(
                "| {:.2f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                    k, m["rmsle_warm"], m["rmsle_cold"], m["rmsle_blend"]
                )
            )
            print(fold.name, k, round(m["rmsle_blend"], 4), flush=True)

        # Tavan: p'yi bilseydik (ayni fold'un gercegi) ne olurdu
        true_dead = (
            pd.Series(y <= 0, index=va.index)
            .groupby(va[C.ENTITY].to_numpy())
            .transform("mean")
            .to_numpy()
        )
        oracle = shrink_level(pred, np.where(cold, true_dead, 0.0), strength=1.0)
        m = segment_report(y, oracle, seg, is_cold=split.is_cold.reindex(va.index))
        lines.append(
            "| KAHIN p (tavan) | {:.4f} | {:.4f} | {:.4f} |\n".format(
                m["rmsle_warm"], m["rmsle_cold"], m["rmsle_blend"]
            )
        )

    df = pd.DataFrame(records)
    lines.append("\n## Ozet (rmsle_blend)\n\n| strength | mean | std | cold |\n")
    lines.append("| --- | --- | --- | --- |\n")
    for model, g in df.groupby("model"):
        s = summarize_folds(g, "rmsle_blend")
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                model, s["mean"], s["std"], g["rmsle_cold"].mean()
            )
        )
    piv = df.pivot(index="fold", columns="model", values="rmsle_blend")
    base = piv["k0.0"]
    lines.append("\n### Fold bazinda k0 farki (negatif = iyi)\n\n")
    for col in piv.columns:
        d = (piv[col] - base).round(4).to_dict()
        lines.append(f"- {col}: {d}\n")

    (C.REPORTS_DIR / "24_zero_shrink.md").write_text("".join(lines), encoding="utf-8")
    print(df.groupby("model")["rmsle_blend"].mean().sort_values().to_string())


if __name__ == "__main__":
    main()
