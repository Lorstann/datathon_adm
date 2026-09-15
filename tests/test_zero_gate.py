"""Olu-trafo kapisinin degismezleri."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.models.zero_gate import dead_probability, shrink_level


def test_shrink_is_applied_in_log_space():
    """RMSLE log1p uzayinda L2; kucultme de orada olmali, ham olcekte degil."""
    pred = np.array([np.expm1(8.0)])
    out = shrink_level(pred, np.array([0.25]), strength=1.0)
    assert np.isclose(np.log1p(out[0]), 6.0)  # 0.75 * 8, ham olcekte 0.75x DEGIL
    assert not np.isclose(out[0], 0.75 * pred[0])


def test_shrink_bounds_and_neutral_cases():
    pred = np.array([100.0, 100.0, 100.0])
    assert np.allclose(shrink_level(pred, np.zeros(3), strength=1.0), pred)
    assert np.allclose(shrink_level(pred, np.ones(3) * 0.5, strength=0.0), pred)
    # p=1 tamamen sifirlar, asiri strength negatife tasmaz
    assert shrink_level(pred, np.ones(3), strength=1.0)[0] == 0.0
    assert shrink_level(pred, np.ones(3), strength=5.0)[0] == 0.0
    assert (shrink_level(pred, np.ones(3) * 0.9, strength=3.0) >= 0).all()


def _panel(dead_locs, alive_locs, days=120):
    """Panel basindan sonra giren trafolar; bir kismi tamamen sifir."""
    dates = pd.date_range(C.TRAIN_START, periods=days, freq="D")
    rows = []
    # kurulu taban: panel basindan beri var, referans populasyona girmemeli
    for i in range(5):
        for d in dates:
            rows.append((f"OLD{i}", 400, d, 0.0, alive_locs[0]))
    for lok, n_dead, n_alive in dead_locs:
        for k in range(n_dead):
            for d in dates[40:]:
                rows.append((f"D_{lok}_{k}", 400, d, 0.0, lok))
        for k in range(n_alive):
            for d in dates[40:]:
                rows.append((f"A_{lok}_{k}", 400, d, 500.0, lok))
    return pd.DataFrame(
        rows, columns=[C.ENTITY, C.POWER, C.DATE, C.TARGET, C.LOCATION]
    )


def test_dead_probability_separates_locations_and_shrinks_to_global():
    panel = _panel(
        dead_locs=[("İZMİR>A", 20, 20), ("İZMİR>B", 0, 40)],
        alive_locs=["İZMİR>B"],
    )
    out = dead_probability(panel, panel[C.DATE].max())
    by = out["by_lokasyon"]

    assert by["İZMİR>A"] > out["global"] > by["İZMİR>B"]
    # ampirik Bayes: ham oran 0.5 ve 0.0 idi, ikisi de globale cekilmis olmali
    assert by["İZMİR>A"] < 0.5
    assert by["İZMİR>B"] > 0.0


def test_dead_probability_ignores_post_origin_rows():
    """Origin sonrasi hicbir hedef p tahminine giremez."""
    panel = _panel(dead_locs=[("İZMİR>A", 20, 20)], alive_locs=["İZMİR>A"])
    origin = panel[C.DATE].max()
    full = dead_probability(panel, origin)

    future = panel.copy()
    future.loc[future[C.LOCATION] == "İZMİR>A", C.TARGET] = 0.0
    future = pd.concat(
        [panel, future.assign(**{C.DATE: future[C.DATE] + pd.Timedelta(days=200)})],
        ignore_index=True,
    )
    assert dead_probability(future, origin)["global"] == full["global"]


def test_xgboost_keeps_categorical_features():
    """`fit_xgboost` kategorikleri dusurmemeli: lokasyon_cat gain'de ilk ucte."""
    import numpy as np

    from src.models.boosting import fit_xgboost

    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "lokasyon_cat": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
        }
    )
    # hedef yalnizca kategoriden turer; kategori dusurulurse ogrenilemez
    y = X["lokasyon_cat"].cat.codes.to_numpy().astype(float) * 3.0

    m = fit_xgboost(X, y, ["lokasyon_cat"], seed=0, n_trees=60)
    assert "lokasyon_cat" in m.feature_names
    pred = m.booster.predict(
        __import__("xgboost").DMatrix(
            X.assign(lokasyon_cat=X["lokasyon_cat"].astype("category")),
            enable_categorical=True,
        )
    )
    assert np.corrcoef(pred, y)[0, 1] > 0.95

    # kategori verilmezse eski davranis: kolon yok
    m2 = fit_xgboost(X, y, seed=0, n_trees=10)
    assert "lokasyon_cat" not in m2.feature_names


def test_lok_offset_is_shrunk_and_excludes_dead():
    """Lokasyon ofseti globale cekilmeli ve olu trafolardan etkilenmemeli."""
    from src.models.level_shape import lok_offset_table

    # A: 60 canli trafo, z'leri yuksek. B: 3 canli trafo, z'leri ayni kadar
    # yuksek ama az gozlem -> daha cok cekilmeli.
    # C cogunlukta olmali ki global medyani o belirlesin (0.0)
    rows = (
        [{"hist_mean_z": 2.0, "hist_zero_rate": 0.0, "lokasyon": "A"}] * 60
        + [{"hist_mean_z": 2.0, "hist_zero_rate": 0.0, "lokasyon": "B"}] * 3
        + [{"hist_mean_z": 0.0, "hist_zero_rate": 0.0, "lokasyon": "C"}] * 200
    )
    hist = pd.DataFrame(rows)

    off = lok_offset_table(hist, prior=50.0)
    assert off["A"] > off["B"] > 0, "az gozlemli lokasyon daha cok cekilmeli"
    assert off["A"] < 2.0, "ham fark kadar buyuk olmamali (cekilme)"

    # Olu trafolar (z cok dusuk, zero_rate yuksek) ofseti bozmamali
    dead = pd.concat(
        [hist, pd.DataFrame([{"hist_mean_z": -9.0, "hist_zero_rate": 1.0,
                              "lokasyon": "A"}] * 30)],
        ignore_index=True,
    )
    assert np.isclose(lok_offset_table(dead, prior=50.0)["A"], off["A"])
