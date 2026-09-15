"""Egitim ve tahmin cerceveleri ayni kolon kumesini uretmeli.

Bu projede en pahali iki hata da ayni kokten cikti: egitim tarafinda var olan
bir kolonun tahmin tarafinda olmamasi (ya da tersi). `align(join="outer")`
farki sessizce NaN ile doldurdugu icin model hic dallanmayan olu bir ozellikle
egitiliyor ya da tahminde hic gormedigi bir dagilimla karsilasiyor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.features.history import entity_history_features, entity_weather_sensitivity
from src.models.multiorigin import frozen_history

from tests.test_multiorigin import _panel, _static  # noqa: F401


def _weather(panel: pd.DataFrame) -> pd.DataFrame:
    days = panel[C.DATE].drop_duplicates().sort_values()
    frames = []
    for lok in panel[C.LOCATION].unique():
        f = pd.DataFrame({C.DATE: days})
        f[C.LOCATION] = lok
        t = 15 + 10 * np.sin(np.arange(len(days)) / 30.0)
        f["cdd"] = np.maximum(0.0, t - 18)
        f["hdd"] = np.maximum(0.0, 18 - t)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def test_frozen_history_adds_weather_sensitivity():
    panel = _panel()
    w = _weather(panel)
    origin = panel[C.DATE].max()

    plain = entity_history_features(panel, origin)
    withw = frozen_history(panel, origin, w)

    assert "hist_cdd_slope" not in plain.columns
    assert {"hist_cdd_slope", "hist_hdd_slope"} <= set(withw.columns)
    assert set(plain.columns) <= set(withw.columns)
    assert len(plain) == len(withw)


def test_frozen_history_column_set_is_origin_independent():
    """Farkli origin'ler ayni kolonlari uretmeli; egitim cercevesi onlari yigiyor."""
    panel = _panel()
    w = _weather(panel)
    a = frozen_history(panel, panel[C.DATE].max(), w)
    b = frozen_history(panel, panel[C.DATE].max() - pd.Timedelta(days=45), w)
    assert set(a.columns) == set(b.columns)


def test_weather_sensitivity_is_none_safe():
    panel = _panel()
    origin = panel[C.DATE].max()
    assert entity_weather_sensitivity(panel, origin, None).empty
    # hava yoksa frozen_history yine de calismali
    assert not frozen_history(panel, origin, None).empty


def test_new_history_features_present_and_finite_somewhere():
    panel = _panel()
    h = frozen_history(panel, panel[C.DATE].max(), _weather(panel))
    for col in (
        "hist_mom_7_28",
        "hist_mom_7_91",
        "hist_mom_28_91",
        "hist_iqr_91",
        "hist_dow_rel_0",
    ):
        assert col in h.columns, col
        assert h[col].notna().any(), col
    # IQR negatif olamaz
    assert (h["hist_iqr_91"].dropna() >= 0).all()


def test_seasonal_lag_never_reads_past_origin():
    """Aranan tarih daima origin'den once olmali; ayrica origin sonrasi
    satirlar kaynak olarak kullanilmamali."""
    from src.features.history import seasonal_lag_columns

    panel = _panel(days=400)
    origin = C.TRAIN_START + pd.Timedelta(days=300)
    horizon = pd.date_range(origin + pd.Timedelta(days=1), periods=122, freq="D")
    ents = panel[C.ENTITY].unique()[:4]
    frame = pd.DataFrame(
        [(e, d) for e in ents for d in horizon], columns=[C.ENTITY, C.DATE]
    )

    out = seasonal_lag_columns(frame, panel, origin)
    assert (frame[C.DATE] - pd.Timedelta(days=364) < origin).all()

    # Origin sonrasi hedefleri bozmak ciktiyi degistirmemeli
    tampered = panel.copy()
    tampered.loc[tampered[C.DATE] > origin, C.TARGET] = 1e9
    out2 = seasonal_lag_columns(frame, tampered, origin)
    pd.testing.assert_frame_equal(out, out2)
