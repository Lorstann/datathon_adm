"""Sizinti ve metrik birim testleri."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.audit.leakage import (
    LeakageError,
    assert_grouped_sorted,
    assert_history_features_constant,
    assert_masked_entities_absent,
    assert_no_future_target,
)
from src.validation.metrics import rmsle, rmsle_from_log, segment_report
from src.validation.folds import mask_history


def test_rmsle_identity():
    a = np.array([0.0, 1.0, 10.0, 100.0])
    p = np.array([0.0, 1.0, 10.0, 100.0])
    assert rmsle(a, p) == pytest.approx(0.0)
    assert rmsle_from_log(np.log1p(a), np.log1p(p)) == pytest.approx(0.0)


def test_rmsle_clips_negative_pred():
    a = np.array([1.0, 2.0])
    p = np.array([-5.0, 2.0])
    assert rmsle(a, p) == pytest.approx(rmsle(a, np.array([0.0, 2.0])))


def test_rmsle_rejects_negative_actual():
    with pytest.raises(ValueError):
        rmsle(np.array([-1.0]), np.array([0.0]))


def test_smearing_would_hurt():
    """Bayes-optimal log1p ortalamasina smearing eklemek RMSLE'yi artirir."""
    rng = np.random.default_rng(0)
    z = rng.normal(5.0, 0.5, size=5000)
    y = np.expm1(z)
    pred_opt = np.expm1(np.full_like(z, z.mean()))
    smeared = pred_opt * float(np.mean(np.exp(z - z.mean())))
    assert rmsle(y, pred_opt) < rmsle(y, smeared)


def test_grouped_sorted_detects_shuffle():
    df = pd.DataFrame(
        {C.ENTITY: ["a", "a", "b"], C.DATE: pd.to_datetime(["2025-01-02", "2025-01-01", "2025-01-01"])}
    )
    with pytest.raises(LeakageError):
        assert_grouped_sorted(df)


def test_no_future_target():
    df = pd.DataFrame(
        {
            C.ENTITY: ["a", "a"],
            C.DATE: pd.to_datetime(["2025-01-01", "2025-02-01"]),
            C.TARGET: [1.0, 2.0],
        }
    )
    with pytest.raises(LeakageError):
        assert_no_future_target(df, pd.Timestamp("2025-01-15"))
    ok = df.loc[df[C.DATE] <= "2025-01-15"]
    assert_no_future_target(ok, pd.Timestamp("2025-01-15"))


def test_mask_removes_history():
    df = pd.DataFrame(
        {
            C.ENTITY: ["a", "a", "b", "b"],
            C.DATE: pd.to_datetime(["2025-01-01", "2025-02-01", "2025-01-01", "2025-02-01"]),
            C.TARGET: [1.0, 2.0, 3.0, 4.0],
        }
    )
    origin = pd.Timestamp("2025-01-15")
    masked = pd.Index(["a"], name=C.ENTITY)
    out = mask_history(df, masked, origin)
    assert_masked_entities_absent(out, masked, origin)
    assert "a" not in set(out.loc[out[C.DATE] <= origin, C.ENTITY])


def test_history_features_constant():
    df = pd.DataFrame(
        {
            C.ENTITY: ["a", "a", "b", "b"],
            "hist_mean": [1.0, 1.0, 2.0, 2.1],
        }
    )
    with pytest.raises(LeakageError):
        assert_history_features_constant(df, ["hist_mean"])


def test_segment_report_blend():
    y = np.array([1.0, 1.0, 10.0, 10.0])
    p = np.array([1.0, 2.0, 10.0, 20.0])
    seg = pd.Series(["0_cold", "0_cold", "4_270g+", "4_270g+"])
    cold = pd.Series([True, True, False, False])
    m = segment_report(y, p, seg, is_cold=cold, blend_cold_rate=0.5)
    assert "rmsle_blend" in m
    assert m["rmsle_cold"] > 0
    assert m["rmsle_warm"] > 0
