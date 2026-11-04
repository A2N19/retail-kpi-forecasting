"""Unit tests for metrics, DM-HLN and Holm adjustment."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import dm_hln, holm, mase_scale, point_metrics  # noqa: E402


def test_mase_scale_manual():
    y = np.array([1, 2, 3, 4, 5, 6, 7, 3, 2, 3, 4, 5, 6, 9], float)
    assert mase_scale(y, 7) == pytest.approx(np.mean(np.abs(y[7:] - y[:7])))


def test_point_metrics():
    y, f = np.array([10.0, 20.0]), np.array([12.0, 17.0])
    r = point_metrics(y, f, np.array([2.0, 2.0]))
    assert r["mae"] == 2.5 and r["mase"] == 1.25
    assert r["rmse"] == pytest.approx(np.sqrt((4 + 9) / 2))
    assert r["wape"] == pytest.approx(5 / 30)


def test_dm_identical_errors_not_significant():
    e = np.random.default_rng(0).normal(size=200)
    r = dm_hln(e, e.copy(), 7)
    assert r["p_value"] == 1.0 and r["dm_hln"] == 0.0


def test_dm_detects_clear_difference_and_sign():
    rng = np.random.default_rng(0)
    e_good, e_bad = rng.normal(0, 1, 300), rng.normal(0, 2, 300)
    r = dm_hln(e_good, e_bad, 1)
    assert r["mean_loss_diff"] < 0 and r["p_value"] < 0.001


def test_hln_correction_shrinks_statistic():
    rng = np.random.default_rng(3)
    e1, e2 = rng.normal(0, 1, 60), rng.normal(0, 1.3, 60)
    d = e1 ** 2 - e2 ** 2
    raw = d.mean() / np.sqrt(np.var(d) / len(d))
    assert abs(dm_hln(e1, e2, 1)["dm_hln"]) < abs(raw)


def test_kernels_equal_for_h1_and_bartlett_positive():
    rng = np.random.default_rng(5)
    e1, e2 = rng.normal(0, 1, 120), rng.normal(0, 1.2, 120)
    assert dm_hln(e1, e2, 1)["dm_hln"] == pytest.approx(dm_hln(e1, e2, 1, kernel="bartlett")["dm_hln"])
    # strongly negatively autocorrelated loss differential: rectangular variance collapses, Bartlett does not
    d = np.tile([3.0, -2.0], 60)
    e1 = np.sqrt(np.clip(d, 0, None)); e2 = np.sqrt(np.clip(-d, 0, None))
    assert np.isfinite(dm_hln(e1, e2, 10, kernel="bartlett")["dm_hln"])


def test_negative_variance_fallback_is_flagged_and_counted():
    import evaluate
    d = np.tile([1.0, -1.0], 100)  # alternating loss differential: rectangular variance < 0 at h=2
    e1, e2 = np.sqrt(np.clip(d, 0, None)), np.sqrt(np.clip(-d, 0, None))
    before = evaluate.FALLBACK_COUNTER["fallback_gamma0"]
    r = dm_hln(e1, e2, 2)
    assert r["variance_status"] == "fallback_gamma0"
    assert evaluate.FALLBACK_COUNTER["fallback_gamma0"] == before + 1
    assert dm_hln(e1, e2, 1)["variance_status"] == "ok"


def test_holm():
    p = pd.Series([0.01, 0.04, 0.03, 0.005])
    adj = holm(p)
    assert adj.tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])
