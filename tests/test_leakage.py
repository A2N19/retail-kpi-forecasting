"""Leakage tests: each legal feature set must pass; the deliberately leaky set must be caught."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import leakage_tests as L  # noqa: E402
from features import feature_columns, lag_set, make_features  # noqa: E402


@pytest.fixture(scope="module")
def daily():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=200, freq="D", name="date")
    d = pd.DataFrame({
        "revenue": rng.uniform(100, 200, len(idx)),
        "orders": rng.integers(1, 50, len(idx)),
        "active_customers": rng.integers(1, 40, len(idx)),
        "quantity_sold": rng.integers(10, 500, len(idx)),
    }, index=idx)
    d["aov"] = d["revenue"] / d["orders"]
    d["cancellation_rate"] = rng.uniform(0, 0.3, len(idx))
    d["day_of_week"] = idx.day_of_week
    return d


@pytest.mark.parametrize("h", [1, 7, 28])
@pytest.mark.parametrize("fs", ["A", "B"])
@pytest.mark.parametrize("target", ["revenue", "orders", "active_customers"])
def test_legal_sets_pass(daily, target, h, fs):
    F = make_features(daily, target, h, fs)
    names = feature_columns(F)
    L.check_lags_respect_horizon(names, h)
    L.check_no_forbidden_same_period(names)
    L.check_no_future_dependence(daily, target, h, fs, daily.index[150])


@pytest.mark.parametrize("h", [1, 7])
def test_leaky_set_is_caught(daily, h):
    names = feature_columns(make_features(daily, "revenue", h, "leaky"))
    with pytest.raises(AssertionError):
        L.check_no_forbidden_same_period(names)
    with pytest.raises(AssertionError):
        L.check_lags_respect_horizon(names, h)
    with pytest.raises(AssertionError):
        L.check_no_future_dependence(daily, "revenue", h, "leaky", daily.index[150])


def test_lag_set_minimum_is_h():
    for h in [1, 7, 28]:
        assert min(lag_set(h)) == h


def test_rolling_shift(daily):
    F = make_features(daily, "revenue", 7, "A")
    L.check_rolling_is_shifted(daily["revenue"], F["own_rmean7"], 7, 7)
    with pytest.raises(AssertionError):
        L.check_rolling_is_shifted(daily["revenue"], daily["revenue"].rolling(7).mean(), 7, 7)


def test_split_chronology():
    T = pd.Timestamp("2020-03-01")
    ok = pd.date_range("2020-01-01", "2020-02-23")
    L.check_split_chronology(ok, T, 7)
    with pytest.raises(AssertionError):
        L.check_split_chronology(pd.date_range("2020-01-01", "2020-02-25"), T, 7)


def test_same_origins():
    a = pd.date_range("2020-01-01", periods=5)
    L.check_same_origins({"m1": a, "m2": a})
    with pytest.raises(AssertionError):
        L.check_same_origins({"m1": a, "m2": a[1:]})
