"""Leakage checks for the forecasting pipeline.

Each check raises AssertionError with an explicit message on violation. They are called both from
tests/test_leakage.py and at runtime inside src/backtest.py.

Rule (direct strategy, horizon h): to forecast the target at date T, a feature may use only
information from dates <= T - h. Calendar features of date T are known in advance and allowed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import KPIS, feature_columns, make_features, source_lag

FORBIDDEN_SAME_PERIOD = ["orders", "aov", "active_customers", "cancellation_rate", "quantity_sold"]


def check_lags_respect_horizon(feature_names: list[str], horizon: int) -> None:
    """Every explicit lag feature must have lag >= h."""
    bad = [n for n in feature_names if (L := source_lag(n)) is not None and L < horizon]
    assert not bad, f"lags < h={horizon}: {bad[:5]}"


def check_no_forbidden_same_period(feature_names: list[str]) -> None:
    """No same-period KPI (bare KPI name or *_lag0) may appear among the features."""
    bad = [n for n in feature_names if n in FORBIDDEN_SAME_PERIOD or n in KPIS or n.endswith("_lag0")]
    assert not bad, f"forbidden same-period features: {bad}"


def check_no_future_dependence(daily: pd.DataFrame, target: str, h: int, feature_set: str,
                               T: pd.Timestamp, seed: int = 0) -> None:
    """Perturbation test: overwrite every KPI at dates > T - h with noise; features at T must not change."""
    base = make_features(daily, target, h, feature_set)
    pert = daily.copy()
    future = pert.index > T - pd.Timedelta(days=h)
    rng = np.random.default_rng(seed)
    num = pert.select_dtypes("number").columns.difference(["day_of_week"])
    pert[num] = pert[num].astype(float)
    pert.loc[future, num] = rng.uniform(1, 1e6, size=(future.sum(), len(num)))
    new = make_features(pert, target, h, feature_set)
    cols = feature_columns(base)
    a, b = base.loc[T, cols].to_numpy(float), new.loc[T, cols].to_numpy(float)
    changed = [c for c, x, z in zip(cols, a, b) if not (np.isclose(x, z) or (np.isnan(x) and np.isnan(z)))]
    assert not changed, f"features at {T.date()} depend on data after T-h (h={h}): {changed[:5]}"


def check_rolling_is_shifted(series: pd.Series, rolled: pd.Series, window: int, shift: int) -> None:
    """A rolling feature must equal series.shift(shift).rolling(window).mean()."""
    expected = series.shift(shift).rolling(window).mean()
    assert np.allclose(rolled, expected, equal_nan=True), "rolling feature not computed on shifted series"


def check_split_chronology(train_index: pd.DatetimeIndex, target_date: pd.Timestamp, horizon: int) -> None:
    """All training targets must be observed at the forecast origin T - h."""
    origin = target_date - pd.Timedelta(days=horizon)
    assert len(train_index) > 0, "empty training set"
    assert train_index.max() <= origin, f"train target {train_index.max().date()} > origin {origin.date()}"


def check_preprocessing_fit_on_train(scaler_mean: np.ndarray, X_train: np.ndarray) -> None:
    """A fitted StandardScaler must carry the column means of the training rows only."""
    assert np.allclose(scaler_mean, X_train.mean(axis=0)), "scaler not fitted on the training window"


def check_same_origins(targets_by_model: dict[str, pd.Index]) -> None:
    """All models must be evaluated on identical target dates (hence identical origins)."""
    ref = None
    for name, idx in targets_by_model.items():
        idx = pd.Index(sorted(idx))
        if ref is None:
            ref = idx
        assert idx.equals(ref), f"model {name} evaluated on different target dates"
