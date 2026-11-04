"""Feature construction for the direct multi-horizon strategy.

For horizon h and target date T, every data-driven feature uses only values at dates <= T - h.
Calendar features (day of week, month, public holidays) depend on the date T only and are known in advance.

Feature sets
  A      own-target lags + own-target rolling statistics + calendar
  B      A + lags / rolling means of the other KPIs (orders, active customers, AOV, cancellation rate, revenue)
  leaky  B + SAME-DAY values of the other KPIs (forbidden; used only for the leakage control experiment)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

KPIS = ["revenue", "orders", "active_customers", "aov", "cancellation_rate"]
LEAKY_SAME_DAY = ["orders", "active_customers", "aov", "cancellation_rate", "quantity_sold", "revenue"]

# England & Wales bank holidays (gov.uk), Dec 2009 - Dec 2011, incl. substitute days and 2011-04-29.
UK_BANK_HOLIDAYS = pd.to_datetime([
    "2009-12-25", "2009-12-28",
    "2010-01-01", "2010-04-02", "2010-04-05", "2010-05-03", "2010-05-31", "2010-08-30", "2010-12-27", "2010-12-28",
    "2011-01-03", "2011-04-22", "2011-04-25", "2011-04-29", "2011-05-02", "2011-05-30", "2011-08-29",
    "2011-12-26", "2011-12-27",
])


def lag_set(h: int) -> list[int]:
    """Recent lags h..h+6 plus the first four weekly (multiple-of-7) lags that are >= h."""
    k0 = math.ceil(h / 7)
    return sorted(set(range(h, h + 7)) | {7 * k for k in range(max(k0, 1), max(k0, 1) + 4)})


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    cal = pd.DataFrame(index=index)
    for d in range(7):
        cal[f"cal_dow_{d}"] = (index.day_of_week == d).astype(int)
    for m in range(1, 13):
        cal[f"cal_month_{m}"] = (index.month == m).astype(int)
    cal["cal_holiday"] = index.isin(UK_BANK_HOLIDAYS).astype(int)
    md = index.month * 100 + index.day
    cal["cal_xmas_period"] = ((md >= 1224) | (md <= 103)).astype(int)
    return cal


def make_features(daily: pd.DataFrame, target: str, h: int, feature_set: str) -> pd.DataFrame:
    """Return a frame indexed by target date T with column 'y' and feature columns (NaN rows not dropped)."""
    others = [c for c in KPIS if c != target]
    if feature_set in {"B", "leaky"}:
        groups = others
    elif feature_set == "A":
        groups = []
    elif feature_set.startswith("A+"):  # ablation: A plus selected driver groups, e.g. "A+orders"
        groups = feature_set[2:].split("+")
        if not set(groups) <= set(others):
            raise ValueError(feature_set)
    else:
        raise ValueError(feature_set)
    s = daily.copy()
    s[["aov", "cancellation_rate"]] = s[["aov", "cancellation_rate"]].fillna(0.0)  # 0 on non-trading days
    cols: dict[str, pd.Series] = {}

    y = s[target]
    for L in lag_set(h):
        cols[f"own_lag{L}"] = y.shift(L)
    past = y.shift(h)
    for w in (7, 28):
        cols[f"own_rmean{w}"] = past.rolling(w).mean()
        cols[f"own_rstd{w}"] = past.rolling(w).std()

    for k in others:  # fixed KPI order so that column order does not depend on the set definition
        if k in groups:
            x = s[k]
            for L in range(h, h + 7):
                cols[f"{k}_lag{L}"] = x.shift(L)
            cols[f"{k}_rmean7"] = x.shift(h).rolling(7).mean()

    if feature_set == "leaky":
        for k in [c for c in LEAKY_SAME_DAY if c != target]:
            cols[f"{k}_lag0"] = s[k]  # same period: FORBIDDEN in real forecasting

    X = pd.DataFrame(cols, index=s.index)
    X = X.join(calendar_features(s.index))
    X.insert(0, "y", y)
    return X


def feature_group(name: str) -> str:
    if name.startswith("cal_"):
        return "calendar"
    if name.startswith("own_lag"):
        return "own_lags"
    if name.startswith("own_r"):
        return "own_rolling"
    if name.endswith("_lag0"):
        return "same_day_forbidden"
    for k in KPIS:
        if name.startswith(k + "_"):
            return k
    return "other"


def source_lag(name: str) -> int | None:
    """Minimal lag (in days) of the data a feature depends on; None for calendar features."""
    if name.startswith("cal_"):
        return None
    if "_lag" in name:
        return int(name.rsplit("_lag", 1)[1])
    return None  # rolling features: checked by perturbation test, see leakage_tests


def feature_columns(F: pd.DataFrame) -> list[str]:
    return [c for c in F.columns if c != "y"]


def finite_rows(F: pd.DataFrame) -> pd.Series:
    ok = np.isfinite(F[feature_columns(F) + ["y"]].to_numpy(dtype=float)).all(axis=1)
    return pd.Series(ok, index=F.index)
