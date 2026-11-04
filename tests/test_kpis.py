"""Correctness of the daily KPI table: synthetic unit test + checks on the real table."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import kpis as K  # noqa: E402


def _tx(rows):
    df = pd.DataFrame(rows, columns=["Invoice", "Quantity", "Price", "Customer ID", "InvoiceDate"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["is_cancellation"] = df["Invoice"].str.startswith("C")
    df["revenue"] = df["Quantity"] * df["Price"]
    return df


def test_build_daily_synthetic():
    tx = _tx([
        ("1", 2, 10.0, 100.0, "2020-01-01 09:00"),  # 20
        ("1", 1, 5.0, 100.0, "2020-01-01 09:00"),   # 5
        ("2", 3, 10.0, np.nan, "2020-01-01 10:00"),  # 30, anonymous
        ("C3", -1, 10.0, 100.0, "2020-01-01 11:00"),
        ("4", 1, 7.0, 200.0, "2020-01-03 09:00"),   # day 2 has no rows
    ])
    d = K.build_daily(tx)
    assert list(d.index.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-01-02", "2020-01-03"]
    day1 = d.loc["2020-01-01"]
    assert day1["revenue"] == 55.0
    assert day1["orders"] == 2
    assert day1["active_customers"] == 1
    assert day1["aov"] == 27.5
    assert day1["cancellation_rate"] == pytest.approx(1 / 3)
    assert day1["quantity_sold"] == 6
    assert day1["revenue_per_customer"] == 25.0
    assert day1["cancelled_value"] == 10.0
    day2 = d.loc["2020-01-02"]
    assert day2["revenue"] == 0 and day2["orders"] == 0 and not day2["is_trading_day"]
    assert np.isnan(day2["aov"]) and np.isnan(day2["cancellation_rate"])


needs_data = pytest.mark.skipif(not K.DAILY.exists(), reason="daily table not built")


@pytest.fixture(scope="module")
def daily():
    return pd.read_parquet(K.DAILY)


@needs_data
def test_calendar_is_contiguous(daily):
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    assert daily.index.equals(full)
    assert daily.index.max() < K.PARTIAL_LAST_DAY


@needs_data
def test_revenue_reconciles_with_transactions(daily):
    tx = pd.read_parquet(K.INTERIM_CLEAN)
    tx = tx[(tx["InvoiceDate"].dt.normalize() != K.PARTIAL_LAST_DAY) & ~tx["is_cancellation"]]
    assert daily["revenue"].sum() == pytest.approx(tx["revenue"].sum(), rel=1e-12)
    assert daily["orders"].sum() == tx.groupby(tx["InvoiceDate"].dt.normalize())["Invoice"].nunique().sum()


@needs_data
def test_kpi_identities(daily):
    t = daily[daily["is_trading_day"]]
    assert np.allclose(t["aov"] * t["orders"], t["revenue"])
    assert t["cancellation_rate"].between(0, 1).all()
    assert (daily.loc[~daily["is_trading_day"], ["revenue", "orders", "active_customers"]] == 0).all().all()
    assert (t["revenue"] > 0).all()
