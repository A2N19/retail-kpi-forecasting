"""Stage 3b: daily KPI table from clean transactions.

Input : data/interim/clean_transactions.parquet
Output: data/processed/daily_kpis.parquet   one row per CALENDAR day (zeros on non-trading days)
        results/cleaning_log.csv            appends the KPI-level rule R10
        results/daily_kpis_summary.csv      descriptive statistics per KPI

Calendar choice (7-day calendar vs trading days only, m = 7 vs 6) is NOT made here:
the table keeps every calendar day plus flags, so both series can be derived later.

KPI definitions (all per day):
  revenue              sum(Quantity * Price) over non-cancelled lines (gross merchandise revenue)
  orders               number of distinct non-cancelled invoices
  active_customers     number of distinct Customer IDs on non-cancelled lines (missing IDs excluded)
  aov                  revenue / orders                       (NaN when orders = 0)
  cancellation_rate    cancelled invoices / (orders + cancelled invoices)   (NaN when both are 0)
  quantity_sold        sum(Quantity) over non-cancelled lines
  revenue_per_customer revenue from identified customers / active_customers (NaN when 0)
Auxiliary columns: cancelled_invoices, cancelled_value, revenue_identified.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM_CLEAN = ROOT / "data" / "interim" / "clean_transactions.parquet"
DAILY = ROOT / "data" / "processed" / "daily_kpis.parquet"
LOG = ROOT / "results" / "cleaning_log.csv"
SUMMARY = ROOT / "results" / "daily_kpis_summary.csv"

PARTIAL_LAST_DAY = pd.Timestamp("2011-12-09")  # data stops at 12:50 (median last sale of day: ~17h)
KPI_COLS = ["revenue", "orders", "active_customers", "aov", "cancellation_rate", "quantity_sold",
            "revenue_per_customer"]


def build_daily(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.assign(day=tx["InvoiceDate"].dt.normalize())
    sales = tx[~tx["is_cancellation"]]
    canc = tx[tx["is_cancellation"]]

    g = sales.groupby("day")
    daily = pd.DataFrame({
        "revenue": g["revenue"].sum(),
        "orders": g["Invoice"].nunique(),
        "active_customers": g["Customer ID"].nunique(),  # nunique ignores NaN
        "quantity_sold": g["Quantity"].sum(),
        "revenue_identified": sales[sales["Customer ID"].notna()].groupby("day")["revenue"].sum(),
    })
    gc = canc.groupby("day")
    daily["cancelled_invoices"] = gc["Invoice"].nunique()
    daily["cancelled_value"] = -gc["revenue"].sum()  # positive amount

    cal = pd.date_range(tx["day"].min(), tx["day"].max(), freq="D", name="date")
    daily = daily.reindex(cal).fillna(0)
    for c in ["orders", "active_customers", "quantity_sold", "cancelled_invoices"]:
        daily[c] = daily[c].astype("int64")

    daily["aov"] = daily["revenue"] / daily["orders"].replace(0, np.nan)
    denom = (daily["orders"] + daily["cancelled_invoices"]).replace(0, np.nan)
    daily["cancellation_rate"] = daily["cancelled_invoices"] / denom
    daily["revenue_per_customer"] = daily["revenue_identified"] / daily["active_customers"].replace(0, np.nan)

    daily["day_of_week"] = daily.index.day_of_week  # 0 = Monday
    daily["is_saturday"] = daily["day_of_week"] == 5
    daily["is_trading_day"] = daily["orders"] > 0
    return daily[KPI_COLS + ["revenue_identified", "cancelled_invoices", "cancelled_value",
                             "day_of_week", "is_saturday", "is_trading_day"]]


def main() -> None:
    tx = pd.read_parquet(INTERIM_CLEAN)
    partial = tx["InvoiceDate"].dt.normalize() == PARTIAL_LAST_DAY
    tx_kpi = tx[~partial]
    daily = build_daily(tx_kpi)
    daily.to_parquet(DAILY)

    log = pd.read_csv(LOG)
    log = log[log["step"] != "R10"]
    r10 = {
        "step": "R10", "action": "exclude_from_kpi",
        "description": f"Exclude partial last day {PARTIAL_LAST_DAY.date()} from the daily KPI table",
        "rationale": "Transactions stop at 12:50 while the median last sale of a day is ~17:00; "
                     "a truncated day would be a biased target observation.",
        "n_rows_before": len(tx), "n_rows_removed": int(partial.sum()), "n_rows_after": len(tx_kpi),
        "net_revenue_removed": round(float(tx.loc[partial, "revenue"].sum()), 2),
        "date_run": date.today().isoformat(),
    }
    log = pd.concat([log, pd.DataFrame([r10])], ignore_index=True)
    log.to_csv(LOG, index=False)

    summ = daily[KPI_COLS].describe().T
    summ["n_zero"] = (daily[KPI_COLS] == 0).sum()
    summ["n_nan"] = daily[KPI_COLS].isna().sum()
    summ.loc["calendar", ["count"]] = len(daily)
    summ.loc["trading_days", ["count"]] = int(daily["is_trading_day"].sum())
    summ.loc["saturday_trading_days", ["count"]] = int((daily["is_trading_day"] & daily["is_saturday"]).sum())
    summ.loc["non_saturday_non_trading_days", ["count"]] = int((~daily["is_trading_day"] & ~daily["is_saturday"]).sum())
    summ = summ.round(4)
    summ.index.name = "kpi"
    summ.to_csv(SUMMARY)
    print(f"daily table: {daily.index.min().date()} .. {daily.index.max().date()}, {len(daily)} days")
    print(summ.to_string())


if __name__ == "__main__":
    main()
