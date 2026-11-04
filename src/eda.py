"""Stage 4: EDA numbers needed for design decisions (seasonal period, calendar, test period).

Output: results/eda_seasonality.csv, results/eda_weekday_profile.csv, results/eda_monthly.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "processed" / "daily_kpis.parquet"
RES = ROOT / "results"


def snaive_mae(y: np.ndarray, m: int) -> float:
    return float(np.mean(np.abs(y[m:] - y[:-m])))


def main() -> None:
    d = pd.read_parquet(DAILY)
    rows = []
    series = {
        "calendar_7day": d["revenue"],
        "no_saturdays_6day": d.loc[~d["is_saturday"], "revenue"],
        "trading_days_only": d.loc[d["is_trading_day"], "revenue"],
    }
    for name, s in series.items():
        y = s.to_numpy()
        a = acf(y, nlags=14, fft=True)
        for lag in [1, 5, 6, 7, 12, 13, 14]:
            rows.append({"series": name, "n": len(y), "stat": f"acf_lag{lag}", "value": round(float(a[lag]), 4)})
        for m in [6, 7]:
            rows.append({"series": name, "n": len(y), "stat": f"snaive_mae_m{m}", "value": round(snaive_mae(y, m), 2)})
        rows.append({"series": name, "n": len(y), "stat": "naive1_mae", "value": round(snaive_mae(y, 1), 2)})
    pd.DataFrame(rows).to_csv(RES / "eda_seasonality.csv", index=False)

    wd = (d.groupby("day_of_week")
          .agg(n_days=("revenue", "size"), n_trading=("is_trading_day", "sum"),
               mean_revenue=("revenue", "mean"), median_revenue=("revenue", "median"),
               mean_orders=("orders", "mean"))
          .round(2))
    wd.to_csv(RES / "eda_weekday_profile.csv")

    mo = d["revenue"].resample("MS").agg(["sum", "mean", "count"]).round(2)
    mo.to_csv(RES / "eda_monthly.csv")

    # country shares on clean, non-cancelled lines within the KPI period (excludes partial 2011-12-09)
    tx = pd.read_parquet(ROOT / "data" / "interim" / "clean_transactions.parquet",
                         columns=["Country", "revenue", "is_cancellation", "InvoiceDate"])
    tx = tx[~tx["is_cancellation"] & (tx["InvoiceDate"].dt.normalize() <= d.index.max())]
    cs = tx.groupby("Country").agg(n_lines=("revenue", "size"), gross_revenue=("revenue", "sum"))
    cs["share_lines"] = cs["n_lines"] / cs["n_lines"].sum()
    cs["share_revenue"] = cs["gross_revenue"] / cs["gross_revenue"].sum()
    cs.sort_values("gross_revenue", ascending=False).round(4).to_csv(RES / "eda_country_share.csv")
    print(pd.DataFrame(rows).pivot(index="stat", columns="series", values="value").to_string())
    print(wd.to_string())
    print(mo.to_string())


if __name__ == "__main__":
    main()
