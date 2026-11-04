"""Thesis EDA figures and the numbers behind them (all from data/processed/daily_kpis.parquet).

Figures (figures/):
  fig_eda_kpis_weekly.png     weekly revenue, orders, active customers, AOV, cancellation rate (full weeks only)
  fig_eda_seasonality.png     mean daily revenue by weekday; mean daily revenue by month and year
  fig_eda_correlations.png    same-day Spearman correlations of KPIs; correlation of revenue with lagged drivers
  fig_eda_christmas.png       daily revenue 1 Dec - 31 Jan for both winters, closure shaded
Tables (results/):
  eda_weekly_kpis.csv, eda_correlations.csv, eda_lagged_correlations.csv, eda_christmas.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures import GRID, INK, INK2, SERIES  # shared style (rcParams set on import)

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
KPI = ["revenue", "orders", "active_customers", "aov", "cancellation_rate", "quantity_sold"]
NAMES = {"revenue": "Revenue", "orders": "Orders", "active_customers": "Active customers", "aov": "AOV",
         "cancellation_rate": "Cancellation rate", "quantity_sold": "Quantity sold"}
DRIVERS = ["orders", "active_customers", "aov", "cancellation_rate"]


def weekly(d: pd.DataFrame) -> pd.DataFrame:
    g = d.resample("W-SUN")
    w = pd.DataFrame({"n_days": g["revenue"].size(), "revenue": g["revenue"].sum(), "orders": g["orders"].sum(),
                      "active_customers_daily_sum": g["active_customers"].sum(),
                      "cancelled_invoices": g["cancelled_invoices"].sum()})
    w = w[w["n_days"] == 7]  # full weeks only
    w["aov"] = w["revenue"] / w["orders"]
    w["cancellation_rate"] = w["cancelled_invoices"] / (w["orders"] + w["cancelled_invoices"])
    return w


def fig_weekly(w: pd.DataFrame) -> None:
    panels = [("revenue", "Revenue, thousand GBP", 1e-3), ("orders", "Orders", 1),
              ("active_customers_daily_sum", "Active customers\n(sum of daily counts)", 1),
              ("aov", "AOV, GBP", 1), ("cancellation_rate", "Cancellation rate", 1)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.0, 7.2), sharex=True)
    for ax, (c, lab, k) in zip(axes, panels):
        ax.plot(w.index, w[c] * k, color=SERIES[0], lw=1.4)
        ax.set_ylabel(lab, fontsize=7.5)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_title("Weekly KPIs, full weeks only (Monday-Sunday)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_eda_kpis_weekly.png")
    plt.close(fig)


def fig_seasonality(d: pd.DataFrame) -> pd.DataFrame:
    wd = d.groupby("day_of_week")["revenue"].mean()
    mo = d["revenue"].groupby([d.index.year, d.index.month]).agg(["mean", "size"])
    mo.index.names = ["year", "month"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1.7]})
    a1.bar(range(7), wd.values / 1e3, color=SERIES[0], width=0.7, edgecolor="white")
    a1.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    a1.set_ylabel("Mean daily revenue, thousand GBP")
    a1.set_title("By day of week", loc="left")
    for i, yr in enumerate([2010, 2011]):
        s = mo.loc[yr]["mean"]
        a2.plot(s.index, s.values / 1e3, marker="o", ms=4, lw=1.6, color=SERIES[i], label=str(yr))
    a2.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    a2.set_title("By month (mean daily revenue; Dec 2011: 8 days)", loc="left")
    a2.legend(frameon=False, fontsize=7)
    for a in (a1, a2):
        a.grid(axis="y", color=GRID, lw=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "fig_eda_seasonality.png")
    plt.close(fig)
    return mo.reset_index()


def correlations(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = d[d["is_trading_day"]]
    same = t[KPI].corr(method="spearman")
    rows = []
    for k in DRIVERS + ["revenue"]:
        for L in [1, 7]:
            x = d[k].shift(L)
            ok = d["is_trading_day"] & d["is_trading_day"].shift(L, fill_value=False)
            rows.append({"driver": k, "lag": L, "n": int(ok.sum()),
                         "spearman_with_revenue": d.loc[ok, "revenue"].corr(x[ok], method="spearman")})
    return same, pd.DataFrame(rows)


def fig_corr(same: pd.DataFrame, lagged: pd.DataFrame) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [SERIES[1], "#f0efec", SERIES[0]])
    im = a1.imshow(same.values, cmap=cmap, vmin=-1, vmax=1)
    labs = [NAMES[k] for k in same.columns]
    a1.set_xticks(range(len(labs)), labs, rotation=45, ha="right", fontsize=6.5)
    a1.set_yticks(range(len(labs)), labs, fontsize=6.5)
    for i in range(len(labs)):
        for j in range(len(labs)):
            a1.text(j, i, f"{same.values[i, j]:.2f}", ha="center", va="center", fontsize=5.8, color=INK)
    a1.set_title("Same-day Spearman correlation\n(trading days)", loc="left")
    fig.colorbar(im, ax=a1, fraction=0.046, pad=0.03)
    order = DRIVERS + ["revenue"]
    y = np.arange(len(order))
    for j, L in enumerate([1, 7]):
        v = lagged[lagged.lag == L].set_index("driver").loc[order, "spearman_with_revenue"]
        a2.barh(y + (j - 0.5) * 0.36, v, height=0.34, color=SERIES[j], label=f"lag {L}", edgecolor="white")
    a2.set_yticks(y, [NAMES[k] for k in order], fontsize=7)
    a2.invert_yaxis()
    a2.axvline(0, color=INK2, lw=0.8)
    a2.set_xlim(-1, 1)
    a2.grid(axis="x", color=GRID, lw=0.6)
    a2.set_axisbelow(True)
    a2.set_title("Spearman correlation of revenue\nwith lagged KPI", loc="left")
    a2.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_eda_correlations.png")
    plt.close(fig)


def christmas(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for i, y0 in enumerate([2009, 2010]):
        win = d.loc[f"{y0}-12-01":f"{y0 + 1}-01-31", "revenue"]
        x = (win.index - pd.Timestamp(f"{y0}-12-01")).days
        ax.plot(x, win.values / 1e3, color=SERIES[i], lw=1.3, label=f"{y0}/{y0 + 1}")
        nt = win[(~d.loc[win.index, "is_trading_day"]) & (d.loc[win.index, "day_of_week"] != 5)]
        run = d.loc[f"{y0}-12-20":f"{y0 + 1}-01-06"]
        closed = run.index[~run["is_trading_day"]]
        start, end = closed.min(), closed.max()
        before = d.loc[:start - pd.Timedelta(days=1)]
        before = before[before["is_trading_day"]].tail(5)["revenue"]
        after = d.loc[end + pd.Timedelta(days=1):]
        after = after[after["is_trading_day"]].head(5)["revenue"]
        rows.append({"winter": f"{y0}/{y0 + 1}", "closure_start": start.date(), "closure_end": end.date(),
                     "closure_days": (end - start).days + 1, "last_trading_day": before.index.max().date(),
                     "mean_rev_5_trading_days_before": round(before.mean(), 2),
                     "first_trading_day_after": after.index.min().date(),
                     "mean_rev_5_trading_days_after": round(after.mean(), 2),
                     "non_saturday_zero_days_in_window": len(nt)})
        ax.axvspan((start - pd.Timestamp(f"{y0}-12-01")).days - 0.5,
                   (end - pd.Timestamp(f"{y0}-12-01")).days + 0.5, color=SERIES[i], alpha=0.08, lw=0)
    ax.set_xlabel("Days since 1 December")
    ax.set_ylabel("Daily revenue, thousand GBP")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Christmas closure: daily revenue 1 Dec - 31 Jan (shaded: closure)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_eda_christmas.png")
    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    d = pd.read_parquet(ROOT / "data" / "processed" / "daily_kpis.parquet")
    w = weekly(d)
    w.round(4).to_csv(RES / "eda_weekly_kpis.csv")
    fig_weekly(w)
    fig_seasonality(d).round(2).to_csv(RES / "eda_monthly_by_year.csv", index=False)
    same, lagged = correlations(d)
    same.round(4).to_csv(RES / "eda_correlations.csv")
    lagged.round(4).to_csv(RES / "eda_lagged_correlations.csv", index=False)
    fig_corr(same, lagged)
    xm = christmas(d)
    xm.to_csv(RES / "eda_christmas.csv", index=False)
    print(same.round(2).to_string())
    print(lagged.round(3).to_string(index=False))
    print(xm.to_string(index=False))


if __name__ == "__main__":
    main()
