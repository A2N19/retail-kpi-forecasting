"""Stage 2: data profile of UCI Online Retail II (read-only w.r.t. data/raw).

Outputs
- data/interim/raw_combined.parquet   both sheets stacked, no rows changed or dropped
- results/data_profile.csv            metric, value, note
- results/calendar_gaps.csv           runs of calendar days without any transaction
- results/stockcode_nonstandard.csv   StockCodes not matching the product pattern
- results/extreme_quantities.csv      largest |Quantity| rows
- figures/daily_revenue_raw.png       daily raw revenue (before cleaning)

Run from repo root: .venv/Scripts/python.exe src/data_profile.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "online_retail_II.xlsx"
INTERIM = ROOT / "data" / "interim" / "raw_combined.parquet"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Product codes: 5 digits optionally followed by letters (e.g. 85123A). Everything else is reviewed manually.
PRODUCT_CODE_RE = r"^\d{5}[A-Za-z]*$"


def load_combined(rebuild: bool = False) -> pd.DataFrame:
    """Combined raw data; rebuild=True always re-reads the Excel file (data/raw is only read, never written)."""
    if INTERIM.exists() and not rebuild:
        return pd.read_parquet(INTERIM)
    sheets = pd.read_excel(
        RAW,
        sheet_name=None,
        dtype={"Invoice": str, "StockCode": str, "Description": str, "Country": str},
        engine="openpyxl",
    )
    parts = []
    for name, df in sheets.items():
        df = df.copy()
        df["source_sheet"] = name
        df["source_row"] = range(2, len(df) + 2)  # Excel row number, for traceability
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(INTERIM, index=False)
    return df


def main(rebuild: bool = False) -> None:
    df = load_combined(rebuild)
    rows: list[tuple[str, object, str]] = []

    def add(metric: str, value: object, note: str = "") -> None:
        rows.append((metric, value, note))

    n = len(df)
    data_cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]

    # --- size and period
    add("n_rows_total", n)
    for sheet, cnt in df["source_sheet"].value_counts(sort=False).items():
        add(f"n_rows_sheet[{sheet}]", cnt)
    add("date_min", df["InvoiceDate"].min())
    add("date_max", df["InvoiceDate"].max())
    for sheet, g in df.groupby("source_sheet"):
        add(f"date_range_sheet[{sheet}]", f"{g['InvoiceDate'].min()} .. {g['InvoiceDate'].max()}")
    add("n_invoices_unique", df["Invoice"].nunique())
    add("n_stockcodes_unique", df["StockCode"].nunique())
    add("n_countries", df["Country"].nunique())
    add("share_rows_uk", round((df["Country"] == "United Kingdom").mean(), 4))

    # --- calendar
    day = df["InvoiceDate"].dt.normalize()
    cal = pd.date_range(day.min(), day.max(), freq="D")
    trading = pd.DatetimeIndex(sorted(day.unique()))
    missing = cal.difference(trading)
    add("n_calendar_days", len(cal), "first..last transaction date inclusive")
    add("n_trading_days", len(trading), "days with >= 1 row (any type)")
    add("n_missing_calendar_days", len(missing))
    wd_trading = pd.Series(trading.day_name()).value_counts()
    wd_missing = pd.Series(missing.day_name()).value_counts()
    for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        add(f"trading_days[{d}]", int(wd_trading.get(d, 0)))
        add(f"missing_days[{d}]", int(wd_missing.get(d, 0)))
    sat = df["InvoiceDate"].dt.day_of_week == 5
    add("n_rows_saturday", int(sat.sum()))
    if sat.any():
        add("saturday_dates", ", ".join(sorted(day[sat].dt.strftime("%Y-%m-%d").unique())))

    # runs of consecutive missing days
    gaps = []
    if len(missing):
        s = pd.Series(missing)
        run_id = (s.diff() != pd.Timedelta(days=1)).cumsum()
        for _, g in s.groupby(run_id):
            gaps.append({
                "gap_start": g.iloc[0].date(),
                "gap_end": g.iloc[-1].date(),
                "n_days": len(g),
                "weekdays": "-".join(g.dt.day_name().str[:3]),
            })
    gaps_df = pd.DataFrame(gaps)
    gaps_df.to_csv(RESULTS / "calendar_gaps.csv", index=False)
    long_gaps = gaps_df[gaps_df["n_days"] > 1] if len(gaps_df) else gaps_df
    add("n_gap_runs", len(gaps_df))
    add("gap_runs_longer_than_1_day",
        "; ".join(f"{r.gap_start}..{r.gap_end} ({r.n_days}d)" for r in long_gaps.itertuples()),
        "see results/calendar_gaps.csv")
    # gaps that are not plain Saturdays
    non_sat_missing = [d for d in missing if d.day_of_week != 5]
    add("n_missing_non_saturday_days", len(non_sat_missing))
    add("missing_non_saturday_dates", ", ".join(d.strftime("%Y-%m-%d") for d in non_sat_missing))

    # --- customers
    na_cust = df["Customer ID"].isna()
    rev = df["Quantity"] * df["Price"]
    add("n_rows_missing_customer_id", int(na_cust.sum()))
    add("share_rows_missing_customer_id", round(na_cust.mean(), 4))
    add("share_net_revenue_missing_customer_id", round(rev[na_cust].sum() / rev.sum(), 4), "net = sum(Quantity*Price), all rows")

    # --- invoices / cancellations
    inv_prefix = df["Invoice"].str.extract(r"^([A-Za-z]*)")[0].replace("", "<digit>")
    for p, cnt in inv_prefix.value_counts().items():
        add(f"n_rows_invoice_prefix[{p}]", int(cnt))
    is_c = df["Invoice"].str.startswith("C")
    add("n_rows_cancellation_C", int(is_c.sum()))
    add("share_rows_cancellation_C", round(is_c.mean(), 4))
    add("n_invoices_cancellation_C", df.loc[is_c, "Invoice"].nunique())
    add("share_invoices_cancellation_C", round(df.loc[is_c, "Invoice"].nunique() / df["Invoice"].nunique(), 4))
    add("cancellation_C_revenue_sum", round(rev[is_c].sum(), 2))

    # --- quantity / price
    q0 = df["Quantity"] <= 0
    add("n_rows_quantity_le_0", int(q0.sum()))
    add("n_rows_quantity_lt_0", int((df["Quantity"] < 0).sum()))
    add("n_rows_quantity_eq_0", int((df["Quantity"] == 0).sum()))
    add("n_rows_quantity_le_0_not_C", int((q0 & ~is_c).sum()), "negative qty without C invoice")
    add("n_rows_quantity_gt_0_and_C", int((~q0 & is_c).sum()))
    add("n_rows_price_le_0", int((df["Price"] <= 0).sum()))
    add("n_rows_price_lt_0", int((df["Price"] < 0).sum()))
    add("n_rows_price_eq_0", int((df["Price"] == 0).sum()))
    add("n_rows_price_eq_0_and_missing_customer", int(((df["Price"] == 0) & na_cust).sum()))
    add("n_rows_missing_description", int(df["Description"].isna().sum()))
    add("n_rows_quantity_le_0_not_C_and_price_eq_0", int((q0 & ~is_c & (df["Price"] == 0)).sum()))

    # --- stock codes
    sc = df["StockCode"].astype(str).str.strip()
    nonstd = ~sc.str.match(PRODUCT_CODE_RE)
    add("n_rows_stockcode_nonstandard", int(nonstd.sum()), f"not matching {PRODUCT_CODE_RE}")
    add("n_stockcodes_nonstandard", sc[nonstd].nunique())
    ns = (
        df.assign(StockCode=sc, revenue=rev, is_C=is_c)[nonstd]
        .groupby("StockCode")
        .agg(
            description=("Description", lambda x: x.dropna().mode().iat[0] if x.notna().any() else ""),
            n_rows=("StockCode", "size"),
            n_rows_C=("is_C", "sum"),
            net_revenue=("revenue", "sum"),
            min_price=("Price", "min"),
            max_price=("Price", "max"),
        )
        .sort_values("n_rows", ascending=False)
    )
    ns["net_revenue"] = ns["net_revenue"].round(2)
    ns.to_csv(RESULTS / "stockcode_nonstandard.csv")
    add("net_revenue_stockcode_nonstandard", round(rev[nonstd].sum(), 2))

    # --- duplicates
    dup_all = df.duplicated(subset=data_cols, keep="first")
    add("n_full_duplicate_rows", int(dup_all.sum()), "all 8 data columns identical; first occurrence kept in count")
    dup_within = df.duplicated(subset=data_cols + ["source_sheet"], keep="first")
    add("n_full_duplicate_rows_within_sheet", int(dup_within.sum()))
    add("n_full_duplicate_rows_cross_sheet_only", int((dup_all & ~dup_within).sum()))
    add("net_revenue_in_duplicate_rows", round(rev[dup_all].sum(), 2))

    # --- extreme quantities
    ext = df.assign(revenue=rev).reindex(df["Quantity"].abs().sort_values(ascending=False).index).head(20)
    ext[["source_sheet", "source_row"] + data_cols + ["revenue"]].to_csv(RESULTS / "extreme_quantities.csv", index=False)

    # --- raw revenue totals
    add("net_revenue_all_rows", round(rev.sum(), 2), "before cleaning")

    # --- daily revenue figure (raw, before cleaning)
    daily = rev.groupby(day).sum().reindex(cal)  # NaN on days without transactions
    add("daily_net_revenue_min", round(daily.min(), 2), "raw, trading days only")
    add("daily_net_revenue_max", round(daily.max(), 2), "raw, trading days only")
    add("n_trading_days_net_revenue_le_0", int((daily <= 0).sum()), "raw")

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(daily.index, daily.values / 1000, lw=0.7, color="#1f4e79")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("Net revenue, thousand GBP")
    ax.set_title("Daily net revenue, all rows before cleaning (gaps = days without transactions)")
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(FIGURES / "daily_revenue_raw.png", dpi=200)

    out = pd.DataFrame(rows, columns=["metric", "value", "note"])
    out.to_csv(RESULTS / "data_profile.csv", index=False)
    pd.set_option("display.max_colwidth", 200, "display.width", 250, "display.max_rows", 200)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
