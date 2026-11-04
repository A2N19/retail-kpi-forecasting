"""Stage 2b: impact of PROPOSED cleaning rules (nothing is applied or saved as data).

Rules are evaluated sequentially in the listed order; each row shows the
incremental effect on the data left after previous rules.
Output: results/cleaning_proposal.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim" / "raw_combined.parquet"
OUT = ROOT / "results" / "cleaning_proposal.csv"

COLS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
OVERLAP_START = pd.Timestamp("2010-12-01")
POSTAGE_CODES = {"POST", "DOT", "C2", "C3"}
ACCOUNTING_CODES = {"M", "m", "D", "S", "BANK CHARGES", "ADJUST", "ADJUST2", "AMAZONFEE", "CRUK", "B",
                    "TEST001", "TEST002"}
PARTIAL_LAST_DAY = pd.Timestamp("2011-12-09")


def extreme_pairs(df: pd.DataFrame, min_qty: int = 50000) -> pd.Series:
    """Orders with |Quantity| >= min_qty that have an exact opposite C-line (same customer, code, price)."""
    big = df[df["Quantity"].abs() >= min_qty]
    mask = pd.Series(False, index=df.index)
    for _, r in big[big["Quantity"] > 0].iterrows():
        opp = big[(big["Quantity"] == -r["Quantity"]) & (big["StockCode"] == r["StockCode"])
                  & (big["Customer ID"] == r["Customer ID"]) & (big["Price"] == r["Price"])
                  & big["Invoice"].str.startswith("C")]
        if len(opp):
            mask.loc[[r.name, *opp.index]] = True
    return mask


def main() -> None:
    df = pd.read_parquet(INTERIM)
    df["revenue"] = df["Quantity"] * df["Price"]
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    is_c = df["Invoice"].str.startswith("C")

    rules = [
        ("R1", "drop", "Sheet 2009-2010 rows dated >= 2010-12-01 (exact copy of sheet 2010-2011)",
         lambda d: (d["source_sheet"] == "Year 2009-2010") & (d["InvoiceDate"] >= OVERLAP_START)),
        ("R2", "drop", "Exact duplicate rows within remaining data (all 8 columns), keep first",
         lambda d: d.duplicated(COLS, keep="first")),
        ("R3", "drop", "Invoice prefix A: 'Adjust bad debt' accounting entries",
         lambda d: d["Invoice"].str.startswith("A")),
        ("R4", "drop", "Price == 0 (stock adjustments, damages, free items; revenue 0)",
         lambda d: d["Price"] == 0),
        ("R5", "drop", "Accounting / fee / test StockCodes: " + ", ".join(sorted(ACCOUNTING_CODES)),
         lambda d: d["StockCode"].isin(ACCOUNTING_CODES)),
        ("R6", "drop", "Postage / carriage StockCodes: " + ", ".join(sorted(POSTAGE_CODES)),
         lambda d: d["StockCode"].isin(POSTAGE_CODES)),
        ("R7", "drop", "Extreme orders (|Qty| >= 50000) with exact same-customer cancellation",
         extreme_pairs),
        ("R8", "flag", "Cancellations (Invoice prefix C): keep, exclude from gross revenue, use for Cancellation Rate",
         lambda d: d["Invoice"].str.startswith("C")),
        ("R9", "flag", "Missing Customer ID: keep for Revenue/Orders, exclude only from Active Customers",
         lambda d: d["Customer ID"].isna()),
        ("R10", "kpi", "Partial last day 2011-12-09 (ends 12:50): exclude from the daily KPI table",
         lambda d: d["InvoiceDate"].dt.normalize() == PARTIAL_LAST_DAY),
    ]

    out = []
    cur = df
    out.append({"rule": "R0", "action": "start", "description": "Combined raw (both sheets)",
                "n_rows_affected": 0, "net_revenue_affected": 0.0, "n_rows_remaining": len(cur),
                "net_revenue_remaining": round(cur["revenue"].sum(), 2),
                "gross_revenue_remaining": round(cur.loc[~cur["Invoice"].str.startswith("C"), "revenue"].sum(), 2)})
    for rid, action, desc, fn in rules:
        m = fn(cur)
        rec = {"rule": rid, "action": action, "description": desc,
               "n_rows_affected": int(m.sum()), "net_revenue_affected": round(cur.loc[m, "revenue"].sum(), 2)}
        if action == "drop":
            cur = cur[~m]
        rec["n_rows_remaining"] = len(cur)
        rec["net_revenue_remaining"] = round(cur["revenue"].sum(), 2)
        rec["gross_revenue_remaining"] = round(cur.loc[~cur["Invoice"].str.startswith("C"), "revenue"].sum(), 2)
        out.append(rec)

    # sanity: remaining negative quantities outside C invoices
    neg_not_c = int(((cur["Quantity"] <= 0) & ~cur["Invoice"].str.startswith("C")).sum())
    out.append({"rule": "check", "action": "info", "description": "Remaining rows Quantity<=0 without C (after R1-R7)",
                "n_rows_affected": neg_not_c})
    out.append({"rule": "check", "action": "info",
                "description": "Remaining invoices total / C after R1-R7",
                "n_rows_affected": cur["Invoice"].nunique(),
                "net_revenue_affected": cur.loc[cur["Invoice"].str.startswith("C"), "Invoice"].nunique()})
    _ = is_c
    res = pd.DataFrame(out)
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
