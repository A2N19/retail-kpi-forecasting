"""Stage 3: apply the cleaning rules to the combined raw data.

Input : data/interim/raw_combined.parquet  (built by src/data_profile.py; data/raw is never modified)
Output: data/interim/clean_transactions.parquet
        results/cleaning_log.csv           one row per rule: what, how many rows, revenue effect, why

Rules are applied in a fixed order (R1..R7). Cancellations (R8) and missing Customer ID (R9)
are kept and flagged; their treatment is defined in src/kpis.py. R10 (partial last day) is
applied at KPI level, not here, so that transactions stay complete.

Run from repo root: .venv/Scripts/python.exe src/cleaning.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM_RAW = ROOT / "data" / "interim" / "raw_combined.parquet"
INTERIM_CLEAN = ROOT / "data" / "interim" / "clean_transactions.parquet"
LOG = ROOT / "results" / "cleaning_log.csv"

DATA_COLS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
SHEET_OLD = "Year 2009-2010"
OVERLAP_START = pd.Timestamp("2010-12-01")
ACCOUNTING_CODES = frozenset({"M", "m", "D", "S", "BANK CHARGES", "ADJUST", "ADJUST2", "AMAZONFEE", "CRUK", "B",
                              "TEST001", "TEST002"})
POSTAGE_CODES = frozenset({"POST", "DOT", "C2", "C3"})
EXTREME_QTY = 50_000


def _extreme_cancelled_pairs(d: pd.DataFrame) -> pd.Series:
    """Rows of orders with Quantity >= EXTREME_QTY plus their exact C-cancellation (same customer, code, price)."""
    big = d[d["Quantity"].abs() >= EXTREME_QTY]
    mask = pd.Series(False, index=d.index)
    for idx, r in big[big["Quantity"] > 0].iterrows():
        opp = big[(big["Quantity"] == -r["Quantity"]) & (big["StockCode"] == r["StockCode"])
                  & (big["Customer ID"] == r["Customer ID"]) & (big["Price"] == r["Price"])
                  & big["Invoice"].str.startswith("C")]
        if len(opp):
            mask.loc[[idx, *opp.index]] = True
    return mask


# (rule_id, description, rationale, mask function)
RULES: list[tuple[str, str, str, Callable[[pd.DataFrame], pd.Series]]] = [
    ("R1", "Drop sheet 2009-2010 rows dated >= 2010-12-01",
     "Both sheets contain 2010-12-01..2010-12-09; rows are identical (22,523 = 22,523, all keys match). "
     "Keeping both double-counts these days. Sheet 2010-2011 copy is kept.",
     lambda d: (d["source_sheet"] == SHEET_OLD) & (d["InvoiceDate"] >= OVERLAP_START)),
    ("R2", "Drop exact duplicate rows (all 8 data columns), keep first",
     "No invoice line number exists to distinguish genuine repeated lines from export duplicates; "
     "conservative choice is to treat identical rows as duplicates. Reported as a limitation.",
     lambda d: d.duplicated(DATA_COLS, keep="first")),
    ("R3", "Drop invoices with prefix A ('Adjust bad debt')",
     "Accounting write-offs, not sales; 6 rows carry -147,614 GBP.",
     lambda d: d["Invoice"].str.startswith("A")),
    ("R4", "Drop rows with Price <= 0",
     "Zero-price lines are stock adjustments, damages and free items (mostly without Customer ID); "
     "they add no revenue but inflate Orders and Quantity Sold.",
     lambda d: d["Price"] <= 0),
    ("R5", "Drop accounting / fee / test StockCodes: " + ", ".join(sorted(ACCOUNTING_CODES)),
     "Manual corrections, discounts, samples, bank charges, marketplace fees, commissions and test items "
     "are not product sales.",
     lambda d: d["StockCode"].isin(ACCOUNTING_CODES)),
    ("R6", "Drop postage / carriage StockCodes: " + ", ".join(sorted(POSTAGE_CODES)),
     "Target KPI is merchandise revenue; shipping charges are pass-through and not driven by the same process.",
     lambda d: d["StockCode"].isin(POSTAGE_CODES)),
    ("R7", f"Drop orders with Quantity >= {EXTREME_QTY:,} together with their exact same-customer cancellation",
     "Two entry errors (80,995 and 74,215 units) cancelled within minutes; they would create artificial "
     "gross-revenue spikes.",
     _extreme_cancelled_pairs),
]


def load_raw_combined() -> pd.DataFrame:
    df = pd.read_parquet(INTERIM_RAW)
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply R1..R7 in order. Returns (clean transactions, log)."""
    log = []
    cur = df.copy()
    rev = cur["Quantity"] * cur["Price"]
    for rid, desc, why, fn in RULES:
        m = fn(cur)
        log.append({
            "step": rid, "action": "drop", "description": desc, "rationale": why,
            "n_rows_before": len(cur), "n_rows_removed": int(m.sum()), "n_rows_after": int((~m).sum()),
            "net_revenue_removed": round(float(rev.loc[m[m].index].sum()), 2),
        })
        cur = cur[~m]

    cur = cur.copy()
    cur["is_cancellation"] = cur["Invoice"].str.startswith("C")
    cur["revenue"] = cur["Quantity"] * cur["Price"]
    n = len(cur)
    log.append({
        "step": "R8", "action": "flag", "description": "Keep cancellations (Invoice prefix C), flag is_cancellation",
        "rationale": "Revenue is defined as gross sales (non-cancelled lines) to avoid negative daily values; "
                     "cancellations feed the Cancellation Rate KPI.",
        "n_rows_before": n, "n_rows_removed": 0, "n_rows_after": n,
        "net_revenue_removed": 0.0, "n_rows_flagged": int(cur["is_cancellation"].sum()),
    })
    log.append({
        "step": "R9", "action": "flag", "description": "Keep rows with missing Customer ID",
        "rationale": "They are real sales (~14% of revenue). Used for Revenue, Orders, Quantity; excluded only "
                     "from Active Customers and Revenue per Customer.",
        "n_rows_before": n, "n_rows_removed": 0, "n_rows_after": n,
        "net_revenue_removed": 0.0, "n_rows_flagged": int(cur["Customer ID"].isna().sum()),
    })
    log_df = pd.DataFrame(log)
    log_df["date_run"] = date.today().isoformat()
    return cur, log_df


def main() -> None:
    df = load_raw_combined()
    out, log_df = clean(df)
    out.to_parquet(INTERIM_CLEAN, index=False)
    log_df.to_csv(LOG, index=False)
    pd.set_option("display.width", 220, "display.max_colwidth", 60)
    print(log_df.drop(columns=["rationale"]).to_string(index=False))


if __name__ == "__main__":
    main()
