"""Robustness: DM-HLN tests (B vs A, models vs seasonal naive) on the subset without Saturdays.

Reads results/predictions.csv; writes results/dm_tests_no_saturdays.csv.
Note: removing Saturdays leaves gaps in the forecast sequence; autocovariances up to h-1 are computed on
the remaining sequence, which is an approximation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import run_dm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pred = pd.read_csv(ROOT / "results" / "predictions.csv", parse_dates=["date", "origin"])
    sub = pred[pred["date"].dt.day_of_week != 5]
    dm = run_dm(sub)
    dm["subset"] = "no_saturdays"
    dm.round(6).to_csv(ROOT / "results" / "dm_tests_no_saturdays.csv", index=False)
    ba = dm[dm.family == "B_vs_A"]
    for loss in ["se", "ae"]:
        b = ba[ba.loss == loss]
        print(loss, "n", len(b), "B better sig", int(((b.p_holm < 0.05) & (b.mean_loss_diff < 0)).sum()),
              "B worse sig", int(((b.p_holm < 0.05) & (b.mean_loss_diff > 0)).sum()))
    print(ba[ba.p_holm < 0.05][["target", "h", "loss", "model_1", "p_holm"]].to_string(index=False))


if __name__ == "__main__":
    main()
