"""Actual vs predicted daily revenue for the best model (lowest MASE, all days, legal feature sets) at h=1 and h=7.

Outputs: results/best_models.csv, figures/fig_actual_vs_pred.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from figures import GRID, INK2, MODEL_LABEL, SERIES

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"


def main() -> None:
    met = pd.read_csv(RES / "metrics.csv")
    m = met[(met.target == "revenue") & (met.subset == "all_days") & (met.feature_set != "leaky")]
    best = m.loc[m.groupby("h")["mase"].idxmin(), ["h", "model", "feature_set", "mase", "mae", "wape"]]
    # report the runner-up too, so near ties are visible
    second = m.sort_values("mase").groupby("h").nth(1)[["h", "model", "feature_set", "mase"]]
    best = best.merge(second, on="h", suffixes=("", "_runner_up"))
    best.round(6).to_csv(RES / "best_models.csv", index=False)
    print(best.round(4).to_string(index=False))

    pred = pd.read_csv(RES / "predictions.csv", parse_dates=["date"])
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True)
    for ax, h in zip(axes, [1, 7]):
        b = best[best.h == h].iloc[0]
        g = pred[(pred.target == "revenue") & (pred.h == h) & (pred.model == b.model)
                 & (pred.feature_set == b.feature_set)].sort_values("date")
        ax.plot(g["date"], g["y"] / 1e3, color=INK2, lw=0.9, label="Actual")
        ax.plot(g["date"], g["yhat"] / 1e3, color=SERIES[0], lw=1.1, label="Forecast")
        ax.set_title(f"h = {h}: {MODEL_LABEL[b.model]} (Model {b.feature_set}), MASE {b.mase:.3f}", loc="left")
        ax.set_ylabel("Revenue, thousand GBP")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_actual_vs_pred.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
