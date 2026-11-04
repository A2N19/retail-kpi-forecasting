"""Diagnostics of the DM long-run variance by horizon (no p-values are recomputed).

For every DM comparison of the main backtest (same pairs as backtest.run_dm, squared-error loss) it reports
  ratio_rect     = V_rect / V_0      (protocol: equal weights on autocovariances 1..h-1)
  ratio_bartlett = V_bart / V_0      (weights 1 - k/h)
where V_0 = gamma_0 / n is the variance ignoring autocorrelation.
Outputs: results/dm_variance_diagnostics.csv (per comparison), results/dm_variance_summary.csv (per horizon)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def variances(d: np.ndarray, h: int) -> dict:
    n = len(d)
    dc = d - d.mean()
    g = np.array([np.dot(dc[k:], dc[:n - k]) / n for k in range(h)])
    v0 = g[0] / n
    vr = (g[0] + 2 * g[1:].sum()) / n
    vb = (g[0] + 2 * sum((1 - k / h) * g[k] for k in range(1, h))) / n
    return {"n": n, "v0": v0, "ratio_rect": vr / v0, "ratio_bartlett": vb / v0,
            "rect_nonpositive": bool(vr <= 0), "sum_autocov_1_to_h-1_over_gamma0": g[1:].sum() / g[0]}


def main() -> None:
    pred = pd.read_csv(RES / "predictions.csv", parse_dates=["date"])
    pred = pred[pred.feature_set != "leaky"].assign(e=lambda x: x.y - x.yhat)
    rows = []
    for (target, h), g in pred.groupby(["target", "h"]):
        E = g.pivot_table(index="date", columns=["model", "feature_set"], values="e")
        pairs = [("vs_seasonal_naive", c, ("seasonal_naive", "-")) for c in E.columns if c[0] != "seasonal_naive"]
        pairs += [("vs_ets", c, ("ets", "-")) for c in E.columns if c[0] in {"ridge", "random_forest", "lightgbm"}]
        pairs += [("B_vs_A", (m, "B"), (m, "A")) for m in ["ridge", "random_forest", "lightgbm"]]
        for fam, c1, c2 in pairs:
            d = E[c1].to_numpy() ** 2 - E[c2].to_numpy() ** 2
            rows.append({"target": target, "h": h, "family": fam, "model_1": f"{c1[0]}[{c1[1]}]",
                         "model_2": f"{c2[0]}[{c2[1]}]", **variances(d, int(h))})
    ab = pd.read_csv(RES / "predictions_ablation.csv", parse_dates=["date"]).assign(e=lambda x: x.y - x.yhat)
    for (h, model), g in ab.groupby(["h", "model"]):
        E = g.pivot(index="date", columns="feature_set", values="e")
        for fs in [c for c in E.columns if c != "A"]:
            d = E[fs].to_numpy() ** 2 - E["A"].to_numpy() ** 2
            rows.append({"target": "revenue", "h": h, "family": "ablation_vs_A", "model_1": f"{model}[{fs}]",
                         "model_2": f"{model}[A]", **variances(d, int(h))})
    det = pd.DataFrame(rows)
    det["rect_below_bartlett"] = det["ratio_rect"] < det["ratio_bartlett"]
    det.round(6).to_csv(RES / "dm_variance_diagnostics.csv", index=False)
    summ = det.groupby("h").agg(n_tests=("ratio_rect", "size"),
                                share_rect_below_bartlett=("rect_below_bartlett", "mean"),
                                ratio_rect_min=("ratio_rect", "min"), ratio_rect_median=("ratio_rect", "median"),
                                ratio_bartlett_min=("ratio_bartlett", "min"),
                                ratio_bartlett_median=("ratio_bartlett", "median"),
                                n_rect_nonpositive=("rect_nonpositive", "sum"),
                                share_rect_below_0_5=("ratio_rect", lambda s: float((s < 0.5).mean())))
    summ.round(4).to_csv(RES / "dm_variance_summary.csv")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
