"""Metrics and Diebold-Mariano test with the Harvey-Leybourne-Newbold correction."""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

LOG = logging.getLogger("evaluate.dm_hln")
FALLBACK_COUNTER: Counter = Counter()  # counts of variance_status over all dm_hln calls in this process


def mase_scale(y_hist: np.ndarray, m: int = 7) -> float:
    """In-sample MAE of the seasonal naive forecast with period m."""
    return float(np.mean(np.abs(y_hist[m:] - y_hist[:-m])))


def point_metrics(y: np.ndarray, yhat: np.ndarray, scale: np.ndarray) -> dict:
    e = y - yhat
    return {
        "n": len(y),
        "mase": float(np.mean(np.abs(e) / scale)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "wape": float(np.sum(np.abs(e)) / np.sum(np.abs(y))),
    }


def dm_hln(e1: np.ndarray, e2: np.ndarray, h: int, loss: str = "se", kernel: str = "rectangular") -> dict:
    """DM test of equal accuracy (two-sided) with HLN small-sample correction.

    d_t = L(e1_t) - L(e2_t); negative mean_d => model 1 more accurate.
    Variance uses autocovariances up to lag h-1 (Diebold & Mariano 1995; Harvey et al. 1997).
    kernel="rectangular" (protocol, equal weights) or "bartlett" (sensitivity: weights 1 - k/h, Newey-West
    type, always non-negative; more stable than the rectangular estimate when h is large relative to n).

    The result has "variance_status": "ok"; "fallback_gamma0" if the HAC variance was <= 0 and the lag-0
    variance gamma_0/n was used instead (unchanged protocol behaviour, now logged and counted); or
    "zero_variance" if the losses are identical (statistic 0, p = 1).
    """
    f = (lambda e: e ** 2) if loss == "se" else np.abs
    d = f(np.asarray(e1, float)) - f(np.asarray(e2, float))
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    gamma = [np.dot(dc[k:], dc[:n - k]) / n for k in range(h)]
    if kernel == "rectangular":
        w = np.ones(h)
    elif kernel == "bartlett":
        w = 1 - np.arange(h) / h
    else:
        raise ValueError(kernel)
    var = (gamma[0] + 2 * sum(w[k] * gamma[k] for k in range(1, h))) / n
    status = "ok"
    if var <= 0:  # protocol behaviour: fall back to lag-0 variance -- explicit and counted
        status = "fallback_gamma0"
        LOG.warning("DM variance <= 0 (h=%d, loss=%s, kernel=%s, n=%d, var=%.6g); using gamma_0/n",
                    h, loss, kernel, n, var)
        var = gamma[0] / n
    if var == 0:  # identical losses: no evidence of a difference
        status = "zero_variance"
        FALLBACK_COUNTER[status] += 1
        LOG.warning("DM zero variance (identical losses), h=%d; statistic 0, p=1", h)
        return {"n": n, "mean_loss_diff": float(dbar), "dm_hln": 0.0, "p_value": 1.0, "variance_status": status}
    FALLBACK_COUNTER[status] += 1
    dm = dbar / np.sqrt(var)
    k = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * k
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"n": n, "mean_loss_diff": float(dbar), "dm_hln": float(stat), "p_value": float(p),
            "variance_status": status}


def holm(pvals: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values."""
    p = pvals.to_numpy(float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return pd.Series(adj, index=pvals.index)
