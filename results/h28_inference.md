# Inference at h = 28: problem and proposal (for the thesis)

Status: diagnosis and proposal only. No main result has been recomputed. Numbers come from
`results/dm_variance_summary.csv` and `results/dm_variance_diagnostics.csv` (built by `src/dm_variance_diagnostics.py`).
In the IEEE paper, h = 28 is reported descriptively (MASE only), without significance tests.

## 1. What the protocol computes

For two forecasts with errors $e_{1t}, e_{2t}$, $t = 1,\dots,n$ ($n = 305$ target days), loss differential
$d_t = L(e_{1t}) - L(e_{2t})$, $\bar d = n^{-1}\sum_t d_t$, and sample autocovariances
$\hat\gamma_k = n^{-1}\sum_{t=k+1}^{n}(d_t-\bar d)(d_{t-k}-\bar d)$.

The protocol (Diebold & Mariano 1995; Harvey, Leybourne & Newbold 1997) uses the truncated, equally weighted
(rectangular) long-run variance

$$\hat V_{\text{rect}} = \frac{1}{n}\Big(\hat\gamma_0 + 2\sum_{k=1}^{h-1}\hat\gamma_k\Big),\qquad
\text{DM} = \frac{\bar d}{\sqrt{\hat V_{\text{rect}}}},\qquad
\text{HLN} = \text{DM}\sqrt{\frac{n+1-2h+h(h-1)/n}{n}}\sim t_{n-1}.$$

The truncation at $h-1$ is motivated by theory: optimal $h$-step forecast errors are MA($h-1$). The code
(`src/evaluate.py`) falls back to $\hat\gamma_0/n$ when $\hat V_{\text{rect}}\le 0$; each such case is logged and
flagged in the output column `variance_status`.

## 2. What goes wrong at h = 28

At $h = 28$ the estimator sums 27 sample autocovariances with weight 1 from only 305 observations.

* **The estimate is unstable and can collapse.** Ratio $\hat V_{\text{rect}}/(\hat\gamma_0/n)$ over the 66
  comparisons at $h=28$ (main tests + ablation): minimum −0.073, median 1.444; 12.1% of comparisons are below 0.5,
  and one is negative (LightGBM, A+aov vs A), where the code used $\hat\gamma_0/n$ (originally silently;
  now logged as `variance_status = fallback_gamma0`, counts in `dm_fallback_counts.csv`)
  (`dm_variance_summary.csv`, `dm_variance_diagnostics.csv`). At $h=1$ the ratio is exactly 1; at $h=7$ the minimum is
  0.516 and none is below 0.5.
* **It is mostly lower than a tapered estimate.** At $h=28$ the rectangular estimate is below the Bartlett
  estimate in 57.6% of comparisons (30.3% at $h=7$).
* **Where it collapses, significance is inflated.** The collapsed cases are exactly the "significant" results
  at $h=28$: random forest B vs A (ratio 0.106; Holm $p$ 0.016 rectangular vs 0.896 Bartlett), random forest
  A+cancellation rate vs A (ratio 0.034; $p<0.001$ vs 1.000), Ridge A+active customers vs A (ratio 0.336;
  $p$ 0.042 vs 0.450) (`dm_tests.csv`, `dm_tests_bartlett.csv`, `ablation_groups.csv`).

**Why.** For a demeaned series the sample autocovariances satisfy the exact identity
$\hat\gamma_0 + 2\sum_{k=1}^{n-1}\hat\gamma_k = 0$, i.e. $\sum_{k\ge1}\hat\gamma_k = -\hat\gamma_0/2$.
When the true autocorrelation of $d_t$ is weak (two similar models, as in B vs A or the ablation), the
sample autocovariances at many lags are mostly noise that inherits this negative total; adding 27 of them
with full weight pulls $\hat V_{\text{rect}}$ towards zero. In the collapsed cases the sum
$\sum_{k=1}^{27}\hat\gamma_k/\hat\gamma_0$ is between −0.54 and −0.31 (`dm_variance_diagnostics.csv`,
column `sum_autocov_1_to_h-1_over_gamma0`). The rectangular kernel also does not guarantee a positive estimate,
and the HLN correction does not repair a near-zero denominator. In our data the weekly pattern
(Saturdays: zero revenue, both forecasts near zero, so $d_t \approx 0$ once a week) adds a periodic component
to $d_t$ that further destabilises long sums of autocovariances [interpretation; not tested separately].

## 3. Proposal for the thesis (to be fixed BEFORE recomputation)

**Primary estimator for h = 28: Newey–West (Bartlett) HAC variance with data-driven bandwidth.**

$$\hat V_{\text{NW}} = \frac{1}{n}\Big(\hat\gamma_0 + 2\sum_{k=1}^{M}\Big(1-\frac{k}{M+1}\Big)\hat\gamma_k\Big),$$

* kernel: Bartlett (positive semi-definite by construction; Newey & West 1987);
* bandwidth: automatic selection of Newey & West (1994), with a floor $M \ge h-1$ so that the MA($h-1$)
  structure implied by overlapping forecasts is always covered;
* reference distribution: $t_{n-1}$ without the HLN factor (the HLN factor is derived for the rectangular
  truncation at $h-1$, so it does not transfer to a different kernel/bandwidth);
* multiple testing: Holm within target × horizon × comparison type × loss, unchanged.

**Robustness (report alongside):**

1. fixed-smoothing (fixed-b) critical values for the DM test with Bartlett kernel (Coroneo & Iacone 2020),
   designed for DM tests in small samples;
2. the current Bartlett check with fixed bandwidth $h$ (`dm_tests_bartlett.csv`), for continuity;
3. the protocol (rectangular) result, reported but not interpreted.

**Decision rule (pre-specified):** a difference at h = 28 is called significant only if it is significant
under the primary estimator AND under fixed-b critical values; otherwise h = 28 remains descriptive.

**Not recommended:** non-overlapping origins every 28 days (only about 11 forecasts per comparison, too few
for any test).

**Implementation plan:** add `kernel="nw_auto"` to `evaluate.dm_hln` with unit tests (positivity; equality
with the rectangular estimator at $h=1$; known bandwidth on a simulated AR(1)), log the change in
`protocol.yaml` `meta.changes` before running, then recompute `dm_tests*.csv` and the ablation for h = 28 only.

## 4. References for this note (verified via Crossref, 2026-11-03)

* Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation
  consistent covariance matrix. *Econometrica*, 55(3), 703–708. https://doi.org/10.2307/1913610
  (Crossref returns first page 703 only; last page from the commonly cited range: **[CHECK]**)
* Newey, W. K., & West, K. D. (1994). Automatic lag selection in covariance matrix estimation. *Review of
  Economic Studies*, 61(4), 631–653. https://doi.org/10.2307/2297912
* Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation.
  *Econometrica*, 59(3), 817–858. https://doi.org/10.2307/2938229 (Crossref returns first page 817 only:
  **[CHECK]** last page)
* Coroneo, L., & Iacone, F. (2020). Comparing predictive accuracy in small samples using fixed-smoothing
  asymptotics. *Journal of Applied Econometrics*, 35(4), 391–409. https://doi.org/10.1002/jae.2756
* Diebold & Mariano (1995) and Harvey, Leybourne & Newbold (1997): see `paper/references.bib`.

These four are not yet in `paper/references.bib`; add them through `paper/verify_refs.py` when the thesis
chapter is written.
