# Do Lagged Operational KPIs Improve Daily Revenue Forecasts?

Code, frozen experimental protocol, result files, figures and LaTeX sources for the paper

> A. Nurseit, Zh. Beldeubayeva. *Do Lagged Operational KPIs Improve Daily Revenue Forecasts? A Leakage-Safe
> Ablation on Online Retail Data.* Submitted to IEEE DG 2027 (Astana IT University).

The paper PDF is [`paper/main.pdf`](paper/main.pdf).

## Research question

Business KPIs such as orders, active customers and average order value (AOV) are mechanically linked to
revenue (revenue = orders x AOV), which makes them attractive predictors but also an easy source of data
leakage. The study asks whether **lagged** operational KPIs improve daily revenue forecasts when evaluated
under a leakage-safe rolling-origin protocol:

- **RQ1.** Do ML models (Ridge, random forest, LightGBM) beat seasonal naive, ETS and SARIMA for daily revenue
  (and, as secondary targets, orders and active customers)?
- **RQ2 (core of the paper).** Does Model B (revenue history + calendar + lagged drivers) beat Model A
  (revenue history + calendar)?
- **RQ3.** How do accuracy and the ML advantage change with the horizon, and which feature groups drive the
  forecasts (SHAP by feature group)?

## Data

UCI **Online Retail II** (transactions of a UK online retailer, 2009-12-01 to 2011-12-09, 1,067,371 rows,
license CC BY 4.0):

> Chen, D. (2012). Online Retail II [Data set]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5CG6D

The raw data is **not** included in this repository. To obtain it:

1. Download the archive from the dataset page https://archive.ics.uci.edu/dataset/502/online+retail+ii
   (direct link: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip).
2. Extract `online_retail_II.xlsx` (one Excel file with the sheets `Year 2009-2010` and `Year 2010-2011`) into
   `data/raw/`, so that the path is `data/raw/online_retail_II.xlsx`.
3. Check the file against the SHA-256 frozen in `protocol.yaml`:

   ```bash
   sha256sum data/raw/online_retail_II.xlsx
   # bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980
   ```

   PowerShell: `Get-FileHash data\raw\online_retail_II.xlsx -Algorithm SHA256`.

`data/raw/` is only read. All derived data (`data/interim/`, `data/processed/`, parquet) is rebuilt by the
pipeline. Preparation is fully scripted: `src/data_profile.py` combines the two sheets and profiles them,
`src/cleaning.py` applies the cleaning rules R1-R9 (documented row counts in `results/cleaning_log.csv`) and
`src/kpis.py` builds the daily KPI table on a full 7-day calendar (R10 drops the partial last day 2011-12-09).

KPIs per calendar day: revenue (sum of quantity x price, non-cancelled lines), orders, active customers, AOV,
cancellation rate, quantity sold, revenue per customer (definitions in `protocol.yaml`, code in `src/kpis.py`).

## Design (frozen in `protocol.yaml`)

| Item | Setting |
|---|---|
| Frequency | daily, full 7-day calendar with zeros on non-trading days, seasonal period m = 7 |
| Targets | revenue (primary); orders, active customers (secondary) |
| Horizons | 1, 7, 28 days, direct strategy (one model per horizon); significance only for h = 1 and 7, h = 28 descriptive |
| Validation | expanding-window rolling origin, one forecast per target day; test period 2011-02-07 to 2011-12-08 (305 target days per horizon); identical target dates for all models |
| Refit | statistical models at every origin; ML models weekly on all data up to the origin |
| Metrics | MASE (primary; scale = in-sample MAE of seasonal naive lag 7 per origin), MAE, RMSE, WAPE |
| Significance | Diebold-Mariano with Harvey-Leybourne-Newbold correction, squared-error loss (absolute error as robustness), Holm correction within families, alpha = 0.05 |
| Seed | 42 |

**Model A**: own lags (h..h+6 plus the first four multiples of 7 that are >= h), rolling mean and standard
deviation over 7 and 28 days of the series shifted by h, calendar features (day of week, month, UK bank
holidays, Christmas period).

**Model B**: Model A + lags h..h+6 and the 7-day rolling mean of the other KPIs (revenue, orders, active
customers, AOV, cancellation rate, minus the target).

Only values available at the forecast origin (dates <= T - h) are used; scaling and Ridge regularisation are
fitted on the training data of each origin. A **leakage control** experiment adds the target day's own
orders, active customers, AOV, cancellation rate and quantity sold to Model B (Ridge and LightGBM) to show
how much accuracy is overstated when such features leak.

**Models**: seasonal naive (baseline), ETS (additive error and seasonality, m = 7), SARIMA(1,0,1)(0,1,1)7,
Ridge, random forest, LightGBM, all with fixed a priori settings (no tuning on the test period).

Post hoc additions (logged in `protocol.yaml` under `meta.changes`): DM tests without Saturdays, a
feature-group ablation (Model A + one driver group) and DM tests with Bartlett weights.

## Repository structure

```
protocol.yaml        frozen experimental protocol (read by src/backtest.py)
run_all.py           whole pipeline: raw Excel -> results, figures, paper tables, PDF, tests
requirements.in      direct dependencies;  requirements.txt  exact environment (pip freeze, Python 3.12.7)
src/                 data_profile, cleaning, kpis, eda, features, backtest, evaluate (metrics, DM-HLN),
                     robustness, ablation, dm_variance_diagnostics, dm_fallback_report, figures,
                     eda_figures, prediction_figures, paper_numbers, leakage_tests
tests/               pytest: leakage, KPI definitions, cleaning, metrics/DM, PDF-vs-LaTeX text check
notebooks/           01_cleaning.ipynb (runs src/ and shows the outputs)
results/             all result files (*.csv, logs); every number in the paper comes from here
figures/             all figures (PNG)
paper/               main.tex, generated tables (tab_*.tex), references.bib, main.pdf,
                     check_text.py (PDF vs LaTeX), verify_refs.py (reference metadata check)
data/raw/            place online_retail_II.xlsx here (not versioned)
data/interim/, data/processed/   rebuilt by the pipeline (not versioned)
```

## Installation

Python 3.12 (the exact environment was built with Python 3.12.7).

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/macOS:  source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` pins the full environment; `requirements.in` lists only the direct dependencies.

## Running the experiments

From the repository root, with the virtual environment activated and the raw file in `data/raw/`:

```bash
python run_all.py               # full pipeline, then builds the paper PDF (if tectonic is found) and runs pytest
python run_all.py --skip-paper  # same without building the PDF
```

`run_all.py` runs, in order: `data_profile`, `cleaning`, `kpis`, `eda`, `eda_figures`, `backtest`,
`robustness`, `ablation`, `dm_variance_diagnostics`, `dm_fallback_report`, `figures`, `prediction_figures`,
`paper_numbers`. Step timings are written to `results/run_all_log.csv`. Each step can also be run on its own,
e.g. `python src/backtest.py`, once the previous steps have produced their inputs.

### Tests

```bash
python -m pytest -q tests
```

Tests that need the derived data or the PDF are skipped if those files are not present.

### Tables and figures

```bash
python src/paper_numbers.py       # results/paper_numbers.csv and paper/tab_mase_revenue.tex, tab_b_vs_a.tex, tab_ablation.tex
python src/figures.py             # figures/fig_rq1_*, fig_rq2_*, fig_rq3_shap_groups.png, fig_leakage_control.png
python src/eda_figures.py         # figures/fig_eda_*.png
python src/prediction_figures.py  # figures/fig_actual_vs_pred.png
```

These scripts read `results/*.csv` only, so they run without the raw data.

### Building the paper

The paper uses the IEEEtran class and is built with [tectonic](https://tectonic-typesetting.github.io/):

```bash
cd paper
tectonic main.tex
cd ..
python paper/check_text.py        # checks that all text of main.tex appears in main.pdf and no citation is unresolved
```

`paper/verify_refs.py` re-checks reference metadata against Crossref, DataCite, OpenAlex and arXiv; it needs
network access and is not part of the pipeline.

## Traceability of numbers

Every number in the paper is taken from `results/*.csv`. `src/paper_numbers.py` collects them into
`results/paper_numbers.csv` (key, value, source file, description) and generates the LaTeX tables; the
`%` comments next to each number in `paper/main.tex` name the key and source file.

## License

The dataset is distributed by UCI under CC BY 4.0. This repository does not include a license file for the
code; please contact the authors about reuse.
