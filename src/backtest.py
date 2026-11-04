"""Stage 5: rolling-origin backtest of all models, feature sets, targets and horizons.

Reads all settings from protocol.yaml. Leakage checks from src/leakage_tests.py are executed at runtime.

Outputs (results/):
  predictions.csv            one row per (target, h, model, feature_set, target date)
  metrics.csv                MASE, MAE, RMSE, WAPE per configuration; subsets: all days / no Saturdays
  dm_tests.csv               DM-HLN tests (squared and absolute loss), Holm-adjusted within family
  shap_groups.csv            mean |SHAP| by feature group for out-of-sample LightGBM forecasts
  permutation_importance.csv group permutation importance (model fitted at first test origin)
  leakage_control.csv        legal set B vs leaky set (same-day KPIs)

Run from repo root: .venv/Scripts/python.exe src/backtest.py
"""

from __future__ import annotations

import math
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import yaml
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import leakage_tests as LT  # noqa: E402
from evaluate import dm_hln, holm, mase_scale, point_metrics  # noqa: E402
from features import feature_columns, feature_group, finite_rows, make_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CFG = yaml.safe_load((ROOT / "protocol.yaml").read_text(encoding="utf-8"))
SEED = CFG["seed"]
M = CFG["frequency"]["seasonal_period"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- statistical models
def _stat_forecast(y_hist: np.ndarray, hmax: int) -> dict[str, np.ndarray]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    warnings.filterwarnings("ignore")
    out = {}
    ets = ExponentialSmoothing(y_hist, trend=None, seasonal="add", seasonal_periods=M,
                               initialization_method="estimated").fit()
    out["ets"] = np.asarray(ets.forecast(hmax))
    sc = CFG["models"]["sarima"]
    sar = SARIMAX(y_hist, order=tuple(sc["order"]), seasonal_order=tuple(sc["seasonal_order"])).fit(disp=False)
    out["sarima"] = np.asarray(sar.forecast(hmax))
    return out


def stat_predictions(y: pd.Series, test_dates: pd.DatetimeIndex, horizons: list[int], target: str) -> list[dict]:
    hmax = max(horizons)
    origins = sorted({T - pd.Timedelta(days=h) for T in test_dates for h in horizons})
    fc = Parallel(n_jobs=-1)(delayed(_stat_forecast)(y.loc[:o].to_numpy(float), hmax) for o in origins)
    fc = dict(zip(origins, fc))
    rows = []
    for h, T in product(horizons, test_dates):
        o = T - pd.Timedelta(days=h)
        sn_date = T - pd.Timedelta(days=M * math.ceil(h / M))
        assert sn_date <= o
        base = {"target": target, "h": h, "feature_set": "-", "date": T, "origin": o, "y": y[T]}
        rows.append({**base, "model": "seasonal_naive", "yhat": y[sn_date]})
        for name, arr in fc[o].items():
            rows.append({**base, "model": name, "yhat": float(arr[h - 1])})
    return rows


# ----------------------------------------------------------------------------- ML models
def ml_models() -> dict:
    mc = CFG["models"]
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))),
        "random_forest": lambda: RandomForestRegressor(**mc["random_forest"], random_state=SEED, n_jobs=-1),
        "lightgbm": lambda: lgb.LGBMRegressor(**mc["lightgbm"], random_state=SEED, deterministic=True,
                                              force_row_wise=True, n_jobs=4, verbose=-1),
    }


def ml_predictions(daily: pd.DataFrame, target: str, h: int, fs: str, models: list[str],
                   test_dates: pd.DatetimeIndex, explain: bool = True) -> tuple[list[dict], list[dict], list[dict]]:
    F = make_features(daily, target, h, fs)
    names = feature_columns(F)
    if fs != "leaky":
        LT.check_lags_respect_horizon(names, h)
        LT.check_no_forbidden_same_period(names)
        LT.check_no_future_dependence(daily, target, h, fs, test_dates[0])
        LT.check_no_future_dependence(daily, target, h, fs, test_dates[-1])
    valid = finite_rows(F)
    assert valid.loc[test_dates].all(), "non-finite features on test dates"
    step = CFG["split"]["ml_refit_every_days"]
    factories = ml_models()
    preds, shap_rows, perm_rows = [], [], []
    rng = np.random.default_rng(SEED)

    for b0 in range(0, len(test_dates), step):
        block = test_dates[b0:b0 + step]
        origin0 = block[0] - pd.Timedelta(days=h)
        train = F[valid & (F.index <= origin0)]
        LT.check_split_chronology(train.index, block[0], h)
        Xtr, ytr = train[names].to_numpy(float), train["y"].to_numpy(float)
        Xte = F.loc[block, names].to_numpy(float)
        for mname in models:
            model = factories[mname]().fit(Xtr, ytr)
            if mname == "ridge":
                LT.check_preprocessing_fit_on_train(model.named_steps["standardscaler"].mean_, Xtr)
            yhat = model.predict(Xte)
            for T, p in zip(block, yhat):
                preds.append({"target": target, "h": h, "model": mname, "feature_set": fs, "date": T,
                              "origin": T - pd.Timedelta(days=h), "y": F.at[T, "y"], "yhat": float(p)})
            if explain and mname == "lightgbm" and fs != "leaky":
                sv = shap.TreeExplainer(model).shap_values(Xte)
                for T, row in zip(block, np.abs(sv)):
                    for n, v in zip(names, row):
                        shap_rows.append({"target": target, "h": h, "feature_set": fs, "date": T,
                                          "feature": n, "abs_shap": float(v)})
            if explain and b0 == 0 and mname in {"lightgbm", "ridge"} and fs != "leaky":
                perm_rows += permutation_importance(model, F.loc[test_dates, names], F.loc[test_dates, "y"],
                                                    names, rng, target, h, fs, mname)
    return preds, shap_rows, perm_rows


def permutation_importance(model, X: pd.DataFrame, y: pd.Series, names: list[str], rng, target, h, fs, mname,
                           n_repeats: int = 10) -> list[dict]:
    Xv, yv = X.to_numpy(float), y.to_numpy(float)
    base = np.mean(np.abs(yv - model.predict(Xv)))
    groups: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        groups.setdefault(feature_group(n), []).append(i)
    rows = []
    for g, idx in groups.items():
        inc = []
        for _ in range(n_repeats):
            Xp = Xv.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, idx] = Xv[perm][:, idx]
            inc.append(np.mean(np.abs(yv - model.predict(Xp))) - base)
        rows.append({"target": target, "h": h, "feature_set": fs, "model": mname, "group": g,
                     "base_mae": base, "mae_increase_mean": float(np.mean(inc)),
                     "mae_increase_std": float(np.std(inc))})
    return rows


# ----------------------------------------------------------------------------- evaluation
def compute_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["target", "h", "model", "feature_set"]
    for subset, mask in [("all_days", pd.Series(True, index=pred.index)),
                         ("no_saturdays", pred["date"].dt.day_of_week != 5)]:
        for k, g in pred[mask].groupby(keys):
            rows.append({**dict(zip(keys, k)), "subset": subset,
                         **point_metrics(g["y"].to_numpy(), g["yhat"].to_numpy(), g["scale"].to_numpy())})
    return pd.DataFrame(rows)


def run_dm(pred: pd.DataFrame, kernel: str = "rectangular") -> pd.DataFrame:
    pred = pred.assign(e=pred["y"] - pred["yhat"])
    rows = []
    for (target, h), g in pred[pred["feature_set"] != "leaky"].groupby(["target", "h"]):
        E = g.pivot_table(index="date", columns=["model", "feature_set"], values="e")
        cols = list(E.columns)
        comps = []
        for c in cols:
            if c[0] != "seasonal_naive":
                comps.append(("vs_seasonal_naive", c, ("seasonal_naive", "-")))
        for c in cols:
            if c[0] in {"random_forest", "lightgbm", "ridge"}:
                comps.append(("vs_ets", c, ("ets", "-")))
        for mname in ["ridge", "random_forest", "lightgbm"]:
            comps.append(("B_vs_A", (mname, "B"), (mname, "A")))
        for fam, c1, c2 in comps:
            for loss in ["se", "ae"]:
                r = dm_hln(E[c1].to_numpy(), E[c2].to_numpy(), h, loss, kernel=kernel)
                rows.append({"target": target, "h": h, "family": fam, "loss": loss,
                             "model_1": f"{c1[0]}[{c1[1]}]", "model_2": f"{c2[0]}[{c2[1]}]", **r})
    dm = pd.DataFrame(rows)
    dm["p_holm"] = dm.groupby(["target", "h", "family", "loss"])["p_value"].transform(holm)
    return dm


def main() -> None:
    t0 = time.time()
    daily = pd.read_parquet(ROOT / "data" / "processed" / "daily_kpis.parquet")
    sp = CFG["split"]
    test_dates = pd.date_range(sp["test_start"], sp["test_end"], freq="D")
    horizons = CFG["horizons"]
    targets = [CFG["targets"]["primary"], *CFG["targets"]["secondary"]]
    ml_names = ["ridge", "random_forest", "lightgbm"]
    log(f"test {test_dates[0].date()}..{test_dates[-1].date()} ({len(test_dates)} days), targets {targets}")

    preds, shap_rows, perm_rows = [], [], []
    for target in targets:
        log(f"{target}: statistical models")
        preds += stat_predictions(daily[target].astype(float), test_dates, horizons, target)
        for h, fs in product(horizons, ["A", "B"]):
            log(f"{target}: ML h={h} set={fs}")
            p, s, pi = ml_predictions(daily, target, h, fs, ml_names, test_dates)
            preds += p; shap_rows += s; perm_rows += pi
    lc = CFG["features"]["leakage_control"]
    for h in horizons:
        log(f"leakage control h={h}")
        p, _, _ = ml_predictions(daily, lc["target"], h, "leaky", lc["models"], test_dates)
        preds += p

    pred = pd.DataFrame(preds)
    scales = {(t, o): mase_scale(daily[t].loc[:o].to_numpy(float), M)
              for t, o in pred[["target", "origin"]].drop_duplicates().itertuples(index=False)}
    pred["scale"] = [scales[(t, o)] for t, o in zip(pred["target"], pred["origin"])]

    # all configurations must share the same target dates per (target, h)
    for (t, h), g in pred.groupby(["target", "h"]):
        LT.check_same_origins({f"{m}[{f}]": gg["date"] for (m, f), gg in g.groupby(["model", "feature_set"])})

    pred.to_csv(RES / "predictions.csv", index=False)
    met = compute_metrics(pred)
    met.round(6).to_csv(RES / "metrics.csv", index=False)
    run_dm(pred).round(6).to_csv(RES / "dm_tests.csv", index=False)
    run_dm(pred, kernel="bartlett").round(6).to_csv(RES / "dm_tests_bartlett.csv", index=False)  # sensitivity

    sh = pd.DataFrame(shap_rows)
    sh["group"] = sh["feature"].map(feature_group)
    sg = (sh.groupby(["target", "h", "feature_set", "date", "group"])["abs_shap"].sum()
          .groupby(["target", "h", "feature_set", "group"]).mean().rename("mean_abs_shap").reset_index())
    sg["share"] = sg["mean_abs_shap"] / sg.groupby(["target", "h", "feature_set"])["mean_abs_shap"].transform("sum")
    sg.round(6).to_csv(RES / "shap_groups.csv", index=False)
    pd.DataFrame(perm_rows).round(4).to_csv(RES / "permutation_importance.csv", index=False)

    leak = met[(met["target"] == lc["target"]) & met["model"].isin(lc["models"])
               & met["feature_set"].isin(["B", "leaky"])]
    leak.round(6).to_csv(RES / "leakage_control.csv", index=False)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
