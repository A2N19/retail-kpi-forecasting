"""Feature-group ablation for daily revenue (POST HOC: added after the main results, see protocol.yaml meta.changes).

Sets: A, A+orders, A+active_customers, A+aov, A+cancellation_rate, B (= A + all four groups).
Models: Ridge, random forest, LightGBM; horizons from protocol.yaml; same rolling origins and settings as backtest.

DM-HLN: each set vs A, per model. Holm family definition is the same as in the main tests
(target x horizon x comparison type x loss); here the comparison type is "ablation_vs_A", so each family
holds 5 sets x 3 models = 15 tests.

Determinism check: sets A and B are recomputed and compared with results/predictions.csv.

Outputs: results/predictions_ablation.csv, results/ablation_groups.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import CFG, M, log, ml_predictions  # noqa: E402
from evaluate import dm_hln, holm, mase_scale, point_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
GROUPS = ["orders", "active_customers", "aov", "cancellation_rate"]
SETS = ["A"] + [f"A+{g}" for g in GROUPS] + ["B"]
MODELS = ["ridge", "random_forest", "lightgbm"]
TARGET = "revenue"


def main() -> None:
    t0 = time.time()
    daily = pd.read_parquet(ROOT / "data" / "processed" / "daily_kpis.parquet")
    sp = CFG["split"]
    test_dates = pd.date_range(sp["test_start"], sp["test_end"], freq="D")
    preds = []
    for h in CFG["horizons"]:
        for fs in SETS:
            log(f"ablation h={h} set={fs}")
            p, _, _ = ml_predictions(daily, TARGET, h, fs, MODELS, test_dates, explain=False)
            preds += p
    pred = pd.DataFrame(preds)
    scales = {o: mase_scale(daily[TARGET].loc[:o].to_numpy(float), M) for o in pred["origin"].unique()}
    pred["scale"] = pred["origin"].map(scales)
    pred.to_csv(RES / "predictions_ablation.csv", index=False)

    # determinism check against the main backtest
    main_pred = pd.read_csv(RES / "predictions.csv", parse_dates=["date"])
    key = ["h", "model", "feature_set", "date"]
    chk = pred[pred.feature_set.isin(["A", "B"])].merge(
        main_pred[(main_pred.target == TARGET) & main_pred.feature_set.isin(["A", "B"])][key + ["yhat"]],
        on=key, suffixes=("", "_main"))
    max_diff = float(np.max(np.abs(chk["yhat"] - chk["yhat_main"])))
    log(f"determinism check: {len(chk)} forecasts compared, max |diff| = {max_diff:.3e}")

    rows = []
    for (h, model, fs), g in pred.groupby(["h", "model", "feature_set"]):
        rows.append({"target": TARGET, "h": h, "model": model, "feature_set": fs,
                     **point_metrics(g["y"].to_numpy(), g["yhat"].to_numpy(), g["scale"].to_numpy())})
    met = pd.DataFrame(rows)
    base = met[met.feature_set == "A"].set_index(["h", "model"])["mase"]
    met["mase_change_vs_A_pct"] = 100 * (met["mase"] / [base[(h, m)] for h, m in zip(met.h, met.model)] - 1)

    pred["e"] = pred["y"] - pred["yhat"]
    dms = []
    for (h, model), g in pred.groupby(["h", "model"]):
        E = g.pivot(index="date", columns="feature_set", values="e")
        for fs in SETS[1:]:
            for loss in ["se", "ae"]:
                for kernel in ["rectangular", "bartlett"]:
                    r = dm_hln(E[fs].to_numpy(), E["A"].to_numpy(), int(h), loss, kernel=kernel)
                    dms.append({"h": h, "model": model, "feature_set": fs, "loss": loss, "kernel": kernel,
                                "family": "ablation_vs_A", **r})
    dm = pd.DataFrame(dms)
    dm["p_holm"] = dm.groupby(["h", "family", "loss", "kernel"])["p_value"].transform(holm)
    dm.round(6).to_csv(RES / "ablation_dm_tests.csv", index=False)  # long format incl. variance_status
    dm["variance_fallback"] = (dm["variance_status"] != "ok").astype(int)
    dm["col"] = dm["loss"] + np.where(dm["kernel"] == "bartlett", "_bartlett", "")
    wide = dm.pivot_table(index=["h", "model", "feature_set"], columns="col",
                          values=["dm_hln", "p_value", "p_holm", "mean_loss_diff", "variance_fallback"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    out = met.merge(wide.reset_index(), on=["h", "model", "feature_set"], how="left")
    out["determinism_max_abs_diff_vs_main"] = max_diff
    out.round(6).to_csv(RES / "ablation_groups.csv", index=False)
    pd.set_option("display.width", 220)
    print(out[["h", "model", "feature_set", "mase", "mase_change_vs_A_pct", "p_holm_se", "p_holm_ae",
               "p_holm_se_bartlett", "p_holm_ae_bartlett"]]
          .round(4).to_string(index=False))
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
