"""Collect every number quoted in the paper from results/*.csv and generate LaTeX tables.

Outputs
  results/paper_numbers.csv   key, value, source, description  (the paper cites these keys in % comments)
  paper/tab_mase_revenue.tex  Table: MASE for daily revenue, all models x horizons
  paper/tab_b_vs_a.tex        Table: Model B vs Model A for daily revenue (MASE change, DM-HLN p, Holm)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, PAPER = ROOT / "results", ROOT / "paper"
PAPER_H = [1, 7]  # horizons with significance statements in the paper; h=28 is descriptive (see results/h28_inference.md)
LABEL = {"seasonal_naive": "Seasonal naive", "ets": "ETS", "sarima": "SARIMA", "ridge": "Ridge",
         "random_forest": "Random forest", "lightgbm": "LightGBM"}


def main() -> None:
    rows: list[dict] = []

    def add(key, value, source, desc):
        rows.append({"key": key, "value": value, "source": source, "description": desc})

    prof = pd.read_csv(RES / "data_profile.csv").set_index("metric")["value"]
    add("raw_rows", int(prof["n_rows_total"]), "data_profile.csv", "rows in raw file (both sheets)")
    add("overlap_rows", 22523, "cleaning_log.csv:R1", "rows duplicated across sheets")
    add("missing_cust_rev_share_pct", round(100 * float(prof["share_net_revenue_missing_customer_id"]), 1),
        "data_profile.csv", "% of raw net revenue with missing Customer ID")
    log = pd.read_csv(RES / "cleaning_log.csv").set_index("step")
    add("overlap_rows", int(log.at["R1", "n_rows_removed"]), "cleaning_log.csv", "R1 rows removed")
    rows.pop(1)  # keep the value read from the log, not the literal
    add("clean_rows", int(log.at["R7", "n_rows_after"]), "cleaning_log.csv", "rows after R1-R7")
    add("cancel_rows", int(log.at["R8", "n_rows_flagged"]), "cleaning_log.csv", "cancellation rows kept as flag")

    summ = pd.read_csv(RES / "daily_kpis_summary.csv").set_index("kpi")
    add("calendar_days", int(summ.at["calendar", "count"]), "daily_kpis_summary.csv", "days in KPI table")
    add("trading_days", int(summ.at["trading_days", "count"]), "daily_kpis_summary.csv", "trading days")
    add("sat_trading_days", int(summ.at["saturday_trading_days", "count"]), "daily_kpis_summary.csv",
        "Saturdays with sales")
    add("rev_median", round(float(summ.at["revenue", "50%"]), 0), "daily_kpis_summary.csv", "median daily revenue")

    eda = pd.read_csv(RES / "eda_seasonality.csv")
    e = eda[eda.series == "calendar_7day"].set_index("stat")["value"]
    add("acf7", round(float(e["acf_lag7"]), 2), "eda_seasonality.csv", "ACF lag 7, calendar series")
    add("snaive_mae_m7", round(float(e["snaive_mae_m7"]), 0), "eda_seasonality.csv", "in-sample SN MAE m=7")
    add("snaive_mae_m6", round(float(e["snaive_mae_m6"]), 0), "eda_seasonality.csv", "in-sample SN MAE m=6")
    add("naive1_mae", round(float(e["naive1_mae"]), 0), "eda_seasonality.csv", "in-sample naive-1 MAE")

    met = pd.read_csv(RES / "metrics.csv")
    pred_n = met[(met.subset == "all_days")]["n"].unique()
    assert len(pred_n) == 1
    add("test_days", int(pred_n[0]), "metrics.csv", "target days per horizon")
    rv = met[(met.target == "revenue") & (met.subset == "all_days")]

    def mase(model, fs, h, subset="all_days", target="revenue"):
        g = met[(met.target == target) & (met.subset == subset) & (met.model == model)
                & (met.feature_set == fs) & (met.h == h)]
        return float(g["mase"].iloc[0])

    for h in [1, 7, 28]:
        for m, fs in [("seasonal_naive", "-"), ("ets", "-"), ("sarima", "-"), ("random_forest", "A"),
                      ("random_forest", "B"), ("ridge", "A"), ("lightgbm", "A"), ("lightgbm", "B")]:
            add(f"mase_rev_{m}_{fs}_h{h}", round(mase(m, fs, h), 3), "metrics.csv", "revenue MASE, all days")
        add(f"mase_rev_nosat_sn_h{h}", round(mase("seasonal_naive", "-", h, "no_saturdays"), 3), "metrics.csv",
            "revenue MASE excl. Saturdays, seasonal naive")
        add(f"mase_rev_nosat_ridgeA_h{h}", round(mase("ridge", "A", h, "no_saturdays"), 3), "metrics.csv",
            "revenue MASE excl. Saturdays, Ridge A")
        add(f"mase_rev_nosat_rfA_h{h}", round(mase("random_forest", "A", h, "no_saturdays"), 3), "metrics.csv",
            "revenue MASE excl. Saturdays, RF A")
    best = rv[rv.feature_set != "leaky"].loc[lambda d: d.groupby("h")["mase"].idxmin()]
    for _, r in best.iterrows():
        add(f"best_rev_h{r.h}", f"{r.model}[{r.feature_set}]", "metrics.csv", "lowest revenue MASE")
    for h in [1, 7, 28]:
        add(f"wape_rev_rfA_h{h}", round(float(rv[(rv.model == "random_forest") & (rv.feature_set == "A")
                                                 & (rv.h == h)]["wape"].iloc[0]), 3), "metrics.csv", "WAPE RF A")

    dm = pd.read_csv(RES / "dm_tests.csv")
    for h in [1, 7, 28]:
        d = dm[(dm.target == "revenue") & (dm.h == h) & (dm.loss == "se")]
        sn = d[d.family == "vs_seasonal_naive"]
        add(f"n_sig_vs_sn_h{h}", int(((sn.p_holm < 0.05) & (sn.mean_loss_diff < 0)).sum()), "dm_tests.csv",
            "revenue: models significantly better than SN (Holm)")
        add(f"n_vs_sn_h{h}", len(sn), "dm_tests.csv", "revenue: comparisons vs SN")
        et = d[d.family == "vs_ets"]
        add(f"n_sig_vs_ets_h{h}", int(((et.p_holm < 0.05) & (et.mean_loss_diff < 0)).sum()), "dm_tests.csv",
            "revenue: ML models significantly better than ETS (Holm)")
        add(f"n_rawsig_vs_ets_h{h}", int(((et.p_value < 0.05) & (et.mean_loss_diff < 0)).sum()), "dm_tests.csv",
            "revenue: ML better than ETS, unadjusted p<0.05")
        add(f"n_vs_ets_h{h}", len(et), "dm_tests.csv", "revenue: ML comparisons vs ETS")
    sar28 = dm[(dm.target == "revenue") & (dm.h == 28) & (dm.loss == "se") & (dm.model_1 == "sarima[-]")
               & (dm.family == "vs_seasonal_naive")]
    add("p_holm_sarima_vs_sn_h28", round(float(sar28.p_holm.iloc[0]), 3), "dm_tests.csv", "SARIMA vs SN h=28")

    ba = dm[dm.family == "B_vs_A"]
    for loss in ["se", "ae"]:
        b = ba[ba.loss == loss]
        add(f"n_b_vs_a_{loss}", len(b), "dm_tests.csv", f"B vs A tests ({loss})")
        add(f"n_b_better_sig_{loss}", int(((b.p_holm < 0.05) & (b.mean_loss_diff < 0)).sum()), "dm_tests.csv",
            f"B significantly better than A (Holm, {loss})")
        add(f"n_b_worse_sig_{loss}", int(((b.p_holm < 0.05) & (b.mean_loss_diff > 0)).sum()), "dm_tests.csv",
            f"B significantly worse than A (Holm, {loss})")
    dns = pd.read_csv(RES / "dm_tests_no_saturdays.csv")
    bns = dns[dns.family == "B_vs_A"]
    for loss in ["se", "ae"]:
        b = bns[bns.loss == loss]
        add(f"n_b_vs_a_nosat_{loss}", len(b), "dm_tests_no_saturdays.csv", f"B vs A tests excl. Saturdays ({loss})")
        add(f"n_b_sig_any_nosat_{loss}", int((b.p_holm < 0.05).sum()), "dm_tests_no_saturdays.csv",
            f"significant B vs A differences excl. Saturdays ({loss})")
    rf28 = ba[(ba.target == "revenue") & (ba.h == 28) & (ba.model_1 == "random_forest[B]")].set_index("loss")
    add("p_holm_rf28_se", round(float(rf28.at["se", "p_holm"]), 3), "dm_tests.csv", "RF B vs A, revenue h=28, SE")
    add("p_holm_rf28_ae", round(float(rf28.at["ae", "p_holm"]), 3), "dm_tests.csv", "RF B vs A, revenue h=28, AE")

    w = met[(met.subset == "all_days") & met.feature_set.isin(["A", "B"])].pivot_table(
        index=["target", "h", "model"], columns="feature_set", values="mase")
    rel = 100 * (w["B"] - w["A"]) / w["A"]
    add("rel_b_vs_a_min_pct", round(float(rel.min()), 1), "metrics.csv", "min % MASE change B vs A (all targets)")
    add("rel_b_vs_a_max_pct", round(float(rel.max()), 1), "metrics.csv", "max % MASE change B vs A (all targets)")
    rr = rel.loc["revenue"]
    add("rel_b_vs_a_rev_min_pct", round(float(rr.min()), 1), "metrics.csv", "revenue min % change B vs A")
    add("rel_b_vs_a_rev_max_pct", round(float(rr.max()), 1), "metrics.csv", "revenue max % change B vs A")

    leak = pd.read_csv(RES / "leakage_control.csv")
    lk = leak[leak.subset == "all_days"].pivot_table(index=["model", "h"], columns="feature_set", values="mase")
    ratio = lk["B"] / lk["leaky"]
    add("leak_mase_min", round(float(lk["leaky"].min()), 2), "leakage_control.csv", "min MASE with leak")
    add("leak_mase_max", round(float(lk["leaky"].max()), 2), "leakage_control.csv", "max MASE with leak")
    add("legal_mase_min", round(float(lk["B"].min()), 2), "leakage_control.csv", "min MASE legal B (ridge, lgbm)")
    add("legal_mase_max", round(float(lk["B"].max()), 2), "leakage_control.csv", "max MASE legal B (ridge, lgbm)")
    add("leak_ratio_min", round(float(ratio.min()), 1), "leakage_control.csv", "legal/leaky MASE ratio min")
    add("leak_ratio_max", round(float(ratio.max()), 1), "leakage_control.csv", "legal/leaky MASE ratio max")
    lw = leak[leak.subset == "all_days"].pivot_table(index=["model", "h"], columns="feature_set", values="wape")
    add("leak_wape_lgbm_h1", round(float(lw.loc[("lightgbm", 1), "leaky"]), 3), "leakage_control.csv", "WAPE leak")
    add("legal_wape_lgbm_h1", round(float(lw.loc[("lightgbm", 1), "B"]), 3), "leakage_control.csv", "WAPE legal")

    sg = pd.read_csv(RES / "shap_groups.csv")
    drivers = ["orders", "active_customers", "aov", "cancellation_rate"]
    for h in [1, 7, 28]:
        b = sg[(sg.target == "revenue") & (sg.feature_set == "B") & (sg.h == h)].set_index("group")["share"]
        a = sg[(sg.target == "revenue") & (sg.feature_set == "A") & (sg.h == h)].set_index("group")["share"]
        add(f"shap_drivers_B_h{h}_pct", round(100 * float(b[drivers].sum()), 0), "shap_groups.csv",
            "revenue LightGBM B: share of driver groups")
        add(f"shap_calendar_B_h{h}_pct", round(100 * float(b["calendar"]), 0), "shap_groups.csv", "calendar share B")
        add(f"shap_calendar_A_h{h}_pct", round(100 * float(a["calendar"]), 0), "shap_groups.csv", "calendar share A")
        add(f"shap_ownlags_A_h{h}_pct", round(100 * float(a["own_lags"]), 0), "shap_groups.csv", "own lags share A")

    pi = pd.read_csv(RES / "permutation_importance.csv")
    p = pi[(pi.target == "revenue") & (pi.feature_set == "B")]
    for (m, h), g in p.groupby(["model", "h"]):
        add(f"perm_top_{m}_h{h}", g.sort_values("mae_increase_mean").iloc[-1]["group"], "permutation_importance.csv",
            "group with largest MAE increase")

    # ---------------- Holm family sizes (as implemented in backtest.run_dm / ablation.py)
    add("holm_family_size_b_vs_a", int(ba[(ba.target == "revenue") & (ba.h == 1) & (ba.loss == "se")].shape[0]),
        "dm_tests.csv", "tests per Holm family, B vs A (target x h x type x loss)")

    # ---------------- Bartlett sensitivity of the main DM tests
    bt = pd.read_csv(RES / "dm_tests_bartlett.csv")
    mm = dm.merge(bt, on=["target", "h", "family", "loss", "model_1", "model_2"], suffixes=("", "_bart"))
    add("n_dm_tests_total", len(mm), "dm_tests.csv", "all DM tests (both losses)")
    add("n_dm_decisions_changed_bartlett", int(((mm.p_holm < 0.05) != (mm.p_holm_bart < 0.05)).sum()),
        "dm_tests_bartlett.csv", "Holm decisions (alpha .05) that differ between kernels")
    for h in [1, 7, 28]:
        sn = bt[(bt.target == "revenue") & (bt.h == h) & (bt.loss == "se") & (bt.family == "vs_seasonal_naive")]
        add(f"n_sig_vs_sn_bartlett_h{h}", int(((sn.p_holm < 0.05) & (sn.mean_loss_diff < 0)).sum()),
            "dm_tests_bartlett.csv", "revenue: better than SN (Holm, Bartlett)")
        et = bt[(bt.target == "revenue") & (bt.h == h) & (bt.loss == "se") & (bt.family == "vs_ets")]
        add(f"n_sig_vs_ets_bartlett_h{h}", int(((et.p_holm < 0.05) & (et.mean_loss_diff < 0)).sum()),
            "dm_tests_bartlett.csv", "revenue: ML better than ETS (Holm, Bartlett)")
    et28 = bt[(bt.target == "revenue") & (bt.h == 28) & (bt.loss == "se") & (bt.family == "vs_ets")]
    add("p_holm_vs_ets_h28_bartlett_min", round(float(et28.p_holm.min()), 3), "dm_tests_bartlett.csv", "min")
    add("p_holm_vs_ets_h28_bartlett_max", round(float(et28.p_holm.max()), 3), "dm_tests_bartlett.csv", "max")
    rfb = bt[(bt.target == "revenue") & (bt.h == 28) & (bt.loss == "se") & (bt.family == "B_vs_A")
             & (bt.model_1 == "random_forest[B]")]
    add("p_holm_rf28_se_bartlett", round(float(rfb.p_holm.iloc[0]), 3), "dm_tests_bartlett.csv", "RF B vs A h28")
    bb = bt[bt.family == "B_vs_A"]
    add("n_b_sig_any_bartlett", int((bb.p_holm < 0.05).sum()), "dm_tests_bartlett.csv", "B vs A significant, Bartlett")

    # ---------------- feature-group ablation (post hoc)
    ab = pd.read_csv(RES / "ablation_groups.csv")
    abx = ab[ab.feature_set != "A"]
    add("abl_n_tests_per_loss", len(abx), "ablation_groups.csv", "set-vs-A tests per loss and kernel")
    add("abl_holm_family_size", int(len(abx[abx.h == 1])), "ablation_groups.csv", "tests per Holm family (per h, loss)")
    add("abl_change_min_pct", round(float(abx.mase_change_vs_A_pct.min()), 1), "ablation_groups.csv", "min % change")
    add("abl_change_max_pct", round(float(abx.mase_change_vs_A_pct.max()), 1), "ablation_groups.csv", "max % change")
    for col, tag in [("p_holm_se", "se"), ("p_holm_ae", "ae"), ("p_holm_se_bartlett", "se_bartlett"),
                     ("p_holm_ae_bartlett", "ae_bartlett")]:
        better = (abx[col] < 0.05) & (abx[f"mean_loss_diff_{tag}"] < 0)
        worse = (abx[col] < 0.05) & (abx[f"mean_loss_diff_{tag}"] > 0)
        add(f"abl_n_better_sig_{tag}", int(better.sum()), "ablation_groups.csv", f"set better than A ({tag})")
        add(f"abl_n_worse_sig_{tag}", int(worse.sum()), "ablation_groups.csv", f"set worse than A ({tag})")
        add(f"abl_n_sig_h1h7_{tag}", int(((abx[col] < 0.05) & (abx.h < 28)).sum()), "ablation_groups.csv",
            f"significant at h=1,7 ({tag})")
    for model, fs in [("ridge", "A+active_customers"), ("random_forest", "A+cancellation_rate")]:
        r = abx[(abx.h == 28) & (abx.model == model) & (abx.feature_set == fs)].iloc[0]
        tag = f"{model}_{fs.replace('A+', '')}_h28"
        add(f"abl_change_{tag}_pct", round(float(r.mase_change_vs_A_pct), 1), "ablation_groups.csv", "% change")
        add(f"abl_p_holm_se_{tag}", round(float(r.p_holm_se), 3), "ablation_groups.csv", "Holm p, SE, rect.")
        add(f"abl_p_holm_se_bartlett_{tag}", round(float(r.p_holm_se_bartlett), 3), "ablation_groups.csv",
            "Holm p, SE, Bartlett")
    rb = abx[(abx.h == 28) & (abx.model == "random_forest") & (abx.feature_set == "B")].iloc[0]
    add("abl_p_holm_se_rf_B_h28", round(float(rb.p_holm_se), 3), "ablation_groups.csv",
        "RF B vs A h28 in the 15-test ablation family")
    add("abl_determinism_max_diff", float(ab.determinism_max_abs_diff_vs_main.iloc[0]), "ablation_groups.csv",
        "max |yhat diff| recomputed A,B vs main backtest")

    # ---------------- inference restricted to the paper horizons (h=1, 7); Holm families are per horizon,
    # so restricting the horizons does not change any adjusted p-value
    ph = ba[ba.h.isin(PAPER_H)]
    for loss in ["se", "ae"]:
        b = ph[ph.loss == loss]
        add(f"n_b_vs_a_h17_{loss}", len(b), "dm_tests.csv", f"B vs A tests at h=1,7 ({loss})")
        add(f"n_b_better_sig_h17_{loss}", int(((b.p_holm < 0.05) & (b.mean_loss_diff < 0)).sum()), "dm_tests.csv",
            f"B significantly better, h=1,7 ({loss})")
        add(f"n_b_worse_sig_h17_{loss}", int(((b.p_holm < 0.05) & (b.mean_loss_diff > 0)).sum()), "dm_tests.csv",
            f"B significantly worse, h=1,7 ({loss})")
        bn = bns[(bns.loss == loss) & bns.h.isin(PAPER_H)]
        add(f"n_b_vs_a_nosat_h17_{loss}", len(bn), "dm_tests_no_saturdays.csv", f"B vs A excl. Sat, h=1,7 ({loss})")
        add(f"n_b_sig_any_nosat_h17_{loss}", int((bn.p_holm < 0.05).sum()), "dm_tests_no_saturdays.csv",
            f"significant B vs A excl. Sat, h=1,7 ({loss})")
    bb17 = bb[bb.h.isin(PAPER_H)]
    add("n_b_sig_any_bartlett_h17", int((bb17.p_holm < 0.05).sum()), "dm_tests_bartlett.csv", "B vs A, Bartlett, h=1,7")
    rel17 = rel[rel.index.get_level_values("h").isin(PAPER_H)]
    add("rel_b_vs_a_h17_min_pct", round(float(rel17.min()), 1), "metrics.csv", "min % change B vs A, h=1,7, all targets")
    add("rel_b_vs_a_h17_max_pct", round(float(rel17.max()), 1), "metrics.csv", "max % change B vs A, h=1,7, all targets")
    rr17 = rel17.loc["revenue"]
    add("rel_b_vs_a_rev_h17_min_pct", round(float(rr17.min()), 1), "metrics.csv", "revenue min % change, h=1,7")
    add("rel_b_vs_a_rev_h17_max_pct", round(float(rr17.max()), 1), "metrics.csv", "revenue max % change, h=1,7")
    mm17 = mm[mm.h.isin(PAPER_H)]
    add("n_dm_tests_h17", len(mm17), "dm_tests.csv", "all DM tests at h=1,7 (both losses)")
    add("n_dm_decisions_changed_bartlett_h17", int(((mm17.p_holm < 0.05) != (mm17.p_holm_bart < 0.05)).sum()),
        "dm_tests_bartlett.csv", "decisions differing between kernels at h=1,7")
    ab17 = abx[abx.h.isin(PAPER_H)]
    add("abl_change_h17_min_pct", round(float(ab17.mase_change_vs_A_pct.min()), 1), "ablation_groups.csv", "h=1,7")
    add("abl_change_h17_max_pct", round(float(ab17.mase_change_vs_A_pct.max()), 1), "ablation_groups.csv", "h=1,7")
    add("abl_n_tests_h17_per_loss", len(ab17), "ablation_groups.csv", "set-vs-A tests at h=1,7 per loss and kernel")

    # ---------------- loss dependence of the baseline comparisons (revenue, h=1,7)
    for loss in ["se", "ae"]:
        for h in PAPER_H:
            x = dm[(dm.target == "revenue") & (dm.h == h) & (dm.loss == loss)]
            for fam, tag in [("vs_seasonal_naive", "sn"), ("vs_ets", "ets")]:
                f = x[x.family == fam]
                add(f"n_sig_vs_{tag}_{loss}_h{h}", int(((f.p_holm < 0.05) & (f.mean_loss_diff < 0)).sum()),
                    "dm_tests.csv", f"revenue: better than {tag} (Holm, {loss}), of {len(f)}")
    x = dm[(dm.target == "revenue") & (dm.loss == "ae") & (dm.family == "vs_seasonal_naive")]
    for h, m in [(1, "ets[-]"), (7, "ets[-]"), (7, "sarima[-]")]:
        add(f"p_holm_ae_{m[:-3]}_vs_sn_h{h}", round(float(x[(x.h == h) & (x.model_1 == m)].p_holm.iloc[0]), 3),
            "dm_tests.csv", "AE loss, Holm p vs seasonal naive")
    cs = pd.read_csv(RES / "eda_country_share.csv").set_index("Country")
    add("n_countries_clean", len(cs), "eda_country_share.csv", "countries in clean non-cancelled lines")
    add("uk_share_revenue_pct", round(100 * float(cs.at["United Kingdom", "share_revenue"]), 1),
        "eda_country_share.csv", "UK share of gross revenue (clean, KPI period)")
    add("uk_share_lines_pct", round(100 * float(cs.at["United Kingdom", "share_lines"]), 1),
        "eda_country_share.csv", "UK share of lines (clean, KPI period)")

    vs = pd.read_csv(RES / "dm_variance_summary.csv").set_index("h")
    add("dmvar_h28_n_tests", int(vs.at[28, "n_tests"]), "dm_variance_summary.csv", "comparisons diagnosed at h=28")
    add("dmvar_h28_share_below_half_pct", round(100 * float(vs.at[28, "share_rect_below_0_5"]), 0),
        "dm_variance_summary.csv", "% with rectangular variance < 0.5 x lag-0 variance")
    add("dmvar_h28_ratio_min", round(float(vs.at[28, "ratio_rect_min"]), 2), "dm_variance_summary.csv", "min ratio")
    add("dmvar_h28_n_nonpositive", int(vs.at[28, "n_rect_nonpositive"]), "dm_variance_summary.csv",
        "rectangular variance <= 0")
    add("dmvar_h7_ratio_min", round(float(vs.at[7, "ratio_rect_min"]), 2), "dm_variance_summary.csv", "min ratio h=7")
    dv = pd.read_csv(RES / "dm_variance_diagnostics.csv")
    dv28 = dv[dv.h == 28]
    add("dmvar_h28_n_main", int((dv28.family != "ablation_vs_A").sum()), "dm_variance_diagnostics.csv",
        "h=28 SE comparisons of the main tests (3 targets x 17)")
    add("dmvar_h28_n_ablation", int((dv28.family == "ablation_vs_A").sum()), "dm_variance_diagnostics.csv",
        "h=28 SE comparisons of the ablation (revenue)")
    add("dmvar_h28_n_small_positive", int(((dv28.ratio_rect > 0) & (dv28.ratio_rect < 0.5)).sum()),
        "dm_variance_diagnostics.csv", "0 < rect/lag0 ratio < 0.5 (SE)")
    add("dmvar_h28_n_negative_se", int((dv28.ratio_rect <= 0).sum()), "dm_variance_diagnostics.csv",
        "rect variance <= 0 (SE)")
    fb = pd.read_csv(RES / "dm_fallback_counts.csv")
    add("fallback_h28_ae", int(fb[(fb.h == 28) & (fb.loss == "ae")].n_fallback_gamma0.sum()), "dm_fallback_counts.csv",
        "negative rect variance, AE loss, h=28 (all sources)")
    add("fallback_h17_total", int(fb[fb.h.isin(PAPER_H)].n_fallback_gamma0.sum()), "dm_fallback_counts.csv",
        "negative-variance fallbacks at h=1,7 (all sources, both losses)")
    add("mase_rev_nosat_ridgeB_h1", round(float(met[(met.target == "revenue") & (met.subset == "no_saturdays")
                                                    & (met.model == "ridge") & (met.feature_set == "B")
                                                    & (met.h == 1)].mase.iloc[0]), 3),
        "metrics.csv", "revenue MASE excl. Saturdays, Ridge B, h=1 (lowest on subset)")

    out = pd.DataFrame(rows)
    out.to_csv(RES / "paper_numbers.csv", index=False)

    # ---------------- Table: ablation
    sets = [("A+orders", "Ord."), ("A+active_customers", "Cust."), ("A+aov", "AOV"),
            ("A+cancellation_rate", "Canc."), ("B", "All")]
    lines = [r"% Generated by src/paper_numbers.py from results/ablation_groups.csv (post hoc analysis)",
             r"\begin{table}[t]", r"\centering", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Feature-group ablation for daily revenue (post hoc): MASE change (\%) of Model~A plus one "
             r"driver group (orders, active customers, AOV, cancellation rate), or all four (= Model~B), relative "
             r"to Model~A. No difference is significant (Holm-adjusted DM-HLN; SE and AE loss; protocol and "
             r"Bartlett variance).}",
             r"\label{tab:abl}", r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"$h$ & Model & " + " & ".join(s for _, s in sets) + r" \\", r"\midrule"]
    for h in PAPER_H:
        for m in ["ridge", "random_forest", "lightgbm"]:
            cells = []
            for fs, _ in sets:
                r = abx[(abx.h == h) & (abx.model == m) & (abx.feature_set == fs)].iloc[0]
                v = round(float(r.mase_change_vs_A_pct), 1)
                s = "0.0" if v == 0 else f"{v:+.1f}"
                cells.append(s + ("$^{*}$" if r.p_holm_se < 0.05 else ""))
            short = {"ridge": "Ridge", "random_forest": "RF", "lightgbm": "LGBM"}[m]
            lines.append(f"{h} & {short} & " + " & ".join(cells) + r" \\")
        if h != PAPER_H[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (PAPER / "tab_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- Table: revenue MASE
    order = [("seasonal_naive", "-"), ("ets", "-"), ("sarima", "-"), ("ridge", "A"), ("ridge", "B"),
             ("random_forest", "A"), ("random_forest", "B"), ("lightgbm", "A"), ("lightgbm", "B")]
    t = rv[rv.feature_set != "leaky"].pivot_table(index=["model", "feature_set"], columns="h", values="mase")
    lines = [r"% Generated by src/paper_numbers.py from results/metrics.csv (target=revenue, subset=all_days)",
             r"\begin{table}[t]", r"\centering",
             r"\caption{Test-period MASE for daily revenue (" + str(int(pred_n[0])) +
             r" target days per horizon). Set A: revenue history and calendar; set B: A plus lagged driver KPIs. "
             r"Lower is better; best per horizon in bold (determined before rounding). Significance tests: see "
             r"text (ML vs.\ SARIMA not "
             r"tested); $h=28$ is descriptive ($p$-values not interpreted, Section~\ref{sec:limits}).}",
             r"\label{tab:mase}", r"\begin{tabular}{llccc}", r"\toprule",
             r"Model & Set & $h=1$ & $h=7$ & $h=28$ \\", r"\midrule"]
    for m, fs in order:
        cells = []
        for h in [1, 7, 28]:
            v = t.loc[(m, fs), h]
            s = f"{v:.3f}"
            cells.append(r"\textbf{" + s + "}" if v == t[h].min() else s)
        lines.append(f"{LABEL[m]} & {fs if fs != '-' else '--'} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (PAPER / "tab_mase_revenue.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- Table: B vs A for revenue
    d = ba[ba.target == "revenue"].copy()
    d["model"] = d["model_1"].str.replace(r"\[B\]", "", regex=True)
    lines = [r"% Generated by src/paper_numbers.py from results/metrics.csv and results/dm_tests.csv",
             r"\begin{table}[t]", r"\centering",
             r"\caption{Model B (with lagged driver KPIs) vs Model A for daily revenue. $\Delta$MASE $=$ "
             r"(B$-$A)/A; $p$: DM-HLN, Holm-adjusted within each target $\times$ horizon $\times$ comparison "
             r"type $\times$ loss family (3 tests: one per model); SE and AE loss.}",
             r"\label{tab:bva}", r"\begin{tabular}{llrcc}", r"\toprule",
             r"$h$ & Model & $\Delta$MASE, \% & $p_{\mathrm{SE}}$ & $p_{\mathrm{AE}}$ \\", r"\midrule"]
    for h in PAPER_H:
        for m in ["ridge", "random_forest", "lightgbm"]:
            pse = d[(d.h == h) & (d.model == m) & (d.loss == "se")]["p_holm"].iloc[0]
            pae = d[(d.h == h) & (d.model == m) & (d.loss == "ae")]["p_holm"].iloc[0]
            dv = round(float(rel.loc[("revenue", h, m)]), 1)
            ds = "0.0" if dv == 0 else f"{dv:+.1f}"
            lines.append(f"{h} & {LABEL[m]} & {ds} & {pse:.3f} & {pae:.3f}" + r" \\")
        if h != PAPER_H[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (PAPER / "tab_b_vs_a.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.set_option("display.width", 200, "display.max_rows", 200, "display.max_colwidth", 60)
    print(out[["key", "value", "source"]].to_string(index=False))


if __name__ == "__main__":
    main()
