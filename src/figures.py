"""Stage 6: figures, each tied to a research question. Reads only results/*.csv.

fig_rq1_mase_revenue.png      RQ1/RQ3  MASE of every model for daily revenue, by horizon
fig_rq2_b_vs_a.png            RQ2      relative MASE change of Model B vs Model A, DM-HLN significance
fig_rq3_shap_groups.png       RQ3      share of mean |SHAP| by feature group and horizon (LightGBM, set B)
fig_leakage_control.png       leakage  MASE with legal features vs same-day (forbidden) features
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"

# Categorical slots in fixed order (validated default palette, light surface).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
H_COLOR = {1: SERIES[0], 7: SERIES[1], 28: SERIES[2]}
MODEL_LABEL = {"seasonal_naive": "Seasonal naive", "ets": "ETS", "sarima": "SARIMA", "ridge": "Ridge",
               "random_forest": "Random forest", "lightgbm": "LightGBM"}
TARGET_LABEL = {"revenue": "Revenue", "orders": "Orders", "active_customers": "Active customers"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 300, "axes.titlesize": 8.5, "axes.titleweight": "bold",
})


def label(model: str, fs: str) -> str:
    return MODEL_LABEL[model] + ("" if fs == "-" else f" ({fs})")


def fig_rq1(met: pd.DataFrame) -> None:
    m = met[(met.target == "revenue") & (met.subset == "all_days") & (met.feature_set != "leaky")].copy()
    m["cfg"] = [label(a, b) for a, b in zip(m.model, m.feature_set)]
    order = (m[m.h == 1].sort_values("mase", ascending=False)["cfg"].tolist())
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ypos = {c: i for i, c in enumerate(order)}
    offset = {1: -0.22, 7: 0.0, 28: 0.22}
    for h, g in m.groupby("h"):
        ax.scatter(g["mase"], g["cfg"].map(ypos) + offset[h], s=20, color=H_COLOR[h], label=f"h = {h}",
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(range(len(order)), order)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("MASE (lower is better)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, handletextpad=0.1,
              fontsize=7, columnspacing=0.8)
    ax.set_title("Daily revenue, MASE by horizon", loc="left", pad=16)
    fig.tight_layout()
    fig.savefig(FIG / "fig_rq1_mase_revenue.png", bbox_inches="tight")
    plt.close(fig)


def fig_rq2(met: pd.DataFrame, dm: pd.DataFrame, horizons=(1, 7, 28), name: str = "fig_rq2_b_vs_a.png") -> None:
    m = met[(met.subset == "all_days") & met.feature_set.isin(["A", "B"]) & met.h.isin(horizons)]
    w = m.pivot_table(index=["target", "h", "model"], columns="feature_set", values="mase").reset_index()
    w["rel"] = 100 * (w["B"] - w["A"]) / w["A"]
    d = dm[(dm.family == "B_vs_A") & (dm.loss == "se")].copy()
    d["model"] = d["model_1"].str.replace(r"\[B\]", "", regex=True)
    w = w.merge(d[["target", "h", "model", "p_holm"]], on=["target", "h", "model"])
    targets = [t for t in TARGET_LABEL if t in w.target.unique()]
    models = ["ridge", "random_forest", "lightgbm"]
    fig, axes = plt.subplots(1, len(targets), figsize=(7.16, 2.4), sharey=True)
    lim = np.ceil(w["rel"].abs().max() / 5) * 5 + 2
    for ax, t in zip(np.atleast_1d(axes), targets):
        g = w[w.target == t]
        hs = sorted(g.h.unique())
        for j, h in enumerate(hs):
            gh = g[g.h == h].set_index("model").loc[models]
            y = np.arange(len(models)) + (j - (len(hs) - 1) / 2) * 0.26
            ax.barh(y, gh["rel"], height=0.24, color=H_COLOR[h], label=f"h = {h}", edgecolor="white", lw=0.8)
            for yy, (r, p) in zip(y, gh[["rel", "p_holm"]].to_numpy()):
                if p < 0.05:
                    ax.text(r + (0.6 if r >= 0 else -0.6), yy, "*", va="center",
                            ha="left" if r >= 0 else "right", fontsize=9, color=INK)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_yticks(range(len(models)), [MODEL_LABEL[x] for x in models])
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(TARGET_LABEL[t], loc="left")
        ax.set_xlabel("MASE change B vs A, %")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.text(0.99, 0.01, "Negative = Model B more accurate.  * DM-HLN, Holm-adjusted p < 0.05",
             ha="right", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / name)
    plt.close(fig)


def fig_rq3(sg: pd.DataFrame) -> None:
    s = sg[(sg.target == "revenue") & (sg.feature_set == "B")]
    groups = (s.groupby("group")["share"].mean().sort_values(ascending=False).index.tolist())
    color = {g: SERIES[i] for i, g in enumerate(groups)}
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    hs = sorted(s.h.unique())
    for i, h in enumerate(hs):
        left = 0.0
        sh = s[s.h == h].set_index("group")["share"]
        for g in groups:
            v = float(sh.get(g, 0.0))
            ax.barh(i, v * 100, left=left, color=color[g], edgecolor="white", lw=1.5, height=0.6,
                    label=g.replace("_", " ") if i == 0 else None)
            if v >= 0.08:
                ax.text(left + v * 50, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=6.5, color="white")
            left += v * 100
    ax.set_yticks(range(len(hs)), [f"h = {h}" for h in hs])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of mean |SHAP|, %")
    ax.legend(frameon=False, fontsize=6.5, ncol=4, loc="lower center", bbox_to_anchor=(0.45, 1.02),
              handlelength=1, columnspacing=0.8)
    ax.set_title("Revenue, LightGBM (B): SHAP share by group", loc="left", pad=28)
    fig.tight_layout()
    fig.savefig(FIG / "fig_rq3_shap_groups.png", bbox_inches="tight")
    plt.close(fig)


def fig_leak(leak: pd.DataFrame) -> None:
    lk = leak[leak.subset == "all_days"]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    rows = [(m, h) for m in ["ridge", "lightgbm"] for h in sorted(lk.h.unique())]
    for i, (m, h) in enumerate(rows):
        g = lk[(lk.model == m) & (lk.h == h)].set_index("feature_set")["mase"]
        ax.plot([g["leaky"], g["B"]], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter(g["B"], i, s=24, color=SERIES[0], zorder=3, edgecolor="white", lw=1,
                   label="Model B: forecast-origin features" if i == 0 else None)
        ax.scatter(g["leaky"], i, s=24, color=SERIES[1], zorder=3, edgecolor="white", lw=1,
                   label="Model B + target-day KPIs (leak)" if i == 0 else None)
    ax.set_yticks(range(len(rows)), [f"{MODEL_LABEL[m]}, h = {h}" for m, h in rows])
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("MASE, daily revenue")
    ax.legend(frameon=False, fontsize=6.3, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=1,
              handletextpad=0.1)
    ax.set_title("Leakage control, daily revenue", loc="left", pad=26)
    fig.tight_layout()
    fig.savefig(FIG / "fig_leakage_control.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    met = pd.read_csv(RES / "metrics.csv")
    dm = pd.read_csv(RES / "dm_tests.csv")
    fig_rq1(met)
    fig_rq2(met, dm)  # thesis: all horizons
    fig_rq2(met, dm, horizons=(1, 7), name="fig_rq2_b_vs_a_h17.png")  # paper: inference only for h=1, 7
    fig_rq3(pd.read_csv(RES / "shap_groups.csv"))
    fig_leak(pd.read_csv(RES / "leakage_control.csv"))
    print("figures written")


if __name__ == "__main__":
    main()
