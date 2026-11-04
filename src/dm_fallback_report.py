"""Count DM tests in which the HAC variance was <= 0 and gamma_0/n was used (variance_status != "ok").

Reads: dm_tests.csv (protocol), dm_tests_bartlett.csv, dm_tests_no_saturdays.csv, ablation_dm_tests.csv
Writes: results/dm_fallback_counts.csv (per source x kernel x loss x h), results/dm_fallback_cases.csv (each case)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SOURCES = [("main", "dm_tests.csv", "rectangular"), ("bartlett", "dm_tests_bartlett.csv", "bartlett"),
           ("no_saturdays", "dm_tests_no_saturdays.csv", "rectangular"), ("ablation", "ablation_dm_tests.csv", None)]


def main() -> None:
    frames = []
    for name, f, kernel in SOURCES:
        d = pd.read_csv(RES / f)
        if "variance_status" not in d.columns:
            raise RuntimeError(f"{f} has no variance_status column; rerun the pipeline")
        if kernel is not None:
            d["kernel"] = kernel
        d["source"] = name
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    counts = (d.groupby(["source", "kernel", "loss", "h"])
              .agg(n_tests=("variance_status", "size"),
                   n_fallback_gamma0=("variance_status", lambda s: int((s == "fallback_gamma0").sum())),
                   n_zero_variance=("variance_status", lambda s: int((s == "zero_variance").sum())))
              .reset_index())
    counts.to_csv(RES / "dm_fallback_counts.csv", index=False)
    cases = d[d["variance_status"] != "ok"]
    keep = [c for c in ["source", "kernel", "loss", "h", "target", "family", "model_1", "model_2", "model",
                        "feature_set", "variance_status", "p_value", "p_holm"] if c in cases.columns]
    cases[keep].to_csv(RES / "dm_fallback_cases.csv", index=False)
    by_h = counts.groupby("h")[["n_tests", "n_fallback_gamma0", "n_zero_variance"]].sum()
    print(by_h.to_string())
    print(cases[keep].to_string(index=False) if len(cases) else "no fallback cases")


if __name__ == "__main__":
    main()
