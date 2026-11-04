"""Run the whole pipeline from the raw Excel file to the paper tables, then pytest.

Usage (repo root):  .venv/Scripts/python.exe run_all.py [--skip-paper]
Steps and wall-clock time are written to results/run_all_log.csv.
data/raw is only read. Reference verification (paper/verify_refs.py) needs network access and is NOT part
of this pipeline; references.bib is versioned.
"""

from __future__ import annotations

import csv
import importlib
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

STEPS = [
    ("data_profile", lambda m: m.main(rebuild=True)),  # re-reads data/raw/online_retail_II.xlsx
    ("cleaning", lambda m: m.main()),
    ("kpis", lambda m: m.main()),
    ("eda", lambda m: m.main()),
    ("eda_figures", lambda m: m.main()),
    ("backtest", lambda m: m.main()),
    ("robustness", lambda m: m.main()),
    ("ablation", lambda m: m.main()),
    ("dm_variance_diagnostics", lambda m: m.main()),
    ("dm_fallback_report", lambda m: m.main()),
    ("figures", lambda m: m.main()),
    ("prediction_figures", lambda m: m.main()),
    ("paper_numbers", lambda m: m.main()),
]


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log = []
    t_all = time.time()
    for name, fn in STEPS:
        t0 = time.time()
        print(f"\n===== {name} =====", flush=True)
        fn(importlib.import_module(name))
        log.append({"step": name, "seconds": round(time.time() - t0, 1)})

    if "--skip-paper" not in sys.argv:
        tectonic = shutil.which("tectonic") or str(Path(os.environ.get("LOCALAPPDATA", "")) /
                                                   "tectonic-env" / "Library" / "bin" / "tectonic.exe")
        t0 = time.time()
        if Path(tectonic).exists():
            subprocess.run([tectonic, "main.tex"], cwd=ROOT / "paper", check=True)
            log.append({"step": "paper_pdf", "seconds": round(time.time() - t0, 1)})
            subprocess.run([sys.executable, str(ROOT / "paper" / "check_text.py")], cwd=ROOT, check=True)
        else:
            print("tectonic not found: paper PDF not built")

    import evaluate  # dm_hln variance status counts over the whole run (backtest, robustness, ablation)
    for status, cnt in sorted(evaluate.FALLBACK_COUNTER.items()):
        log.append({"step": f"dm_hln calls: {status}", "seconds": "", "count": cnt})

    t0 = time.time()
    rc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT).returncode
    log.append({"step": f"pytest (exit {rc})", "seconds": round(time.time() - t0, 1)})
    log.append({"step": "TOTAL", "seconds": round(time.time() - t_all, 1)})
    with open(ROOT / "results" / "run_all_log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "seconds", "count"], restval="")
        w.writeheader()
        w.writerows(log)
    for r in log:
        if r.get("count", "") != "":
            print(f"{r['step']:32s} {r['count']:>8d}")
        else:
            print(f"{r['step']:32s} {r['seconds']:>8.1f} s")
    sys.exit(rc)


if __name__ == "__main__":
    main()
