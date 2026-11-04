"""Correctness of the cleaning stage. Requires data produced by src/data_profile.py and src/cleaning.py."""

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cleaning as C  # noqa: E402

PROTOCOL = yaml.safe_load((ROOT / "protocol.yaml").read_text(encoding="utf-8"))
needs_data = pytest.mark.skipif(not C.INTERIM_CLEAN.exists(), reason="clean data not built")


def test_raw_file_unchanged():
    raw = ROOT / PROTOCOL["data"]["raw_file"]
    if not raw.exists():
        pytest.skip("raw file not present")
    h = hashlib.sha256(raw.read_bytes()).hexdigest().upper()
    assert h == PROTOCOL["data"]["raw_sha256"], "data/raw file was modified"


@pytest.fixture(scope="module")
def clean():
    return pd.read_parquet(C.INTERIM_CLEAN)


@needs_data
def test_no_sheet_overlap(clean):
    old = clean[clean["source_sheet"] == C.SHEET_OLD]
    assert old["InvoiceDate"].max() < C.OVERLAP_START


@needs_data
def test_no_exact_duplicates(clean):
    assert not clean.duplicated(C.DATA_COLS).any()


@needs_data
def test_no_nonsales_rows(clean):
    assert (clean["Price"] > 0).all()
    assert not clean["Invoice"].str.startswith("A").any()
    assert not clean["StockCode"].isin(C.ACCOUNTING_CODES | C.POSTAGE_CODES).any()
    assert (clean["Quantity"].abs() < C.EXTREME_QTY).all()


@needs_data
def test_quantity_sign_matches_cancellation(clean):
    assert (clean.loc[clean["is_cancellation"], "Quantity"] < 0).all()
    assert (clean.loc[~clean["is_cancellation"], "Quantity"] > 0).all()


@needs_data
def test_log_is_consistent(clean):
    log = pd.read_csv(C.LOG)
    drops = log[log["action"] == "drop"]
    assert (drops["n_rows_before"] - drops["n_rows_removed"] == drops["n_rows_after"]).all()
    assert (drops["n_rows_after"].iloc[:-1].to_numpy() == drops["n_rows_before"].iloc[1:].to_numpy()).all()
    assert drops["n_rows_after"].iloc[-1] == len(clean)
    raw_n = len(pd.read_parquet(C.INTERIM_RAW, columns=["Invoice"]))
    assert drops["n_rows_before"].iloc[0] == raw_n
