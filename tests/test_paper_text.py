"""Regression guard: no text of main.tex is hidden by % comments, lost, or accidentally duplicated in main.pdf."""

import sys
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper"
sys.path.insert(0, str(PAPER))


@pytest.mark.skipif(not (PAPER / "main.pdf").exists(), reason="paper PDF not built")
def test_pdf_text_matches_tex():
    import check_text
    if (PAPER / "main.pdf").stat().st_mtime < (PAPER / "main.tex").stat().st_mtime:
        pytest.skip("main.pdf is older than main.tex; rebuild the PDF first")
    assert check_text.main() == 0
