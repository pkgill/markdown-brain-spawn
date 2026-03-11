"""Tests for brain/pipeline.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from brain.ai.base import FileAnalysis, AIError
from brain.ocr import OCRResult, OCRError
from brain.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ai():
    ai = MagicMock()
    ai.analyze.return_value = FileAnalysis(
        title="Test Doc",
        category="Personal",
        summary="A test document.",
        tags=["test"],
    )
    return ai


@pytest.fixture
def mock_ocr_result():
    return OCRResult(text="some extracted text", method="pdfplumber", page_count=1)


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF fake")
    return f


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

def test_process_success(app_config, mock_ai, mock_ocr_result, source_file):
    with patch("brain.pipeline.extract_text", return_value=mock_ocr_result):
        pipeline = Pipeline(app_config, mock_ai)
        result = pipeline.process(source_file)

    assert result is True
    # File should have been moved into the vault
    vault_files = list(app_config.watch.vault_path.rglob("*.pdf"))
    assert vault_files


def test_process_creates_markdown(app_config, mock_ai, mock_ocr_result, source_file):
    with patch("brain.pipeline.extract_text", return_value=mock_ocr_result):
        pipeline = Pipeline(app_config, mock_ai)
        pipeline.process(source_file)

    md_files = list(app_config.watch.vault_path.rglob("*.md"))
    assert md_files
    assert "Test Doc" in md_files[0].read_text()


# ---------------------------------------------------------------------------
# OCR failure
# ---------------------------------------------------------------------------

def test_process_ocr_failure_returns_false(app_config, mock_ai, source_file):
    with patch("brain.pipeline.extract_text", side_effect=OCRError("ocr failed")):
        pipeline = Pipeline(app_config, mock_ai)
        result = pipeline.process(source_file)

    assert result is False
    # File should be in _failed/
    failed = list(app_config.watch.vault_path.glob("_failed/*.pdf"))
    assert failed


# ---------------------------------------------------------------------------
# AI failure — non-retryable
# ---------------------------------------------------------------------------

def test_process_ai_non_retryable_failure_returns_false(
    app_config, mock_ai, mock_ocr_result, source_file
):
    mock_ai.analyze.side_effect = AIError("bad json", retryable=False)

    with patch("brain.pipeline.extract_text", return_value=mock_ocr_result):
        pipeline = Pipeline(app_config, mock_ai)
        result = pipeline.process(source_file)

    assert result is False


# ---------------------------------------------------------------------------
# AI failure — retryable with eventual success
# ---------------------------------------------------------------------------

def test_process_ai_retryable_succeeds_on_third_attempt(
    app_config, mock_ai, mock_ocr_result, source_file
):
    good_result = FileAnalysis(
        title="Retry Success",
        category="Personal",
        summary="Worked on third try.",
        tags=["retry"],
    )
    mock_ai.analyze.side_effect = [
        AIError("rate limit", retryable=True),
        AIError("rate limit", retryable=True),
        good_result,
    ]

    with (
        patch("brain.pipeline.extract_text", return_value=mock_ocr_result),
        patch("time.sleep"),   # Don't actually wait in tests
    ):
        pipeline = Pipeline(app_config, mock_ai)
        result = pipeline.process(source_file)

    assert result is True
    assert mock_ai.analyze.call_count == 3


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def test_process_dry_run_does_not_move_files(
    app_config, mock_ai, mock_ocr_result, source_file
):
    with patch("brain.pipeline.extract_text", return_value=mock_ocr_result):
        pipeline = Pipeline(app_config, mock_ai, dry_run=True)
        result = pipeline.process(source_file)

    assert result is True
    assert source_file.exists()   # not moved
    vault_files = list(app_config.watch.vault_path.rglob("*.pdf"))
    assert not vault_files
