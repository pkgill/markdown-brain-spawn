"""Tests for brain/ocr.py.

Tests that require Tesseract are marked @pytest.mark.tesseract.
Run without Tesseract:  pytest -m "not tesseract"
Run all:                pytest
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from brain.ocr import extract_text, OCRError, UnsupportedFormatError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Extension / existence guards
# ---------------------------------------------------------------------------

def test_unsupported_extension_raises(tmp_path):
    f = tmp_path / "data.zip"
    f.write_text("content")
    with pytest.raises(UnsupportedFormatError, match="Unsupported"):
        extract_text(f)


def test_missing_file_raises(tmp_path):
    with pytest.raises(OCRError, match="not found"):
        extract_text(tmp_path / "ghost.pdf")


# ---------------------------------------------------------------------------
# PDF — embedded text (mocked pdfplumber)
# ---------------------------------------------------------------------------

class _FakePage:
    def __init__(self, text):
        self._text = text
    def extract_text(self):
        return self._text


class _FakePDF:
    def __init__(self, text):
        self.pages = [_FakePage(text)]
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass


def test_pdf_embedded_text_uses_pdfplumber(tmp_path):
    fake_pdf_path = tmp_path / "test.pdf"
    fake_pdf_path.write_bytes(b"%PDF-1.4 fake content")

    rich_text = "Hello world " * 20  # well above 100 char threshold
    with patch("pdfplumber.open", return_value=_FakePDF(rich_text)):
        result = extract_text(fake_pdf_path)

    assert result.method == "pdfplumber"
    assert "Hello world" in result.text


def test_pdf_sparse_text_falls_back_to_tesseract(tmp_path):
    fake_pdf_path = tmp_path / "scan.pdf"
    fake_pdf_path.write_bytes(b"%PDF-1.4 minimal")

    sparse_pdf = _FakePDF("ab")  # only 2 chars → below threshold

    mock_page = MagicMock()
    mock_page.get_pixmap.return_value = MagicMock(
        width=100, height=100,
        samples=b"\xFF" * (100 * 100 * 3),
    )
    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_doc.__len__ = MagicMock(return_value=1)

    with (
        patch("pdfplumber.open", return_value=sparse_pdf),
        patch("fitz.open", return_value=mock_doc),
        patch("pytesseract.image_to_string", return_value="scanned text here"),
        patch("pytesseract.get_tesseract_version", return_value="5.0"),
    ):
        result = extract_text(fake_pdf_path, min_embedded_chars=100)

    assert result.method == "tesseract_pdf"
    assert "scanned text" in result.text


# ---------------------------------------------------------------------------
# Image — Tesseract
# ---------------------------------------------------------------------------

@pytest.mark.tesseract
def test_extract_image_real_tesseract(tmp_path):
    """Integration test: requires Tesseract installed."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Hello Tesseract", fill="black")
    img_path = tmp_path / "test.png"
    img.save(str(img_path))

    result = extract_text(img_path)
    assert result.method == "tesseract_image"
    assert "Hello" in result.text or len(result.text) > 0


def test_extract_image_mocked(tmp_path):
    img_path = tmp_path / "test.png"
    from PIL import Image
    Image.new("RGB", (100, 100), color="white").save(str(img_path))

    mock_data = {"conf": [90, 85, 92]}
    with (
        patch("pytesseract.get_tesseract_version", return_value="5.0"),
        patch("pytesseract.image_to_data", return_value=mock_data),
        patch("pytesseract.image_to_string", return_value="mocked text"),
    ):
        result = extract_text(img_path)

    assert result.method == "tesseract_image"
    assert result.text == "mocked text"
    assert result.confidence == pytest.approx(89.0)


def test_tesseract_not_found_raises_helpful_error(tmp_path):
    img_path = tmp_path / "test.png"
    from PIL import Image
    Image.new("RGB", (100, 100), color="white").save(str(img_path))

    with patch(
        "pytesseract.get_tesseract_version",
        side_effect=Exception("not found"),
    ):
        with pytest.raises(OCRError, match="brew install tesseract|apt-get"):
            extract_text(img_path)
