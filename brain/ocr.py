"""OCR module: extracts text from PDFs and images.

Strategy:
  PDF  → try pdfplumber (embedded text); if sparse, fall back to
         PyMuPDF page rendering + Tesseract per page.
  Image → preprocess with Pillow, then Tesseract.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp", ".heic", ".heif"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class OCRResult:
    text: str
    method: str          # "pdfplumber" | "tesseract_pdf" | "tesseract_image"
    page_count: int | None = None
    confidence: float | None = None   # Tesseract mean confidence 0-100 if available


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_text(path: Path, min_embedded_chars: int = 100,
                 pdf_render_dpi: int = 300, language: str = "eng") -> OCRResult:
    """Extract text from a PDF or image file.

    Args:
        path: Path to the file.
        min_embedded_chars: Minimum chars from pdfplumber before triggering Tesseract fallback.
        pdf_render_dpi: DPI for rendering PDF pages to images (Tesseract fallback).
        language: Tesseract language string, e.g. "eng" or "eng+fra".

    Returns:
        OCRResult with extracted text and method metadata.

    Raises:
        OCRError: If the file cannot be processed.
    """
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise OCRError(
            f"Unsupported file type: {suffix!r}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if not path.exists():
        raise OCRError(f"File not found: {path}")

    if suffix in SUPPORTED_PDF_EXTENSIONS:
        return _extract_pdf(path, min_embedded_chars, pdf_render_dpi, language)
    return _extract_image(path, language)


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def _extract_pdf(path: Path, min_embedded_chars: int,
                 pdf_render_dpi: int, language: str) -> OCRResult:
    if _pdf_has_embedded_text(path, min_embedded_chars):
        logger.debug("PDF has embedded text, using pdfplumber: %s", path.name)
        return _extract_pdf_with_pdfplumber(path)
    logger.debug("PDF appears to be scanned, using Tesseract: %s", path.name)
    return _extract_pdf_with_tesseract(path, pdf_render_dpi, language)


def _pdf_has_embedded_text(path: Path, min_chars: int) -> bool:
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            if not pdf.pages:
                return False
            # Sample first 3 pages
            sample = pdf.pages[:3]
            total_chars = sum(
                len(p.extract_text() or "") for p in sample
            )
            return total_chars >= min_chars
    except Exception as exc:
        logger.warning("pdfplumber check failed for %s: %s", path.name, exc)
        return False


def _extract_pdf_with_pdfplumber(path: Path) -> OCRResult:
    try:
        import pdfplumber
        pages_text: list[str] = []
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages_text.append(text)

        full_text = "\n\n".join(t for t in pages_text if t.strip())
        return OCRResult(text=full_text.strip(), method="pdfplumber", page_count=page_count)
    except Exception as exc:
        raise OCRError(f"pdfplumber failed on {path.name}: {exc}") from exc


def _extract_pdf_with_tesseract(path: Path, dpi: int, language: str) -> OCRResult:
    _ensure_tesseract()
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image

        pages_text: list[str] = []
        doc = fitz.open(str(path))
        page_count = len(doc)

        mat = fitz.Matrix(dpi / 72, dpi / 72)   # 72 dpi is the PDF base
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = _preprocess_image(img)
            text = pytesseract.image_to_string(img, lang=language)
            pages_text.append(text)

        doc.close()
        full_text = "\n\n".join(t for t in pages_text if t.strip())
        return OCRResult(text=full_text.strip(), method="tesseract_pdf", page_count=page_count)
    except OCRError:
        raise
    except Exception as exc:
        raise OCRError(f"Tesseract PDF extraction failed on {path.name}: {exc}") from exc


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------

def _extract_image(path: Path, language: str) -> OCRResult:
    _ensure_tesseract()
    try:
        import pytesseract
        from PIL import Image

        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            pass

        img = Image.open(path)
        img = _preprocess_image(img)

        # image_to_data gives per-word confidence; image_to_string gives clean text
        data = pytesseract.image_to_data(
            img, lang=language, output_type=pytesseract.Output.DICT
        )
        text = pytesseract.image_to_string(img, lang=language)

        # Compute mean confidence (filter out -1 sentinel values)
        confidences = [c for c in data["conf"] if isinstance(c, (int, float)) and c >= 0]
        mean_conf = sum(confidences) / len(confidences) if confidences else None

        return OCRResult(
            text=text.strip(),
            method="tesseract_image",
            page_count=1,
            confidence=mean_conf,
        )
    except OCRError:
        raise
    except Exception as exc:
        raise OCRError(f"Tesseract image extraction failed on {path.name}: {exc}") from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preprocess_image(img: "Image.Image") -> "Image.Image":
    """Convert to grayscale and boost contrast — Tesseract performs better on preprocessed images."""
    from PIL import ImageEnhance
    img = img.convert("L")   # Grayscale
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    return img


def _ensure_tesseract() -> None:
    """Raise a helpful OCRError if the Tesseract binary is not available."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise OCRError(
            "Tesseract OCR binary not found or not working.\n"
            "  macOS:   brew install tesseract\n"
            "  Ubuntu:  sudo apt-get install tesseract-ocr\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "           Then set ocr.tesseract_cmd in config.yaml to the .exe path."
        ) from exc


def configure_tesseract(cmd: str | None) -> None:
    """Set the Tesseract binary path if provided in config (Windows support)."""
    if cmd:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = cmd


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class OCRError(Exception):
    """Raised when OCR extraction fails for any reason."""
