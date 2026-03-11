"""Pipeline: orchestrates OCR → AI → Organizer for a single file."""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

from brain.ai.base import AIProvider, AIError
from brain.config import AppConfig
from brain.ocr import extract_text, OCRError, configure_tesseract
from brain.organizer import organize_file, move_to_failed, OrganizerError

logger = logging.getLogger(__name__)

# Exponential backoff config for retryable AI errors
_RETRY_DELAYS = (5, 10, 20)  # seconds


class Pipeline:
    """Processes a single file through OCR, AI analysis, and vault organization."""

    def __init__(self, config: AppConfig, ai: AIProvider, dry_run: bool = False):
        self._config = config
        self._ai = ai
        self._dry_run = dry_run

        # Apply Tesseract binary path from config (Windows support)
        configure_tesseract(config.ocr.tesseract_cmd)

    def process(self, file_path: Path) -> bool:
        """Run the full pipeline for one file.

        Returns:
            True on success, False on recoverable failure (file moved to _failed/).

        Raises:
            PipelineError: On an unexpected / unrecoverable error.
        """
        logger.info("Processing: %s", file_path.name)

        # --- Stage 1: OCR ---
        try:
            ocr_result = extract_text(
                file_path,
                min_embedded_chars=self._config.ocr.min_embedded_text_chars,
                pdf_render_dpi=self._config.ocr.pdf_render_dpi,
                language=self._config.ocr.language,
            )
            logger.debug(
                "OCR complete — method: %s, chars: %d",
                ocr_result.method, len(ocr_result.text),
            )
        except OCRError as exc:
            logger.error("OCR failed for %s: %s", file_path.name, exc)
            self._fail_file(file_path, str(exc))
            return False

        # --- Stage 2: AI Analysis ---
        try:
            analysis = self._call_ai_with_retry(
                ocr_result.text, file_path.name, ocr_result.method
            )
            logger.debug("AI analysis: title=%r, category=%r", analysis.title, analysis.category)
        except AIError as exc:
            logger.error("AI analysis failed permanently for %s: %s", file_path.name, exc)
            if exc.raw:
                logger.debug("Raw AI response: %s", exc.raw[:500])
            self._fail_file(file_path, str(exc))
            return False

        # --- Stage 3: Organize ---
        try:
            md_path, moved_source = organize_file(
                source_path=file_path,
                analysis=analysis,
                ocr_text=ocr_result.text,
                config=self._config,
                ocr_method=ocr_result.method,
                dry_run=self._dry_run,
            )
            logger.info(
                "Done: %s → %s (markdown: %s)",
                file_path.name,
                moved_source.relative_to(self._config.watch.vault_path),
                md_path.name,
            )
            return True
        except OrganizerError as exc:
            logger.error("Organizer failed for %s: %s", file_path.name, exc)
            self._fail_file(file_path, str(exc))
            return False
        except Exception as exc:
            raise PipelineError(
                f"Unexpected error organizing {file_path.name}"
            ) from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_ai_with_retry(
        self, text: str, filename: str, ocr_method: str
    ) -> "FileAnalysis":  # noqa: F821 (imported above via AIProvider)
        from brain.ai.base import FileAnalysis  # local to avoid circular issues
        last_exc: AIError | None = None

        for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
            try:
                return self._ai.analyze(text, filename, ocr_method)
            except AIError as exc:
                last_exc = exc
                if not exc.retryable or delay is None:
                    break
                logger.warning(
                    "AI call failed (attempt %d/%d), retrying in %ds: %s",
                    attempt, len(_RETRY_DELAYS) + 1, delay, exc,
                )
                time.sleep(delay)

        raise last_exc  # type: ignore[misc]

    def _fail_file(self, file_path: Path, reason: str) -> None:
        if not self._dry_run:
            move_to_failed(file_path, self._config.watch.vault_path, reason)
        else:
            logger.info("[DRY RUN] Would move failed file: %s", file_path)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class PipelineError(Exception):
    """Raised for unexpected / unrecoverable pipeline errors."""
