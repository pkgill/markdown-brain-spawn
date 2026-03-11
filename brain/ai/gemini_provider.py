"""Google Gemini implementation of AIProvider."""

from __future__ import annotations

import logging

from brain.ai.base import (
    AIProvider,
    AIError,
    FileAnalysis,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from brain.ai.claude_provider import _parse_response  # shared parsing logic
from brain.config import AIConfig

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 8_000


class GeminiProvider(AIProvider):
    def __init__(self, config: AIConfig):
        self._config = config
        self._model = None   # Lazy init so import errors surface clearly

    def _get_model(self):
        if self._model is None:
            try:
                import google.generativeai as genai
            except ImportError as exc:
                raise AIError(
                    "google-generativeai is not installed. "
                    "Run: pip install google-generativeai"
                ) from exc

            genai.configure(api_key=self._config.api_key)
            self._model = genai.GenerativeModel(
                model_name=self._config.model,
                generation_config={
                    "temperature": self._config.temperature,
                    "max_output_tokens": self._config.max_tokens,
                    "response_mime_type": "application/json",
                },
                system_instruction=SYSTEM_PROMPT,
            )
        return self._model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, text: str, source_filename: str,
                ocr_method: str = "unknown") -> FileAnalysis:
        """Call the Gemini API and parse the JSON response."""
        user_msg = USER_PROMPT_TEMPLATE.format(
            source_filename=source_filename,
            ocr_method=ocr_method,
            text=text[:MAX_TEXT_CHARS],
        )

        try:
            model = self._get_model()
            response = model.generate_content(user_msg)
            raw = response.text
        except AIError:
            raise
        except Exception as exc:
            exc_str = str(exc).lower()
            retryable = any(k in exc_str for k in ("quota", "rate", "503", "429", "unavailable"))
            raise AIError(
                f"Gemini API error: {exc}",
                retryable=retryable,
            ) from exc

        logger.debug("Gemini raw response: %s", raw[:500])
        return _parse_response(raw)

    def health_check(self) -> bool:
        try:
            model = self._get_model()
            model.generate_content("ping")
            return True
        except Exception as exc:
            logger.warning("Gemini health check failed: %s", exc)
            return False
