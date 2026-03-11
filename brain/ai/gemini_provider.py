"""Google Gemini implementation of AIProvider (google-genai SDK)."""

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
        self._client = None   # Lazy init so import errors surface clearly

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai  # noqa: PLC0415
            except ImportError as exc:
                raise AIError(
                    "google-genai is not installed. "
                    "Run: pip install google-genai"
                ) from exc

            self._client = genai.Client(api_key=self._config.api_key)
        return self._client

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, text: str, source_filename: str,
                ocr_method: str = "unknown") -> FileAnalysis:
        """Call the Gemini API and parse the JSON response."""
        from google.genai import types  # noqa: PLC0415

        user_msg = USER_PROMPT_TEMPLATE.format(
            source_filename=source_filename,
            ocr_method=ocr_method,
            text=text[:MAX_TEXT_CHARS],
        )

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_tokens,
            response_mime_type="application/json",
        )

        try:
            client = self._get_client()
            response = client.models.generate_content(
                model=self._config.model,
                contents=user_msg,
                config=config,
            )
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
            from google.genai import types  # noqa: PLC0415

            client = self._get_client()
            client.models.generate_content(
                model=self._config.model,
                contents="ping",
                config=types.GenerateContentConfig(max_output_tokens=16),
            )
            return True
        except Exception as exc:
            logger.warning("Gemini health check failed: %s", exc)
            return False
