"""Anthropic Claude implementation of AIProvider."""

from __future__ import annotations

import json
import logging

import anthropic

from brain.ai.base import (
    AIProvider,
    AIError,
    FileAnalysis,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    VALID_CATEGORIES,
)
from brain.config import AIConfig

logger = logging.getLogger(__name__)

# Guard against sending enormous documents to the API
MAX_TEXT_CHARS = 8_000


class ClaudeProvider(AIProvider):
    def __init__(self, config: AIConfig):
        self._config = config
        self._client = anthropic.Anthropic(api_key=config.api_key)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, text: str, source_filename: str,
                ocr_method: str = "unknown") -> FileAnalysis:
        """Call the Claude Messages API and parse the JSON response."""
        user_msg = USER_PROMPT_TEMPLATE.format(
            source_filename=source_filename,
            ocr_method=ocr_method,
            text=text[:MAX_TEXT_CHARS],
        )

        try:
            response = self._client.messages.create(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
        except anthropic.RateLimitError as exc:
            raise AIError(f"Claude rate limit exceeded: {exc}", retryable=True) from exc
        except anthropic.APIStatusError as exc:
            retryable = exc.status_code in (500, 502, 503, 529)
            raise AIError(
                f"Claude API error {exc.status_code}: {exc.message}",
                retryable=retryable,
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise AIError(f"Claude connection error: {exc}", retryable=True) from exc
        except anthropic.AuthenticationError as exc:
            raise AIError(f"Claude authentication failed — check your API key: {exc}") from exc

        raw = response.content[0].text if response.content else ""
        logger.debug("Claude raw response: %s", raw[:500])
        return _parse_response(raw)

    def health_check(self) -> bool:
        try:
            self._client.messages.create(
                model=self._config.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception as exc:
            logger.warning("Claude health check failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Response parsing (shared logic; Gemini provider calls the same function)
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> FileAnalysis:
    """Parse the raw AI text response into a FileAnalysis.

    Strips accidental markdown fences, parses JSON, validates all fields.
    """
    cleaned = raw.strip()

    # Strip markdown code fences if the model misbehaved
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove first and last fence lines
        cleaned = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise AIError(
            f"AI returned invalid JSON: {exc}",
            retryable=False,
            raw=raw,
        ) from exc

    if not isinstance(data, dict):
        raise AIError("AI response is not a JSON object.", retryable=False, raw=raw)

    # Validate required fields
    for field_name in ("title", "category", "summary", "tags"):
        if field_name not in data:
            raise AIError(
                f"AI response missing required field: {field_name!r}",
                retryable=False,
                raw=raw,
            )

    title = str(data["title"]).strip() or "Untitled Document"
    summary = str(data["summary"]).strip()

    # Normalise category — fall back to "Other" rather than crashing
    category = str(data["category"]).strip()
    if category not in VALID_CATEGORIES:
        logger.warning(
            "AI returned unknown category %r — defaulting to 'Other'", category
        )
        category = "Other"

    # Normalise tags
    raw_tags = data["tags"]
    if not isinstance(raw_tags, list):
        raw_tags = [str(raw_tags)]
    tags = [str(t).lower().strip() for t in raw_tags if t][:8]

    return FileAnalysis(title=title, category=category, summary=summary, tags=tags)
