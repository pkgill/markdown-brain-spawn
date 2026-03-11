"""Tests for brain/ai/ — providers and factory."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from brain.ai.base import FileAnalysis, AIError, VALID_CATEGORIES
from brain.ai.claude_provider import ClaudeProvider, _parse_response
from brain.config import AIConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ai_config(provider="claude") -> AIConfig:
    return AIConfig(
        provider=provider,
        api_key="test-key",
        model="claude-haiku-4-5",
        max_tokens=512,
        temperature=0.2,
    )


VALID_JSON = json.dumps({
    "title": "W-2 Wage Statement 2024",
    "category": "Tax",
    "summary": "This is a W-2 form from Acme Corp for 2024.",
    "tags": ["w2", "tax-2024", "acme-corp"],
})


# ---------------------------------------------------------------------------
# _parse_response (shared logic)
# ---------------------------------------------------------------------------

def test_parse_valid_json():
    result = _parse_response(VALID_JSON)
    assert isinstance(result, FileAnalysis)
    assert result.title == "W-2 Wage Statement 2024"
    assert result.category == "Tax"
    assert "w2" in result.tags


def test_parse_strips_markdown_fences():
    fenced = f"```json\n{VALID_JSON}\n```"
    result = _parse_response(fenced)
    assert result.title == "W-2 Wage Statement 2024"


def test_parse_unknown_category_defaults_to_other():
    data = json.loads(VALID_JSON)
    data["category"] = "RandomInvalidCategory"
    result = _parse_response(json.dumps(data))
    assert result.category == "Other"


def test_parse_tags_clamped_to_8():
    data = json.loads(VALID_JSON)
    data["tags"] = [f"tag{i}" for i in range(20)]
    result = _parse_response(json.dumps(data))
    assert len(result.tags) == 8


def test_parse_invalid_json_raises():
    with pytest.raises(AIError, match="invalid JSON"):
        _parse_response("not json at all")


def test_parse_missing_required_field_raises():
    data = json.loads(VALID_JSON)
    del data["summary"]
    with pytest.raises(AIError, match="summary"):
        _parse_response(json.dumps(data))


def test_parse_non_object_raises():
    with pytest.raises(AIError, match="not a JSON object"):
        _parse_response("[1, 2, 3]")


# ---------------------------------------------------------------------------
# ClaudeProvider.analyze — mocked API
# ---------------------------------------------------------------------------

def _make_anthropic_response(text: str):
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def test_claude_analyze_success():
    provider = ClaudeProvider(_ai_config())
    provider._client = MagicMock()
    provider._client.messages.create.return_value = _make_anthropic_response(VALID_JSON)

    result = provider.analyze("some document text", "document.pdf")
    assert result.title == "W-2 Wage Statement 2024"


def test_claude_analyze_rate_limit_raises_retryable():
    import anthropic as ant
    provider = ClaudeProvider(_ai_config())
    provider._client = MagicMock()
    provider._client.messages.create.side_effect = ant.RateLimitError(
        "rate limit", response=MagicMock(), body={}
    )

    with pytest.raises(AIError) as exc_info:
        provider.analyze("text", "file.pdf")
    assert exc_info.value.retryable is True


def test_claude_analyze_bad_json_raises_non_retryable():
    provider = ClaudeProvider(_ai_config())
    provider._client = MagicMock()
    provider._client.messages.create.return_value = _make_anthropic_response("bad response")

    with pytest.raises(AIError) as exc_info:
        provider.analyze("text", "file.pdf")
    assert exc_info.value.retryable is False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_get_provider_returns_claude(app_config):
    from brain.ai import get_provider
    from brain.ai.claude_provider import ClaudeProvider
    provider = get_provider(app_config)
    assert isinstance(provider, ClaudeProvider)


def test_get_provider_unknown_raises(app_config):
    from brain.ai import get_provider
    app_config.ai.provider = "gpt4"
    with pytest.raises(ValueError, match="Unknown AI provider"):
        get_provider(app_config)
