"""AI provider factory."""

from brain.config import AppConfig
from brain.ai.base import AIProvider


def get_provider(config: AppConfig) -> AIProvider:
    """Return the correct AIProvider subclass based on config.ai.provider."""
    if config.ai.provider == "claude":
        from brain.ai.claude_provider import ClaudeProvider
        return ClaudeProvider(config.ai)
    if config.ai.provider == "gemini":
        from brain.ai.gemini_provider import GeminiProvider
        return GeminiProvider(config.ai)
    raise ValueError(f"Unknown AI provider: {config.ai.provider!r}")
