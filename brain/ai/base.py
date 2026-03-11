"""Abstract base class and shared types for AI providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------

VALID_CATEGORIES = {
    "Finance", "Medical", "Legal", "Personal", "Work",
    "Education", "Receipts", "Insurance", "Tax", "Correspondence", "Other",
}


@dataclass
class FileAnalysis:
    """Structured output returned by every AI provider."""
    title: str
    category: str        # One of VALID_CATEGORIES
    summary: str
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract provider interface
# ---------------------------------------------------------------------------

class AIProvider(ABC):
    """All AI providers must implement this interface."""

    @abstractmethod
    def analyze(self, text: str, source_filename: str) -> FileAnalysis:
        """Analyze OCR-extracted text and return structured metadata.

        Args:
            text: Raw text extracted from the document.
            source_filename: Original filename (used as hint in the prompt).

        Returns:
            FileAnalysis with title, category, summary, and tags.

        Raises:
            AIError: On API failure or unparseable response.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Verify that the API key is valid and the service is reachable.

        Returns:
            True if the provider is available, False otherwise.
        """


# ---------------------------------------------------------------------------
# Shared prompt (identical for all providers)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a document librarian assistant for a personal knowledge management system.
You will be given OCR-extracted text from a document and the original filename.
Your job is to analyze the content and return a JSON object.

REQUIRED JSON FIELDS:
- "title": A concise descriptive title (3-8 words, Title Case, hyphens allowed, no other special chars)
- "category": Exactly one folder name from this fixed list:
    Finance | Medical | Legal | Personal | Work | Education | Receipts | Insurance | Tax | Correspondence | Other
- "summary": 2-3 plain English sentences describing what the document is, who produced it, and its key details
- "tags": A JSON array of 3-8 lowercase kebab-case strings. Always include the year if visible (e.g. "2024").
          Include institution names, document types, and key subjects.

RULES:
- Respond ONLY with valid JSON. No markdown fences, no explanation, no trailing text.
- If you cannot determine a field, make a reasonable inference rather than leaving it empty.
- If the OCR text is illegible or nearly empty, use the source filename as the basis for title/category.
- The category value MUST be spelled exactly as shown in the list above."""

USER_PROMPT_TEMPLATE = """\
Source filename: {source_filename}
Extraction method: {ocr_method}

--- DOCUMENT TEXT START ---
{text}
--- DOCUMENT TEXT END ---

Analyze this document and return the JSON object now."""


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class AIError(Exception):
    """Raised by AI providers on API failure or bad response."""

    def __init__(self, message: str, retryable: bool = False, raw: str | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.raw = raw  # Raw API response if available, for logging
