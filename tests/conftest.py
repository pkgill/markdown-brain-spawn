"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.config import (
    AppConfig, WatchConfig, AIConfig, SchedulerConfig, OCRConfig, LoggingConfig
)


# ---------------------------------------------------------------------------
# Fixtures dir
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def sample_pdf(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample.pdf"


@pytest.fixture
def sample_image(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample.png"


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


@pytest.fixture
def tmp_inbox(tmp_path: Path) -> Path:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    return inbox


@pytest.fixture
def app_config(tmp_vault: Path, tmp_inbox: Path) -> AppConfig:
    """Minimal valid AppConfig using tmp directories."""
    return AppConfig(
        watch=WatchConfig(
            inbox_path=tmp_inbox,
            vault_path=tmp_vault,
            extensions=[".pdf", ".png", ".jpg"],
        ),
        ai=AIConfig(
            provider="claude",
            api_key="test-key",
            model="claude-haiku-4-5",
            max_tokens=512,
            temperature=0.2,
        ),
        scheduler=SchedulerConfig(
            mode="realtime",
            batch_schedule="daily",
            batch_time="02:00",
            batch_day="sunday",
        ),
        ocr=OCRConfig(
            tesseract_cmd=None,
            min_embedded_text_chars=100,
            pdf_render_dpi=300,
            language="eng",
        ),
        logging=LoggingConfig(
            level="DEBUG",
            log_file=None,
            max_log_mb=10,
            backup_count=5,
        ),
    )
