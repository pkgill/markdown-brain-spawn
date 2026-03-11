"""Tests for brain/organizer.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from brain.ai.base import FileAnalysis
from brain.organizer import (
    build_markdown,
    organize_file,
    safe_filename,
    move_to_failed,
    OrganizerError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_analysis():
    return FileAnalysis(
        title="Tax Return 2024",
        category="Tax",
        summary="Federal tax return for 2024 filed by Paul Gilmore.",
        tags=["tax-2024", "federal", "income"],
    )


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    f = tmp_path / "w2_scan.pdf"
    f.write_bytes(b"fake pdf content")
    return f


# ---------------------------------------------------------------------------
# safe_filename
# ---------------------------------------------------------------------------

def test_safe_filename_basic():
    assert safe_filename("Tax Return 2024") == "tax-return-2024.md"


def test_safe_filename_strips_special_chars():
    assert safe_filename("Tax Return (2024)!") == "tax-return-2024.md"


def test_safe_filename_truncates_at_60():
    long_title = "a" * 80
    result = safe_filename(long_title)
    assert len(result) <= 63  # 60 chars + ".md"


def test_safe_filename_no_leading_trailing_hyphens():
    result = safe_filename("---hello---")
    assert not result.startswith("-") and not result.endswith("-.md")


# ---------------------------------------------------------------------------
# build_markdown
# ---------------------------------------------------------------------------

def test_build_markdown_has_yaml_frontmatter(sample_analysis):
    md = build_markdown(sample_analysis, "[[Tax/w2.pdf]]", "raw text", "pdfplumber")
    assert md.startswith("---\n")
    # Find the frontmatter block
    end = md.index("---\n", 4)
    fm = yaml.safe_load(md[4:end])
    assert fm["title"] == "Tax Return 2024"
    assert fm["category"] == "Tax"
    assert "tax-2024" in fm["tags"]


def test_build_markdown_all_required_frontmatter_fields(sample_analysis):
    md = build_markdown(sample_analysis, "[[Tax/w2.pdf]]", "text", "pdfplumber")
    end = md.index("---\n", 4)
    fm = yaml.safe_load(md[4:end])
    for field in ("title", "date_created", "source_file", "category", "summary", "tags",
                  "ocr_method", "pipeline_version"):
        assert field in fm, f"Missing frontmatter field: {field}"


def test_build_markdown_wiki_link_present(sample_analysis):
    md = build_markdown(sample_analysis, "[[Tax/w2.pdf]]", "text", "pdfplumber")
    assert "[[Tax/w2.pdf]]" in md


def test_build_markdown_body_contains_ocr_text(sample_analysis):
    md = build_markdown(sample_analysis, "[[x.pdf]]", "important ocr text", "pdfplumber")
    assert "important ocr text" in md


def test_build_markdown_body_contains_summary(sample_analysis):
    md = build_markdown(sample_analysis, "[[x.pdf]]", "text", "pdfplumber")
    assert sample_analysis.summary in md


# ---------------------------------------------------------------------------
# organize_file
# ---------------------------------------------------------------------------

def test_organize_file_creates_category_dir(sample_analysis, source_file, app_config):
    organize_file(source_file, sample_analysis, "ocr text", app_config)
    expected_dir = app_config.watch.vault_path / "Tax"
    assert expected_dir.is_dir()


def test_organize_file_writes_markdown(sample_analysis, source_file, app_config):
    md_path, _ = organize_file(source_file, sample_analysis, "ocr text", app_config)
    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "Tax Return 2024" in content


def test_organize_file_moves_original(sample_analysis, source_file, app_config):
    _, moved = organize_file(source_file, sample_analysis, "ocr text", app_config)
    assert not source_file.exists()   # moved away
    assert moved.exists()             # now in vault


def test_organize_file_moved_inside_vault(sample_analysis, source_file, app_config):
    _, moved = organize_file(source_file, sample_analysis, "ocr text", app_config)
    assert str(moved).startswith(str(app_config.watch.vault_path))


def test_organize_file_dry_run_does_nothing(sample_analysis, source_file, app_config):
    organize_file(source_file, sample_analysis, "ocr text", app_config, dry_run=True)
    assert source_file.exists()   # not moved
    category_dir = app_config.watch.vault_path / "Tax"
    assert not category_dir.exists()  # not created


def test_organize_file_unique_filename_on_collision(sample_analysis, source_file, app_config):
    # Process the same logical file twice to trigger collision avoidance
    src2 = source_file.parent / "w2_scan2.pdf"
    src2.write_bytes(b"another fake pdf")

    md1, _ = organize_file(source_file, sample_analysis, "text", app_config)

    # Change the source path name to avoid move collision, keep same title
    md2, _ = organize_file(src2, sample_analysis, "text", app_config)

    # Both markdown files should exist and have different names
    assert md1.exists()
    assert md2.exists()
    assert md1 != md2


# ---------------------------------------------------------------------------
# move_to_failed
# ---------------------------------------------------------------------------

def test_move_to_failed_creates_failed_dir(source_file, app_config):
    move_to_failed(source_file, app_config.watch.vault_path, "test error")
    failed_dir = app_config.watch.vault_path / "_failed"
    assert failed_dir.is_dir()


def test_move_to_failed_writes_error_sidecar(source_file, app_config):
    move_to_failed(source_file, app_config.watch.vault_path, "something broke")
    failed_dir = app_config.watch.vault_path / "_failed"
    error_files = list(failed_dir.glob("*.error.txt"))
    assert error_files
    assert "something broke" in error_files[0].read_text()
