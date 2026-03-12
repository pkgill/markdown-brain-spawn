"""File organizer: generates Obsidian markdown and moves files into the vault."""

from __future__ import annotations

import logging
import re
import shutil
from datetime import date, datetime
from pathlib import Path

import yaml

from brain import __version__
from brain.ai.base import FileAnalysis
from brain.config import AppConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def organize_file(
    source_path: Path,
    analysis: FileAnalysis,
    ocr_text: str,
    config: AppConfig,
    ocr_method: str = "unknown",
    dry_run: bool = False,
) -> tuple[Path, Path]:
    """Full organize operation for a single file.

    1. Determine vault subfolder from analysis.category.
    2. Create the subfolder if needed.
    3. Generate the markdown string.
    4. Write the .md file.
    5. Move the original file.

    Returns:
        (md_path, moved_source_path) — both inside the vault.

    Raises:
        OrganizerError: If anything goes wrong during filesystem operations.
    """
    vault = config.watch.vault_path
    subfolder = _vault_subdir(vault, analysis.category, dry_run=dry_run)

    # Build destination paths
    # We want the source file and the markdown file to share the same base name (stem).
    # We find a unique base name that doesn't conflict with any existing files in the vault folder.
    base_name = safe_filename(analysis.title, suffix="")
    source_ext = source_path.suffix

    counter = 1
    while True:
        suffix_count = f"_{counter}" if counter > 1 else ""
        candidate_stem = base_name + suffix_count
        md_candidate = subfolder / (candidate_stem + ".md")
        source_candidate = subfolder / (candidate_stem + source_ext)

        # Neither the markdown file nor the source file should exist at the destination.
        if not md_candidate.exists() and not source_candidate.exists():
            md_path = md_candidate
            dest_source = source_candidate
            break
        counter += 1

    if dry_run:
        logger.info("[DRY RUN] Would write:  %s", md_path)
        logger.info("[DRY RUN] Would move:   %s → %s", source_path, dest_source)
        return md_path, dest_source

    # Vault-relative wiki-link (computed before the move, uses expected destination)
    vault_relative = dest_source.relative_to(vault).as_posix()
    wiki_link = f"[[{vault_relative}]]"

    # Generate and write the markdown
    md_content = build_markdown(
        analysis=analysis,
        wiki_link=wiki_link,
        ocr_text=ocr_text,
        ocr_method=ocr_method,
    )
    try:
        md_path.write_text(md_content, encoding="utf-8")
        logger.info("Wrote markdown: %s", md_path)
    except OSError as exc:
        raise OrganizerError(f"Failed to write markdown file {md_path}: {exc}") from exc

    # Move the original file
    try:
        shutil.move(str(source_path), str(dest_source))
        logger.info("Moved source: %s → %s", source_path, dest_source)
    except OSError as exc:
        raise OrganizerError(
            f"Failed to move {source_path} to {dest_source}: {exc}"
        ) from exc

    return md_path, dest_source


def build_markdown(
    analysis: FileAnalysis,
    wiki_link: str,
    ocr_text: str,
    ocr_method: str = "unknown",
    created_date: date | None = None,
) -> str:
    """Build the full markdown string (frontmatter + body) for an Obsidian note.

    Does NOT touch the filesystem.
    """
    today = (created_date or date.today()).isoformat()

    frontmatter = {
        "title": analysis.title,
        "date_created": today,
        "source_file": wiki_link,
        "category": analysis.category,
        "summary": analysis.summary,
        "tags": analysis.tags,
        "ocr_method": ocr_method,
        "pipeline_version": __version__,
    }

    # yaml.dump produces valid YAML; default_flow_style=False gives block style for tags
    fm_str = yaml.dump(
        frontmatter,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    ).rstrip()

    body = f"""\
---
{fm_str}
---

# {analysis.title}

> **Original file:** {wiki_link}

## Summary

{analysis.summary}

## Extracted Text

```
{ocr_text}
```

---
*Processed by markdown-brain-spawn on {today}*
"""
    return body


def safe_filename(title: str, suffix: str = ".md") -> str:
    """Convert a title to a filesystem-safe filename.

    Example: "Tax Return (2024)!" → "tax-return-2024.md"
    """
    name = title.lower()
    # Replace anything that isn't alphanumeric or hyphen with a hyphen
    name = re.sub(r"[^a-z0-9]+", "-", name)
    # Collapse multiple hyphens and strip leading/trailing
    name = re.sub(r"-{2,}", "-", name).strip("-")
    # Truncate at 60 chars (before suffix)
    name = name[:60].rstrip("-")
    return name + suffix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vault_subdir(vault: Path, category: str, dry_run: bool = False) -> Path:
    """Return the subfolder path, creating it if necessary."""
    subdir = vault / category
    if not dry_run:
        try:
            subdir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OrganizerError(
                f"Cannot create vault subfolder {subdir}: {exc}"
            ) from exc
    return subdir


def _unique_path(path: Path) -> Path:
    """If path already exists, append _2, _3, etc. to make it unique."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_to_failed(source_path: Path, vault_path: Path, reason: str) -> None:
    """Move a file that failed processing to the vault/_failed/ directory.

    Also writes a sidecar .error.txt with the failure reason.
    Silently logs if even this fails (avoids masking the original error).
    """
    _move_with_error(source_path, vault_path / "_failed", reason)


def move_to_unsupported(source_path: Path, vault_path: Path) -> None:
    """Move a file that is not supported by the pipeline to the vault/Unsupported files/ directory."""
    _move_with_error(source_path, vault_path / "Unsupported files", "Unsupported file format.")


def _move_with_error(source_path: Path, dest_dir: Path, reason: str) -> None:
    """Helper to move a file to a destination directory and write an error sidecar."""
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = _unique_path(dest_dir / source_path.name)
        shutil.move(str(source_path), str(dest))

        error_file = dest.with_suffix(dest.suffix + ".error.txt")
        error_file.write_text(
            f"Reason: {reason}\nTimestamp: {datetime.now().isoformat()}\n",
            encoding="utf-8",
        )
        logger.info("Moved file to: %s", dest)
    except Exception as exc:
        logger.error("Could not move file %s to %s: %s", source_path, dest_dir, exc)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class OrganizerError(Exception):
    """Raised when the file organizer cannot complete its work."""
