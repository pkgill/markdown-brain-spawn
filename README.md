# markdown-brain-spawn

An intelligent file ingestion pipeline for your Obsidian vault.

Drop PDFs, scans, and images into your Google Drive — the pipeline OCRs them,
asks an AI to categorize and summarize the content, and neatly files everything
into your second brain with Obsidian-ready markdown frontmatter.

## What it does

1. **Watches** your Google Drive root folder for new files (PDFs, images)
2. **OCRs** them via Tesseract (with pdfplumber fast-path for digital PDFs)
3. **Analyzes** the extracted text with Claude or Gemini — getting back a title,
   category, summary, and tags
4. **Organizes** everything into your vault:
   - Creates `{vault}/{Category}/` subfolders as needed
   - Writes an Obsidian markdown note with YAML frontmatter
   - Moves the original file alongside the note
   - Links the note back to the source file via Obsidian wiki-links

## Quick start

### 1. Install system dependencies

**Tesseract** (required for scanned PDFs and images):

```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# Windows: download installer from https://github.com/UB-Mannheim/tesseract/wiki
# Then set ocr.tesseract_cmd in config.yaml to the .exe path
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure

```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your paths and API key
```

Key settings in `config.yaml`:

```yaml
watch:
  inbox_path: "/Users/you/Google Drive"          # watch this folder
  vault_path: "/Users/you/Google Drive/my_vault" # your Obsidian vault

ai:
  provider: "claude"          # or "gemini"
  api_key: "$ANTHROPIC_API_KEY"  # or paste key directly (don't commit!)
  model: "claude-haiku-4-5"

scheduler:
  mode: "realtime"   # or "batch"
```

**Windows path tip:** Google Drive Desktop mounts at something like
`C:/Users/you/Google Drive/My Drive` — check File Explorer.

### 4. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # Claude
# or
export GOOGLE_API_KEY=AIza...            # Gemini (set api_key: "$GOOGLE_API_KEY")
```

### 5. Validate your setup

```bash
python main.py --check
```

### 6. Run

```bash
python main.py                  # realtime mode (Ctrl-C to stop)
python main.py --dry-run        # simulate without moving files
```

## Generated markdown format

Each processed file produces a note like this:

```markdown
---
title: "W-2 Wage Statement 2024"
date_created: "2024-03-10"
source_file: "[[Tax/W2_scan_2024.pdf]]"
category: "Tax"
summary: "This is a 2024 W-2 issued by Acme Corp..."
tags:
  - w2
  - tax-2024
  - acme-corp
ocr_method: "pdfplumber"
pipeline_version: "1.0.0"
---

# W-2 Wage Statement 2024

> **Original file:** [[Tax/W2_scan_2024.pdf]]

## Summary
...

## Extracted Text
...
```

## Supported categories

The AI will place documents into one of these vault subfolders:

`Finance` · `Medical` · `Legal` · `Personal` · `Work` · `Education` ·
`Receipts` · `Insurance` · `Tax` · `Correspondence` · `Other`

## Processing modes

| Mode | Behavior |
|---|---|
| `realtime` | Processes each file immediately when it appears |
| `batch` | Accumulates files, processes all at once at a scheduled time |

Batch schedule options in `config.yaml`:
```yaml
scheduler:
  mode: "batch"
  batch_schedule: "daily"    # or "weekly"
  batch_time: "02:00"        # HH:MM local time
  batch_day: "sunday"        # for weekly only
```

## Failed files

Files that can't be processed (corrupt, unreadable, API failure) are moved to
`{vault}/_failed/` with a sidecar `.error.txt` explaining what went wrong.

## Running tests

```bash
# Fast (no Tesseract required)
pytest tests/ -m "not tesseract" -v

# All tests (requires Tesseract installed)
pytest tests/ -v

# With coverage
pytest tests/ --cov=brain --cov-report=term-missing
```

## Project structure

```
brain/
├── config.py          # Config loader and dataclasses
├── ocr.py             # OCR: pdfplumber + Tesseract fallback
├── organizer.py       # Markdown generation + file moves
├── pipeline.py        # Orchestrates OCR → AI → Organizer
├── scheduler.py       # Watchdog observer + batch scheduling
└── ai/
    ├── base.py        # Abstract AIProvider + FileAnalysis
    ├── claude_provider.py
    └── gemini_provider.py
main.py                # CLI entry point
config.example.yaml    # Reference config with all options
```

## Extending

**Add a new AI provider:** Create `brain/ai/your_provider.py` implementing
`AIProvider.analyze()` and `AIProvider.health_check()`, then add it to the
factory in `brain/ai/__init__.py`.

**Add new file types:** Add the extension to `watch.extensions` in `config.yaml`
and ensure the OCR module handles it (images work automatically via Tesseract).
