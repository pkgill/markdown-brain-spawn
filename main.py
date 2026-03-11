#!/usr/bin/env python3
"""markdown-brain-spawn — intelligent Obsidian vault ingestion pipeline.

Usage:
    python main.py                           # uses config.yaml
    python main.py --config my_config.yaml
    python main.py --dry-run                 # simulate without moving files
    python main.py --check                   # validate config and API key, then exit
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="markdown-brain-spawn",
        description="Intelligent file ingestion pipeline for your Obsidian vault.",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate processing without moving any files.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Validate config and AI API key, then exit.",
    )
    return parser.parse_args()


def _setup_logging(level: str, log_file: Path | None,
                   max_log_mb: int, backup_count: int) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Always log to stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Optionally also log to a rotating file
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_log_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)


def main() -> int:
    args = _parse_args()

    # --- Load config ---
    from brain.config import load_config, ConfigError
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"[ERROR] Config error:\n{exc}", file=sys.stderr)
        return 1

    # --- Setup logging ---
    _setup_logging(
        level=config.logging.level,
        log_file=config.logging.log_file,
        max_log_mb=config.logging.max_log_mb,
        backup_count=config.logging.backup_count,
    )

    logger = logging.getLogger(__name__)
    logger.info("markdown-brain-spawn starting up.")

    if args.dry_run:
        logger.info("DRY RUN mode — no files will be moved.")

    # --- Build AI provider ---
    from brain.ai import get_provider
    try:
        ai = get_provider(config)
    except Exception as exc:
        logger.error("Failed to initialise AI provider: %s", exc)
        return 1

    # --- Optional health check / validate mode ---
    if args.check:
        logger.info("Checking AI provider connectivity...")
        ok = ai.health_check()
        if ok:
            logger.info("AI provider is reachable. Config looks good!")
            return 0
        else:
            logger.error("AI provider health check failed. Check your API key and model name.")
            return 1

    # --- Validate key paths ---
    inbox = config.watch.inbox_path
    vault = config.watch.vault_path

    if not inbox.exists():
        logger.error("Inbox path does not exist: %s", inbox)
        return 1

    if not vault.exists():
        logger.warning(
            "Vault path does not exist yet: %s — it will be created as needed.", vault
        )

    # --- Build pipeline and scheduler ---
    from brain.pipeline import Pipeline
    from brain.scheduler import Scheduler

    pipeline = Pipeline(config, ai, dry_run=args.dry_run)
    scheduler = Scheduler(config, pipeline)

    # --- Run (blocks until Ctrl-C) ---
    scheduler.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
