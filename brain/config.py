"""Config loader: reads YAML, validates, and returns typed dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WatchConfig:
    inbox_path: Path
    vault_path: Path
    extensions: list[str]  # normalised to lowercase with leading dot


@dataclass
class AIConfig:
    provider: Literal["claude", "gemini"]
    api_key: str
    model: str
    max_tokens: int
    temperature: float


@dataclass
class SchedulerConfig:
    mode: Literal["realtime", "batch"]
    batch_schedule: Literal["daily", "weekly"]
    batch_time: str   # "HH:MM"
    batch_day: str    # e.g. "sunday"


@dataclass
class OCRConfig:
    tesseract_cmd: str | None
    min_embedded_text_chars: int
    pdf_render_dpi: int
    language: str


@dataclass
class LoggingConfig:
    level: str
    log_file: Path | None
    max_log_mb: int
    backup_count: int


@dataclass
class AppConfig:
    watch: WatchConfig
    ai: AIConfig
    scheduler: SchedulerConfig
    ocr: OCRConfig
    logging: LoggingConfig


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load and validate a YAML config file, returning a typed AppConfig."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(
            f"Config file not found: {path}\n"
            "Copy config.example.yaml to config.yaml and fill in your values."
        )

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ConfigError("Config file must be a YAML mapping at the top level.")

    return _parse(raw)


# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------

_VALID_PROVIDERS = {"claude", "gemini"}
_VALID_MODES = {"realtime", "batch"}
_VALID_BATCH_SCHEDULES = {"daily", "weekly"}
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_VALID_BATCH_DAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
}


def _parse(raw: dict) -> AppConfig:
    return AppConfig(
        watch=_parse_watch(raw.get("watch", {})),
        ai=_parse_ai(raw.get("ai", {})),
        scheduler=_parse_scheduler(raw.get("scheduler", {})),
        ocr=_parse_ocr(raw.get("ocr", {})),
        logging=_parse_logging(raw.get("logging", {})),
    )


def _parse_watch(w: dict) -> WatchConfig:
    inbox_raw = _require_str(w, "inbox_path", section="watch")
    vault_raw = _require_str(w, "vault_path", section="watch")

    exts_raw = w.get("extensions", [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"])
    if not isinstance(exts_raw, list):
        raise ConfigError("watch.extensions must be a list of file extension strings.")

    extensions = []
    for e in exts_raw:
        e = str(e).lower()
        if not e.startswith("."):
            e = "." + e
        extensions.append(e)

    return WatchConfig(
        inbox_path=_resolve_path(inbox_raw),
        vault_path=_resolve_path(vault_raw),
        extensions=extensions,
    )


def _parse_ai(a: dict) -> AIConfig:
    provider = _require_str(a, "provider", section="ai").lower()
    if provider not in _VALID_PROVIDERS:
        raise ConfigError(
            f"ai.provider must be one of {sorted(_VALID_PROVIDERS)}, got: {provider!r}"
        )

    api_key_raw = _require_str(a, "api_key", section="ai")
    api_key = _resolve_api_key(api_key_raw)

    model = _require_str(a, "model", section="ai")
    max_tokens = int(a.get("max_tokens", 512))
    temperature = float(a.get("temperature", 0.2))

    return AIConfig(
        provider=provider,  # type: ignore[arg-type]
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _parse_scheduler(s: dict) -> SchedulerConfig:
    mode = s.get("mode", "realtime").lower()
    if mode not in _VALID_MODES:
        raise ConfigError(
            f"scheduler.mode must be one of {sorted(_VALID_MODES)}, got: {mode!r}"
        )

    batch_schedule = s.get("batch_schedule", "daily").lower()
    if batch_schedule not in _VALID_BATCH_SCHEDULES:
        raise ConfigError(
            f"scheduler.batch_schedule must be one of "
            f"{sorted(_VALID_BATCH_SCHEDULES)}, got: {batch_schedule!r}"
        )

    batch_time = str(s.get("batch_time", "02:00"))
    _validate_time_format(batch_time)

    batch_day = str(s.get("batch_day", "sunday")).lower()
    if batch_day not in _VALID_BATCH_DAYS:
        raise ConfigError(
            f"scheduler.batch_day must be a day of the week, got: {batch_day!r}"
        )

    return SchedulerConfig(
        mode=mode,  # type: ignore[arg-type]
        batch_schedule=batch_schedule,  # type: ignore[arg-type]
        batch_time=batch_time,
        batch_day=batch_day,
    )


def _parse_ocr(o: dict) -> OCRConfig:
    tesseract_cmd = o.get("tesseract_cmd", None)
    if tesseract_cmd is not None:
        tesseract_cmd = str(tesseract_cmd)

    return OCRConfig(
        tesseract_cmd=tesseract_cmd,
        min_embedded_text_chars=int(o.get("min_embedded_text_chars", 100)),
        pdf_render_dpi=int(o.get("pdf_render_dpi", 300)),
        language=str(o.get("language", "eng")),
    )


def _parse_logging(l: dict) -> LoggingConfig:
    level = str(l.get("level", "INFO")).upper()
    if level not in _VALID_LOG_LEVELS:
        raise ConfigError(
            f"logging.level must be one of {sorted(_VALID_LOG_LEVELS)}, got: {level!r}"
        )

    log_file_raw = l.get("log_file", None)
    log_file = Path(log_file_raw) if log_file_raw else None

    return LoggingConfig(
        level=level,
        log_file=log_file,
        max_log_mb=int(l.get("max_log_mb", 10)),
        backup_count=int(l.get("backup_count", 5)),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_str(mapping: dict, key: str, section: str) -> str:
    value = mapping.get(key)
    if not value:
        raise ConfigError(f"Missing required config key: {section}.{key}")
    return str(value)


def _resolve_path(raw: str) -> Path:
    """Accept forward-slash or backslash paths and return a Path.

    Does NOT require the path to exist at load time — the pipeline will
    emit clear errors when it tries to use a non-existent path.
    """
    # pathlib.Path accepts forward slashes on Windows natively.
    return Path(raw)


def _resolve_api_key(value: str) -> str:
    """If value starts with '$', read from the matching environment variable."""
    if value.startswith("$"):
        var_name = value[1:]
        key = os.environ.get(var_name)
        if not key:
            raise ConfigError(
                f"Environment variable {var_name!r} is not set or is empty.\n"
                f"Set it with: export {var_name}=your_api_key"
            )
        return key
    return value


def _validate_time_format(value: str) -> None:
    parts = value.split(":")
    if len(parts) != 2:
        raise ConfigError(f"scheduler.batch_time must be HH:MM format, got: {value!r}")
    try:
        hour, minute = int(parts[0]), int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
    except ValueError:
        raise ConfigError(
            f"scheduler.batch_time must be a valid HH:MM time, got: {value!r}"
        )


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised for any config validation failure. Always fatal."""
