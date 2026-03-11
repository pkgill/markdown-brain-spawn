"""Tests for brain/config.py."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from brain.config import load_config, ConfigError


def _write_config(tmp_path: Path, data: dict) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump(data), encoding="utf-8")
    return cfg


MINIMAL_VALID = {
    "watch": {
        "inbox_path": "/tmp/inbox",
        "vault_path": "/tmp/vault",
    },
    "ai": {
        "provider": "claude",
        "api_key": "sk-test",
        "model": "claude-haiku-4-5",
    },
}


def test_load_valid_config(tmp_path):
    cfg_path = _write_config(tmp_path, MINIMAL_VALID)
    config = load_config(cfg_path)
    assert config.ai.provider == "claude"
    assert config.ai.model == "claude-haiku-4-5"
    assert config.scheduler.mode == "realtime"   # default
    assert ".pdf" in config.watch.extensions      # default


def test_missing_inbox_path_raises(tmp_path):
    data = {**MINIMAL_VALID, "watch": {"vault_path": "/tmp/vault"}}
    cfg_path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="inbox_path"):
        load_config(cfg_path)


def test_missing_vault_path_raises(tmp_path):
    data = {**MINIMAL_VALID, "watch": {"inbox_path": "/tmp/inbox"}}
    cfg_path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="vault_path"):
        load_config(cfg_path)


def test_missing_api_key_raises(tmp_path):
    data = {**MINIMAL_VALID}
    data["ai"] = {**data["ai"]}
    del data["ai"]["api_key"]
    cfg_path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="api_key"):
        load_config(cfg_path)


def test_env_var_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_BRAIN_KEY", "real-key-123")
    data = {**MINIMAL_VALID, "ai": {**MINIMAL_VALID["ai"], "api_key": "$TEST_BRAIN_KEY"}}
    cfg_path = _write_config(tmp_path, data)
    config = load_config(cfg_path)
    assert config.ai.api_key == "real-key-123"


def test_env_var_not_set_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
    data = {**MINIMAL_VALID, "ai": {**MINIMAL_VALID["ai"], "api_key": "$NONEXISTENT_KEY"}}
    cfg_path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="NONEXISTENT_KEY"):
        load_config(cfg_path)


def test_invalid_provider_raises(tmp_path):
    data = {**MINIMAL_VALID, "ai": {**MINIMAL_VALID["ai"], "provider": "gpt4"}}
    cfg_path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="provider"):
        load_config(cfg_path)


def test_invalid_mode_raises(tmp_path):
    data = {**MINIMAL_VALID, "scheduler": {"mode": "turbo"}}
    cfg_path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="mode"):
        load_config(cfg_path)


def test_invalid_batch_time_raises(tmp_path):
    data = {**MINIMAL_VALID, "scheduler": {"mode": "batch", "batch_time": "25:99"}}
    cfg_path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="batch_time"):
        load_config(cfg_path)


def test_forward_slash_path_parses(tmp_path):
    data = {**MINIMAL_VALID, "watch": {
        "inbox_path": "/Users/pkgil/Google Drive",
        "vault_path": "/Users/pkgil/Google Drive/vault",
    }}
    cfg_path = _write_config(tmp_path, data)
    config = load_config(cfg_path)
    assert isinstance(config.watch.inbox_path, Path)


def test_extensions_normalised_lowercase(tmp_path):
    data = {**MINIMAL_VALID, "watch": {
        **MINIMAL_VALID["watch"],
        "extensions": [".PDF", "PNG", ".Jpg"],
    }}
    cfg_path = _write_config(tmp_path, data)
    config = load_config(cfg_path)
    assert set(config.watch.extensions) == {".pdf", ".png", ".jpg"}


def test_config_file_not_found_raises():
    with pytest.raises(ConfigError, match="not found"):
        load_config("/nonexistent/path/config.yaml")


def test_example_config_is_valid(monkeypatch):
    """The committed config.example.yaml should always parse without errors."""
    example = Path(__file__).parent.parent / "config.example.yaml"
    assert example.exists(), "config.example.yaml is missing from the repo"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-for-test")
    config = load_config(example)
    assert config.ai.provider == "claude"
