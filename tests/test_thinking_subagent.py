"""Tests for the opt-in extended-thinking subagent (KYNETIC_THINKING=1).

When the flag is set, `_build_chat_model(..., thinking=True)` enables Anthropic's
extended-thinking parameter (honored by DeepSeek's Anthropic-compatible endpoint,
which ignores `budget_tokens`), and `_create_agent` registers a `deep-thinker`
subagent backed by that model. The main agent's own model stays thinking-off.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

import kynetic_agents.app as app_mod
from kynetic_agents.config import RepoLayout, load_config


def _fake_app(thinking_enabled: bool) -> SimpleNamespace:
    """Minimal stand-in exposing the config fields the builder reads."""
    config = SimpleNamespace(
        thinking_enabled=thinking_enabled,
        model_max_retries=6,
        model_max_output_tokens=32768,
        model_request_timeout_seconds=600,
    )
    return SimpleNamespace(config=config)


def test_build_chat_model_omits_thinking_by_default(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model_name: str, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "stub-model"

    monkeypatch.setattr(app_mod, "init_chat_model", fake_init_chat_model)

    app_mod._build_chat_model("anthropic:deepseek-v4-pro")
    assert "thinking" not in captured
    assert "temperature" not in captured


def test_build_chat_model_enables_thinking(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model_name: str, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "stub-model"

    monkeypatch.setattr(app_mod, "init_chat_model", fake_init_chat_model)

    app_mod._build_chat_model("anthropic:deepseek-v4-pro", thinking=True)
    assert captured["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    # Extended thinking requires temperature=1; budget kept < max_tokens.
    assert captured["temperature"] == 1
    assert captured["thinking"]["budget_tokens"] < captured["max_tokens"]


def test_thinking_flag_defaults_off(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("KYNETIC_THINKING", raising=False)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({"model": "deepseek-v4-pro"}), encoding="utf-8")
    layout = RepoLayout(home=tmp_path, state_dir_name="state")
    assert load_config(layout).thinking_enabled is False


def test_thinking_flag_enabled_by_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KYNETIC_THINKING", "1")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({"model": "deepseek-v4-pro"}), encoding="utf-8")
    layout = RepoLayout(home=tmp_path, state_dir_name="state")
    assert load_config(layout).thinking_enabled is True


def test_thinking_flag_requires_exact_one(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({"model": "deepseek-v4-pro"}), encoding="utf-8")
    layout = RepoLayout(home=tmp_path, state_dir_name="state")
    for value in ("0", "true", "yes", ""):
        monkeypatch.setenv("KYNETIC_THINKING", value)
        assert load_config(layout).thinking_enabled is False


def test_deep_thinker_absent_when_flag_off() -> None:
    app = _fake_app(thinking_enabled=False)
    spec = app_mod.OpenStrixApp._build_deep_thinker_subagent(
        app,
        "anthropic:deepseek-v4-pro",
        max_tokens=32768,
        request_timeout_seconds=600,
    )
    assert spec is None


def test_deep_thinker_registered_with_thinking_model(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def fake_build_chat_model(model_name: str, **kwargs: Any) -> str:
        calls["model_name"] = model_name
        calls.update(kwargs)
        return "thinking-model-instance"

    monkeypatch.setattr(app_mod, "_build_chat_model", fake_build_chat_model)

    app = _fake_app(thinking_enabled=True)
    spec = app_mod.OpenStrixApp._build_deep_thinker_subagent(
        app,
        "anthropic:deepseek-v4-pro",
        max_tokens=32768,
        request_timeout_seconds=600,
    )

    assert spec is not None
    assert spec["name"] == "deep-thinker"
    # Model is a constructed instance (not a provider:model string) so it can
    # carry the thinking param; deepagents uses the instance directly.
    assert spec["model"] == "thinking-model-instance"
    assert calls["thinking"] is True
    assert calls["model_name"] == "anthropic:deepseek-v4-pro"
    # Description must steer the main agent toward hard problems, including
    # challenging coding, while leaving simple work inline.
    assert "coding" in spec["description"].lower()
