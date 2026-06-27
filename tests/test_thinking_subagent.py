"""Tests for the opt-in extended-thinking subagent (thinking_enabled in config).

When the flag is set, `_build_chat_model(..., thinking=True)` enables Anthropic's
extended-thinking parameter (honored by DeepSeek's Anthropic-compatible endpoint,
which ignores `budget_tokens`), and `_create_agent` registers `deep-thinker`
(effort `high`) and `deep-thinker-max` (effort `max`) subagents backed by that
model. The main agent's own model stays thinking-off and picks the effort per
delegation by choosing the subagent.
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


def test_build_chat_model_disables_thinking_on_deepseek_main_path(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model_name: str, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "stub-model"

    monkeypatch.setattr(app_mod, "init_chat_model", fake_init_chat_model)

    app_mod._build_chat_model("anthropic:deepseek-v4-pro")
    # DeepSeek V4 defaults thinking ON, so the non-thinking (main-agent) path
    # must explicitly disable it rather than omit the param.
    assert captured["thinking"] == {"type": "disabled"}
    assert "temperature" not in captured


def test_build_chat_model_omits_thinking_on_non_deepseek_path(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model_name: str, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "stub-model"

    monkeypatch.setattr(app_mod, "init_chat_model", fake_init_chat_model)

    # Other Anthropic-compatible providers already default thinking off when the
    # param is omitted; the disable is DeepSeek-scoped, so they get no param.
    app_mod._build_chat_model("anthropic:MiniMax-M2.5")
    assert "thinking" not in captured


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
    # No reasoning_effort passed -> no output_config in the request body.
    assert "model_kwargs" not in captured


def test_build_chat_model_injects_reasoning_effort(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model_name: str, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "stub-model"

    monkeypatch.setattr(app_mod, "init_chat_model", fake_init_chat_model)

    app_mod._build_chat_model(
        "anthropic:deepseek-v4-pro", thinking=True, reasoning_effort="max"
    )
    # DeepSeek V4 controls reasoning depth via output_config.effort, forwarded
    # by langchain-anthropic as an unknown model_kwarg.
    assert captured["model_kwargs"] == {"output_config": {"effort": "max"}}


def test_thinking_flag_defaults_off_when_missing(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({"model": "deepseek-v4-pro"}), encoding="utf-8")
    layout = RepoLayout(home=tmp_path, state_dir_name="state")
    assert load_config(layout).thinking_enabled is False


def test_thinking_flag_enabled_via_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump({"model": "deepseek-v4-pro", "thinking_enabled": True}),
        encoding="utf-8",
    )
    layout = RepoLayout(home=tmp_path, state_dir_name="state")
    assert load_config(layout).thinking_enabled is True


def test_thinking_flag_disabled_via_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump({"model": "deepseek-v4-pro", "thinking_enabled": False}),
        encoding="utf-8",
    )
    layout = RepoLayout(home=tmp_path, state_dir_name="state")
    assert load_config(layout).thinking_enabled is False


def test_deep_thinker_absent_when_flag_off() -> None:
    app = _fake_app(thinking_enabled=False)
    specs = app_mod.OpenStrixApp._build_deep_thinker_subagents(
        app,
        "anthropic:deepseek-v4-pro",
        max_tokens=32768,
        request_timeout_seconds=600,
    )
    assert specs == []


def test_deep_thinker_registers_high_and_max_variants(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_build_chat_model(model_name: str, **kwargs: Any) -> str:
        calls.append({"model_name": model_name, **kwargs})
        return f"thinking-model-{kwargs.get('reasoning_effort')}"

    monkeypatch.setattr(app_mod, "_build_chat_model", fake_build_chat_model)

    app = _fake_app(thinking_enabled=True)
    specs = app_mod.OpenStrixApp._build_deep_thinker_subagents(
        app,
        "anthropic:deepseek-v4-pro",
        max_tokens=32768,
        request_timeout_seconds=600,
    )

    by_name = {s["name"]: s for s in specs}
    # The main agent selects effort per delegation via subagent_type.
    assert set(by_name) == {"deep-thinker", "deep-thinker-max"}
    # Each variant carries a constructed model instance built with thinking on
    # and the matching DeepSeek effort level.
    assert by_name["deep-thinker"]["model"] == "thinking-model-high"
    assert by_name["deep-thinker-max"]["model"] == "thinking-model-max"
    efforts = {c["reasoning_effort"] for c in calls}
    assert efforts == {"high", "max"}
    assert all(c["thinking"] is True for c in calls)
    assert all(c["model_name"] == "anthropic:deepseek-v4-pro" for c in calls)
    # Descriptions must steer the main agent: coding mention on both, and the
    # max variant must flag its higher cost.
    assert "coding" in by_name["deep-thinker"]["description"].lower()
    assert "max" in by_name["deep-thinker-max"]["description"].lower()
