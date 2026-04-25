"""Tests for the mempalace shared-memory integration.

Covers four areas of intent:
  1. Config — new fields parse correctly from YAML.
  2. MCP allowed_tools filter — servers only expose permitted tools to agents.
  3. Channel detection — _is_mempalace_channel() identifies configured channels
     by ID or by name.
  4. Message enqueuing — _remember_message() enqueues to the writer queue at
     the right times and stays silent when it should.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import kynetic_agents.app as app_mod
from kynetic_agents.config import AppConfig, RepoLayout, load_config
from kynetic_agents.mcp_client import MCPConnection, MCPServerConfig
from kynetic_agents.models import MempalaceWriteItem
from kynetic_agents.phone_book import PhoneBookEntry


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class DummyAgent:
    async def ainvoke(self, _: dict[str, Any]) -> dict[str, Any]:
        return {"messages": []}


def _stub_agent_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_mod, "create_deep_agent", lambda **_: DummyAgent())


def _make_app(
    tmp_path: Path,
    config_yaml: str,
    monkeypatch: pytest.MonkeyPatch,
) -> app_mod.OpenStrixApp:
    _stub_agent_factory(monkeypatch)
    (tmp_path / "config.yaml").write_text(config_yaml, encoding="utf-8")
    return app_mod.OpenStrixApp(tmp_path)


# ---------------------------------------------------------------------------
# 1. Config parsing
# ---------------------------------------------------------------------------

class TestMempalaceConfig:
    def test_mempalace_fields_absent_gives_safe_defaults(self, tmp_path: Path) -> None:
        """When no mempalace keys are present, the config is safe to run without mempalace."""
        (tmp_path / "config.yaml").write_text("model: test-model\n", encoding="utf-8")
        layout = RepoLayout(home=tmp_path, state_dir_name="state")
        config = load_config(layout)

        assert config.mempalace_path is None
        assert config.mempalace_writer is False
        assert config.mempalace_channels == []

    def test_mempalace_path_is_parsed(self, tmp_path: Path) -> None:
        (tmp_path / "config.yaml").write_text(
            "mempalace_path: /shared/palace\n", encoding="utf-8"
        )
        layout = RepoLayout(home=tmp_path, state_dir_name="state")
        config = load_config(layout)

        assert config.mempalace_path == "/shared/palace"

    def test_mempalace_writer_true_is_parsed(self, tmp_path: Path) -> None:
        (tmp_path / "config.yaml").write_text(
            "mempalace_writer: true\n", encoding="utf-8"
        )
        layout = RepoLayout(home=tmp_path, state_dir_name="state")
        config = load_config(layout)

        assert config.mempalace_writer is True

    def test_mempalace_channels_parsed_as_strings(self, tmp_path: Path) -> None:
        (tmp_path / "config.yaml").write_text(
            "mempalace_channels:\n  - '1234567890'\n  - general\n",
            encoding="utf-8",
        )
        layout = RepoLayout(home=tmp_path, state_dir_name="state")
        config = load_config(layout)

        assert "1234567890" in config.mempalace_channels
        assert "general" in config.mempalace_channels

    def test_mempalace_path_empty_string_becomes_none(self, tmp_path: Path) -> None:
        (tmp_path / "config.yaml").write_text(
            "mempalace_path: ''\n", encoding="utf-8"
        )
        layout = RepoLayout(home=tmp_path, state_dir_name="state")
        config = load_config(layout)

        assert config.mempalace_path is None

    def test_appconfig_defaults(self) -> None:
        config = AppConfig()

        assert config.mempalace_path is None
        assert config.mempalace_writer is False
        assert config.mempalace_channels == []


# ---------------------------------------------------------------------------
# 2. MCP allowed_tools filter
# ---------------------------------------------------------------------------

class TestMCPAllowedTools:
    def test_allowed_tools_none_exposes_all_tools(self) -> None:
        """No allowlist → every tool the server advertises reaches the agent."""
        cfg = MCPServerConfig(name="test", command="echo", args=[], allowed_tools=None)
        assert cfg.allowed_tools is None

    def test_allowed_tools_parsed_from_dict(self) -> None:
        cfg = MCPServerConfig.from_dict({
            "name": "mempalace",
            "command": "python",
            "args": ["-m", "mempalace.mcp_server"],
            "allowed_tools": ["mempalace_search", "mempalace_get_drawer"],
        })

        assert cfg.allowed_tools == ["mempalace_search", "mempalace_get_drawer"]

    def test_allowed_tools_absent_in_dict_gives_none(self) -> None:
        cfg = MCPServerConfig.from_dict({
            "name": "test",
            "command": "echo",
            "args": [],
        })

        assert cfg.allowed_tools is None

    @pytest.mark.asyncio
    async def test_discover_tools_filters_to_allowlist(self) -> None:
        """Only tools in allowed_tools are returned; others are silently dropped."""
        mock_session = AsyncMock()

        def _fake_tool(name: str) -> MagicMock:
            t = MagicMock()
            t.name = name
            t.description = f"Tool {name}"
            t.inputSchema = {"type": "object", "properties": {}}
            return t

        mock_result = MagicMock()
        mock_result.tools = [
            _fake_tool("mempalace_search"),
            _fake_tool("mempalace_add_drawer"),   # write tool — should be filtered
            _fake_tool("mempalace_get_drawer"),
        ]
        mock_session.list_tools = AsyncMock(return_value=mock_result)

        cfg = MCPServerConfig(
            name="mempalace",
            command="python",
            args=[],
            allowed_tools=["mempalace_search", "mempalace_get_drawer"],
        )
        conn = MCPConnection(config=cfg, session=mock_session)
        tools = await conn.discover_tools()

        tool_names = [t.name for t in tools]
        assert "mcp_mempalace_mempalace_search" in tool_names
        assert "mcp_mempalace_mempalace_get_drawer" in tool_names
        assert not any("add_drawer" in n for n in tool_names)

    @pytest.mark.asyncio
    async def test_discover_tools_without_allowlist_exposes_everything(self) -> None:
        """No allowlist → all tools returned, including write tools."""
        mock_session = AsyncMock()

        def _fake_tool(name: str) -> MagicMock:
            t = MagicMock()
            t.name = name
            t.description = f"Tool {name}"
            t.inputSchema = {"type": "object", "properties": {}}
            return t

        mock_result = MagicMock()
        mock_result.tools = [_fake_tool("mempalace_search"), _fake_tool("mempalace_add_drawer")]
        mock_session.list_tools = AsyncMock(return_value=mock_result)

        cfg = MCPServerConfig(name="mempalace", command="python", args=[], allowed_tools=None)
        conn = MCPConnection(config=cfg, session=mock_session)
        tools = await conn.discover_tools()

        assert len(tools) == 2


# ---------------------------------------------------------------------------
# 3. Channel detection
# ---------------------------------------------------------------------------

class TestMempalaceChannelDetection:
    def test_no_mempalace_channels_configured_never_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no mempalace_channels, no channel is ever considered a mempalace channel."""
        app = _make_app(tmp_path, "model: test-model\n", monkeypatch)

        assert app._is_mempalace_channel("1234567890") is False
        assert app._is_mempalace_channel("general") is False

    def test_channel_matched_by_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_app(
            tmp_path,
            "mempalace_channels:\n  - '1234567890'\n",
            monkeypatch,
        )

        assert app._is_mempalace_channel("1234567890") is True

    def test_channel_not_matched_when_id_differs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_app(
            tmp_path,
            "mempalace_channels:\n  - '1234567890'\n",
            monkeypatch,
        )

        assert app._is_mempalace_channel("9999999999") is False

    def test_channel_matched_by_name_via_phone_book(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the channel_id is in the phone book under a configured name, it matches."""
        app = _make_app(
            tmp_path,
            "mempalace_channels:\n  - general\n",
            monkeypatch,
        )
        app.phone_book.add(
            PhoneBookEntry(id="111222333", name="general", kind="channel")
        )

        assert app._is_mempalace_channel("111222333") is True

    def test_channel_not_matched_when_name_differs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_app(
            tmp_path,
            "mempalace_channels:\n  - general\n",
            monkeypatch,
        )
        app.phone_book.add(
            PhoneBookEntry(id="111222333", name="random-channel", kind="channel")
        )

        assert app._is_mempalace_channel("111222333") is False


# ---------------------------------------------------------------------------
# 4. Message enqueuing
# ---------------------------------------------------------------------------

class TestMempalaceMessageEnqueue:
    """_remember_message() should enqueue exactly when all conditions are met."""

    def _app_with_queue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[app_mod.OpenStrixApp, asyncio.Queue]:
        app = _make_app(
            tmp_path,
            "mempalace_channels:\n  - '555'\n",
            monkeypatch,
        )
        queue: asyncio.Queue[MempalaceWriteItem] = asyncio.Queue()
        app._mempalace_write_queue = queue
        return app, queue

    def test_message_in_mempalace_channel_is_enqueued(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, queue = self._app_with_queue(tmp_path, monkeypatch)

        app._remember_message(
            channel_id="555",
            author="alice",
            content="hello world",
            attachment_names=[],
        )

        assert queue.qsize() == 1
        item = queue.get_nowait()
        assert item.channel_id == "555"
        assert item.author == "alice"
        assert item.content == "hello world"

    def test_message_in_non_mempalace_channel_is_not_enqueued(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, queue = self._app_with_queue(tmp_path, monkeypatch)

        app._remember_message(
            channel_id="999",  # not in mempalace_channels
            author="alice",
            content="hello world",
            attachment_names=[],
        )

        assert queue.qsize() == 0

    def test_history_replay_does_not_enqueue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """persist=False (used during startup history replay) must never enqueue.
        This prevents re-ingesting the full chat history every time the bot restarts."""
        app, queue = self._app_with_queue(tmp_path, monkeypatch)

        app._remember_message(
            channel_id="555",
            author="alice",
            content="hello world",
            attachment_names=[],
            persist=False,
        )

        assert queue.qsize() == 0

    def test_empty_content_is_not_enqueued(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, queue = self._app_with_queue(tmp_path, monkeypatch)

        app._remember_message(
            channel_id="555",
            author="alice",
            content="   ",
            attachment_names=[],
        )

        assert queue.qsize() == 0

    def test_no_queue_means_no_enqueue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-writer processes have _mempalace_write_queue=None; nothing should blow up."""
        app = _make_app(
            tmp_path,
            "mempalace_channels:\n  - '555'\n",
            monkeypatch,
        )
        assert app._mempalace_write_queue is None

        # Should complete without error and not attempt to enqueue anything.
        app._remember_message(
            channel_id="555",
            author="alice",
            content="hello",
            attachment_names=[],
        )

    def test_duplicate_message_is_not_enqueued_twice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_remember_message deduplicates by message_id; the second call is a no-op."""
        app, queue = self._app_with_queue(tmp_path, monkeypatch)

        app._remember_message(
            channel_id="555",
            author="alice",
            content="hello",
            attachment_names=[],
            message_id="msg-1",
        )
        app._remember_message(
            channel_id="555",
            author="alice",
            content="hello",
            attachment_names=[],
            message_id="msg-1",  # same ID — duplicate
        )

        assert queue.qsize() == 1

    def test_enqueued_item_carries_bot_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bot messages enqueued to mempalace are marked is_bot=True."""
        app, queue = self._app_with_queue(tmp_path, monkeypatch)

        app._remember_message(
            channel_id="555",
            author="kynetic_agents",
            content="I can help with that.",
            attachment_names=[],
            is_bot=True,
        )

        item = queue.get_nowait()
        assert item.is_bot is True


# ---------------------------------------------------------------------------
# 5. Writer process guard
# ---------------------------------------------------------------------------

class TestMempalaceWriterGuard:
    def test_write_queue_is_none_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A freshly constructed app never has a write queue — mempalace_writer
        defaults to false and the queue is only created during run()."""
        app = _make_app(tmp_path, "model: test-model\n", monkeypatch)

        assert app._mempalace_write_queue is None

    def test_mempalace_session_is_none_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Session reference is only wired during run() after MCP servers start."""
        app = _make_app(tmp_path, "model: test-model\n", monkeypatch)

        assert app._mempalace_session is None
