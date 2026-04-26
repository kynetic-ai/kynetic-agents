"""Tests for the mempalace shared-memory integration.

Covers four areas of intent:
  1. Config — new fields parse correctly from YAML.
  2. MCP allowed_tools filter — servers only expose permitted tools to agents.
  3. Channel detection — _is_mempalace_channel() identifies configured channels
     by ID or by name.
  4. Message enqueuing — on_message() enqueues all messages (human and bot) in
     mempalace_channels before the home_channels gate runs.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import kynetic_agents.app as app_mod
import kynetic_agents.discord as discord_mod
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

def _make_bridge_for_enqueue(app: app_mod.OpenStrixApp) -> discord_mod.DiscordBridge:
    """Construct a DiscordBridge without going through discord.Client startup."""
    bridge = object.__new__(discord_mod.DiscordBridge)
    bridge._app = app
    bridge._connection = SimpleNamespace(user=SimpleNamespace(id=999, name="test-bot"))
    return bridge


def _fake_discord_message(
    *,
    channel_id: int = 555,
    channel_name: str = "collab",
    content: str = "hello world",
    author_is_bot: bool = False,
    author_id: int = 1,
    message_id: int = 1001,
) -> SimpleNamespace:
    author = SimpleNamespace(id=author_id, bot=author_is_bot, name=f"user-{author_id}")
    channel = SimpleNamespace(id=channel_id, name=channel_name)
    return SimpleNamespace(
        id=message_id,
        author=author,
        channel=channel,
        mentions=[],
        content=content,
        attachments=[],
    )


class TestMempalaceMessageEnqueue:
    """All messages in mempalace_channels must be enqueued regardless of who sent them
    or whether the home_channels gate would process them.

    Enqueue happens in on_message() *before* should_process_discord_message(), so
    the collaboration channel is captured even when the agent isn't tagged and would
    otherwise ignore the message. Both human and bot messages are stored.
    """

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

    @pytest.mark.asyncio
    async def test_human_message_in_mempalace_channel_is_enqueued(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Human messages in a mempalace channel must be captured."""
        app, queue = self._app_with_queue(tmp_path, monkeypatch)
        bridge = _make_bridge_for_enqueue(app)
        monkeypatch.setattr(app, "should_process_discord_message", lambda **kw: False)

        await bridge.on_message(
            _fake_discord_message(channel_id=555, content="hello world", author_is_bot=False)
        )

        assert queue.qsize() == 1
        item = queue.get_nowait()
        assert item.channel_id == "555"
        assert item.content == "hello world"
        assert item.is_bot is False

    @pytest.mark.asyncio
    async def test_bot_message_in_mempalace_channel_is_enqueued(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bot messages in a mempalace channel must also be captured, marked is_bot=True."""
        app, queue = self._app_with_queue(tmp_path, monkeypatch)
        bridge = _make_bridge_for_enqueue(app)
        monkeypatch.setattr(app, "should_process_discord_message", lambda **kw: False)

        await bridge.on_message(
            _fake_discord_message(channel_id=555, content="I can help.", author_is_bot=True)
        )

        assert queue.qsize() == 1
        item = queue.get_nowait()
        assert item.is_bot is True

    @pytest.mark.asyncio
    async def test_message_in_non_mempalace_channel_is_not_enqueued(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, queue = self._app_with_queue(tmp_path, monkeypatch)
        bridge = _make_bridge_for_enqueue(app)
        monkeypatch.setattr(app, "should_process_discord_message", lambda **kw: False)

        await bridge.on_message(_fake_discord_message(channel_id=999, content="hello world"))

        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_enqueue_happens_before_home_channels_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate returning False must not prevent the message from being stored.

        This is the core architectural guarantee: the collaboration channel is captured
        in mempalace regardless of whether any agent processes the message."""
        app, queue = self._app_with_queue(tmp_path, monkeypatch)
        bridge = _make_bridge_for_enqueue(app)
        gate_calls: list[bool] = []
        monkeypatch.setattr(
            app,
            "should_process_discord_message",
            lambda **kw: gate_calls.append(True) or False,
        )

        await bridge.on_message(_fake_discord_message(channel_id=555, content="human update"))

        assert gate_calls, "gate should have been called"
        assert queue.qsize() == 1, "message must be enqueued even though gate returned False"

    @pytest.mark.asyncio
    async def test_empty_content_is_not_enqueued(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, queue = self._app_with_queue(tmp_path, monkeypatch)
        bridge = _make_bridge_for_enqueue(app)
        monkeypatch.setattr(app, "should_process_discord_message", lambda **kw: False)

        await bridge.on_message(_fake_discord_message(channel_id=555, content="   "))

        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_no_queue_does_not_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-writer processes have _mempalace_write_queue=None; on_message must not blow up."""
        app = _make_app(
            tmp_path,
            "mempalace_channels:\n  - '555'\n",
            monkeypatch,
        )
        assert app._mempalace_write_queue is None
        bridge = _make_bridge_for_enqueue(app)
        monkeypatch.setattr(app, "should_process_discord_message", lambda **kw: False)

        await bridge.on_message(_fake_discord_message(channel_id=555, content="hello"))

    def test_remember_message_does_not_enqueue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Structural guarantee: _remember_message has no enqueue logic.

        History replay on startup calls _remember_message(persist=False) for every
        stored message. Since enqueue was moved to on_message(), this path can never
        re-ingest the full chat history on restart."""
        app, queue = self._app_with_queue(tmp_path, monkeypatch)

        app._remember_message(
            channel_id="555",
            author="alice",
            content="hello world",
            attachment_names=[],
        )

        assert queue.qsize() == 0


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
