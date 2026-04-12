"""Tests for the home_channels gate (Steps 3 & 4).

Step 4 — config schema: home_channels field on AppConfig / load_config
Step 3 — on_message gate: should_process_discord_message honours home_channels

TDD: these tests are written first and are expected to FAIL until the
implementation is added.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import kynetic_agents.app as app_mod
import kynetic_agents.discord as discord_mod
from kynetic_agents.config import AppConfig, load_config, RepoLayout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyAgent:
    async def ainvoke(self, _: dict[str, Any]) -> dict[str, Any]:
        return {"messages": []}


def _stub_agent_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_mod, "create_deep_agent", lambda **_: DummyAgent())


def _make_app(tmp_path: Path, config_yaml: str, monkeypatch: pytest.MonkeyPatch) -> app_mod.OpenStrixApp:
    _stub_agent_factory(monkeypatch)
    (tmp_path / "config.yaml").write_text(config_yaml, encoding="utf-8")
    return app_mod.OpenStrixApp(tmp_path)


# ---------------------------------------------------------------------------
# Step 4 — config schema
# ---------------------------------------------------------------------------

class TestHomeChannelsConfig:
    def test_home_channels_defaults_to_empty_list_when_absent(self, tmp_path: Path) -> None:
        """home_channels should be [] when the key is not in config.yaml."""
        (tmp_path / "config.yaml").write_text("model: test-model\n", encoding="utf-8")
        layout = RepoLayout(home=tmp_path, state_dir_name="state")
        config = load_config(layout)
        assert config.home_channels == []

    def test_home_channels_parsed_as_list_of_strings(self, tmp_path: Path) -> None:
        """home_channels should load a YAML list of channel names."""
        (tmp_path / "config.yaml").write_text(
            "home_channels:\n  - archon\n  - mission-control\n",
            encoding="utf-8",
        )
        layout = RepoLayout(home=tmp_path, state_dir_name="state")
        config = load_config(layout)
        assert config.home_channels == ["archon", "mission-control"]

    def test_home_channels_normalises_integer_ids_to_strings(self, tmp_path: Path) -> None:
        """Channel IDs given as integers in YAML are stored as strings."""
        (tmp_path / "config.yaml").write_text(
            "home_channels:\n  - 1234567890\n  - neuron\n",
            encoding="utf-8",
        )
        layout = RepoLayout(home=tmp_path, state_dir_name="state")
        config = load_config(layout)
        assert "1234567890" in config.home_channels
        assert "neuron" in config.home_channels

    def test_appconfig_home_channels_field_defaults_to_empty_list(self) -> None:
        """AppConfig() should have home_channels == [] out of the box."""
        cfg = AppConfig()
        assert cfg.home_channels == []

    def test_appconfig_accepts_home_channels_kwarg(self) -> None:
        """AppConfig should accept home_channels at construction time."""
        cfg = AppConfig(home_channels=["forge", "ops"])
        assert cfg.home_channels == ["forge", "ops"]


# ---------------------------------------------------------------------------
# Step 3 — should_process_discord_message home_channels gate
# ---------------------------------------------------------------------------

class TestHomeChannelsGate:
    """Parameterised tests for the home_channels routing gate.

    The new signature of should_process_discord_message is:

        should_process_discord_message(
            *,
            author_is_bot: bool,
            author_id: str | int | None,
            channel_name: str = "",
            channel_id: str | int | None = None,
            mentions_bot: bool = False,
            author_addressed_bot: bool = False,
        ) -> bool

    ``author_addressed_bot`` is True when the message is from a bot in
    always_respond_bot_ids AND the message content contains the bot's own
    ID or @name (i.e. the spoke explicitly addressed this agent).
    """

    # ── When home_channels is NOT configured, existing behaviour is unchanged ──
    # These tests also verify that the new kwargs (channel_name, channel_id,
    # mentions_bot, author_addressed_bot) are completely ignored when
    # home_channels is empty — they must not change prior behaviour.

    def test_human_passes_when_no_home_channels(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_app(tmp_path, "model: test-model\n", monkeypatch)
        assert app.should_process_discord_message(
            author_is_bot=False, author_id=None
        ) is True

    def test_human_passes_when_no_home_channels_even_with_channel_kwargs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """New channel/mention kwargs must not block humans when home_channels is empty."""
        app = _make_app(tmp_path, "model: test-model\n", monkeypatch)
        assert app.should_process_discord_message(
            author_is_bot=False,
            author_id=None,
            channel_name="some-random-channel",
            channel_id="9999",
            mentions_bot=False,
            author_addressed_bot=False,
        ) is True

    def test_unknown_bot_blocked_when_no_home_channels(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_app(tmp_path, "model: test-model\n", monkeypatch)
        assert app.should_process_discord_message(
            author_is_bot=True, author_id="99"
        ) is False

    def test_unknown_bot_blocked_when_no_home_channels_even_if_mentioned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """mentions_bot=True must not promote an unknown bot when home_channels is empty."""
        app = _make_app(tmp_path, "model: test-model\n", monkeypatch)
        assert app.should_process_discord_message(
            author_is_bot=True,
            author_id="99",
            mentions_bot=True,
            author_addressed_bot=True,
        ) is False

    def test_always_respond_bot_passes_when_no_home_channels(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_app(
            tmp_path,
            "always_respond_bot_ids:\n  - 42\n",
            monkeypatch,
        )
        assert app.should_process_discord_message(
            author_is_bot=True, author_id="42"
        ) is True

    def test_always_respond_bot_passes_when_no_home_channels_regardless_of_channel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """always_respond bots must not be gated by channel when home_channels is empty."""
        app = _make_app(
            tmp_path,
            "always_respond_bot_ids:\n  - 42\n",
            monkeypatch,
        )
        assert app.should_process_discord_message(
            author_is_bot=True,
            author_id="42",
            channel_name="some-random-channel",
            mentions_bot=False,
            author_addressed_bot=False,
        ) is True

    # ── With home_channels configured ────────────────────────────────────────

    def test_human_in_home_channel_by_name_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_app(
            tmp_path,
            "home_channels:\n  - archon\n  - mission-control\n",
            monkeypatch,
        )
        assert app.should_process_discord_message(
            author_is_bot=False,
            author_id=None,
            channel_name="archon",
        ) is True

    def test_human_in_home_channel_by_id_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_app(
            tmp_path,
            "home_channels:\n  - '888111222'\n",
            monkeypatch,
        )
        assert app.should_process_discord_message(
            author_is_bot=False,
            author_id=None,
            channel_id="888111222",
        ) is True

    def test_human_not_in_home_channel_and_no_mention_is_blocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Human in a random channel should be ignored when home_channels is set."""
        app = _make_app(
            tmp_path,
            "home_channels:\n  - archon\n",
            monkeypatch,
        )
        assert app.should_process_discord_message(
            author_is_bot=False,
            author_id=None,
            channel_name="general",
            mentions_bot=False,
        ) is False

    def test_human_not_in_home_channel_but_mentions_bot_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """@mention anywhere overrides the home_channel restriction."""
        app = _make_app(
            tmp_path,
            "home_channels:\n  - archon\n",
            monkeypatch,
        )
        assert app.should_process_discord_message(
            author_is_bot=False,
            author_id=None,
            channel_name="general",
            mentions_bot=True,
        ) is True

    def test_always_respond_bot_in_home_channel_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_app(
            tmp_path,
            "always_respond_bot_ids:\n  - 42\nhome_channels:\n  - archon\n",
            monkeypatch,
        )
        assert app.should_process_discord_message(
            author_is_bot=True,
            author_id="42",
            channel_name="archon",
        ) is True

    def test_always_respond_bot_not_in_home_channel_and_not_addressed_is_blocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spoke bot outside the home channel that did NOT @mention us → silence."""
        app = _make_app(
            tmp_path,
            "always_respond_bot_ids:\n  - 42\nhome_channels:\n  - archon\n",
            monkeypatch,
        )
        assert app.should_process_discord_message(
            author_is_bot=True,
            author_id="42",
            channel_name="neuron",
            author_addressed_bot=False,
        ) is False

    def test_always_respond_bot_not_in_home_channel_but_addressed_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spoke bot that sent DONE/BLOCKER containing our @mention → respond."""
        app = _make_app(
            tmp_path,
            "always_respond_bot_ids:\n  - 42\nhome_channels:\n  - archon\n",
            monkeypatch,
        )
        assert app.should_process_discord_message(
            author_is_bot=True,
            author_id="42",
            channel_name="neuron",
            author_addressed_bot=True,
        ) is True

    def test_unknown_bot_still_blocked_even_with_home_channels(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """First filter (bot not in always_respond_bot_ids) runs before home_channels gate."""
        app = _make_app(
            tmp_path,
            "always_respond_bot_ids:\n  - 42\nhome_channels:\n  - archon\n",
            monkeypatch,
        )
        assert app.should_process_discord_message(
            author_is_bot=True,
            author_id="99",  # not in allowlist
            channel_name="archon",
            mentions_bot=True,
        ) is False

    def test_integer_channel_id_matched_against_string_home_channel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Channel IDs should be compared as strings regardless of input type."""
        app = _make_app(
            tmp_path,
            "home_channels:\n  - '777000111'\n",
            monkeypatch,
        )
        # Pass integer channel_id — should still match the string "777000111"
        assert app.should_process_discord_message(
            author_is_bot=False,
            author_id=None,
            channel_id=777000111,
        ) is True


# ---------------------------------------------------------------------------
# DiscordBridge.on_message wiring
# Verify that on_message actually extracts the new context from the Discord
# message object and forwards it to should_process_discord_message.
# Without this wiring the home_channels gate is a dead letter.
# ---------------------------------------------------------------------------

def _make_bridge(app: app_mod.OpenStrixApp, bot_user: Any) -> discord_mod.DiscordBridge:
    """Construct a DiscordBridge without going through discord.Client startup.

    discord.Client.user is a read-only property backed by self._connection.user,
    so we set _connection directly rather than assigning to the property.
    """
    bridge = object.__new__(discord_mod.DiscordBridge)
    bridge._app = app
    # discord.py: ClientUser is stored at self._connection.user
    bridge._connection = SimpleNamespace(user=bot_user)
    return bridge


def _fake_message(
    *,
    author_id: int = 999,
    author_is_bot: bool = False,
    channel_name: str = "general",
    channel_id: int = 555,
    mentions: list[Any] | None = None,
    content: str = "hello",
) -> SimpleNamespace:
    author = SimpleNamespace(id=author_id, bot=author_is_bot, name=f"user-{author_id}")
    channel = SimpleNamespace(id=channel_id, name=channel_name)
    return SimpleNamespace(
        id=1,
        author=author,
        channel=channel,
        mentions=mentions if mentions is not None else [],
        content=content,
        attachments=[],
    )


class TestOnMessageWiring:
    """DiscordBridge.on_message must extract all gate context from the message."""

    @pytest.mark.asyncio
    async def test_on_message_passes_channel_name_to_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_app(tmp_path, "home_channels:\n  - archon\n", monkeypatch)
        bot_user = SimpleNamespace(id=111, name="archon-bot")
        bridge = _make_bridge(app, bot_user)

        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            app,
            "should_process_discord_message",
            lambda **kw: calls.append(kw) or False,
        )

        await bridge.on_message(_fake_message(channel_name="archon"))

        assert calls, "should_process_discord_message was never called"
        assert calls[0]["channel_name"] == "archon"

    @pytest.mark.asyncio
    async def test_on_message_passes_channel_id_to_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_app(tmp_path, "home_channels:\n  - '12345'\n", monkeypatch)
        bot_user = SimpleNamespace(id=111, name="archon-bot")
        bridge = _make_bridge(app, bot_user)

        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            app,
            "should_process_discord_message",
            lambda **kw: calls.append(kw) or False,
        )

        await bridge.on_message(_fake_message(channel_id=12345))

        assert calls
        assert calls[0]["channel_id"] == 12345

    @pytest.mark.asyncio
    async def test_on_message_sets_mentions_bot_true_when_bot_is_mentioned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_app(tmp_path, "home_channels:\n  - archon\n", monkeypatch)
        bot_user = SimpleNamespace(id=111, name="archon-bot")
        bridge = _make_bridge(app, bot_user)

        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            app,
            "should_process_discord_message",
            lambda **kw: calls.append(kw) or False,
        )

        # Message mentions the bot user object
        await bridge.on_message(
            _fake_message(channel_name="general", mentions=[bot_user])
        )

        assert calls
        assert calls[0]["mentions_bot"] is True

    @pytest.mark.asyncio
    async def test_on_message_sets_mentions_bot_false_when_bot_not_mentioned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_app(tmp_path, "home_channels:\n  - archon\n", monkeypatch)
        bot_user = SimpleNamespace(id=111, name="archon-bot")
        bridge = _make_bridge(app, bot_user)

        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            app,
            "should_process_discord_message",
            lambda **kw: calls.append(kw) or False,
        )

        await bridge.on_message(_fake_message(channel_name="general", mentions=[]))

        assert calls
        assert calls[0]["mentions_bot"] is False

    @pytest.mark.asyncio
    async def test_on_message_sets_author_addressed_bot_via_id_in_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """always_respond bot whose message contains the bot's ID → author_addressed_bot=True."""
        app = _make_app(
            tmp_path,
            "always_respond_bot_ids:\n  - 42\nhome_channels:\n  - archon\n",
            monkeypatch,
        )
        bot_user = SimpleNamespace(id=111, name="archon-bot")
        bridge = _make_bridge(app, bot_user)

        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            app,
            "should_process_discord_message",
            lambda **kw: calls.append(kw) or False,
        )

        await bridge.on_message(
            _fake_message(
                author_id=42,
                author_is_bot=True,
                channel_name="neuron",
                content="DONE <@111> task complete",  # contains bot's ID
            )
        )

        assert calls
        assert calls[0]["author_addressed_bot"] is True

    @pytest.mark.asyncio
    async def test_on_message_sets_author_addressed_bot_false_when_not_addressed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_app(
            tmp_path,
            "always_respond_bot_ids:\n  - 42\nhome_channels:\n  - archon\n",
            monkeypatch,
        )
        bot_user = SimpleNamespace(id=111, name="archon-bot")
        bridge = _make_bridge(app, bot_user)

        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            app,
            "should_process_discord_message",
            lambda **kw: calls.append(kw) or False,
        )

        await bridge.on_message(
            _fake_message(
                author_id=42,
                author_is_bot=True,
                channel_name="neuron",
                content="DONE task complete",  # does NOT mention our bot
            )
        )

        assert calls
        assert calls[0]["author_addressed_bot"] is False

    @pytest.mark.asyncio
    async def test_on_message_gates_out_message_not_in_home_channel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: human in #general is silenced when home_channels=[archon]."""
        app = _make_app(tmp_path, "home_channels:\n  - archon\n", monkeypatch)
        bot_user = SimpleNamespace(id=111, name="archon-bot")
        bridge = _make_bridge(app, bot_user)

        handled: list[Any] = []
        monkeypatch.setattr(app, "handle_discord_message", lambda m: handled.append(m))

        await bridge.on_message(_fake_message(channel_name="general"))

        assert handled == [], "message should have been silenced by the home_channels gate"

    @pytest.mark.asyncio
    async def test_on_message_allows_message_in_home_channel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: human in #archon is processed when home_channels=[archon]."""
        app = _make_app(tmp_path, "home_channels:\n  - archon\n", monkeypatch)
        bot_user = SimpleNamespace(id=111, name="archon-bot")
        bridge = _make_bridge(app, bot_user)

        handled: list[Any] = []

        async def fake_handle(m: Any) -> None:
            handled.append(m)

        monkeypatch.setattr(app, "handle_discord_message", fake_handle)

        await bridge.on_message(_fake_message(channel_name="archon"))

        assert len(handled) == 1


# ---------------------------------------------------------------------------
# Scaffolding — DEFAULT_CONFIG and _ensure_config_defaults
# ---------------------------------------------------------------------------

class TestHomeChannelsScaffolding:
    """home_channels must be present in the scaffolded config and back-filled
    into existing configs that pre-date the feature."""

    def test_default_config_string_includes_home_channels(self) -> None:
        """DEFAULT_CONFIG must declare home_channels so new agents see the field."""
        import yaml as _yaml
        from kynetic_agents.config import DEFAULT_CONFIG
        parsed = _yaml.safe_load(DEFAULT_CONFIG)
        assert "home_channels" in parsed
        assert parsed["home_channels"] == []

    def test_ensure_config_defaults_adds_home_channels_to_existing_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing config.yaml without home_channels gets it backfilled on startup."""
        # Write a config that looks like a pre-feature agent (no home_channels key)
        (tmp_path / "config.yaml").write_text(
            "model: claude-haiku-4-5-20251001\nalways_respond_bot_ids: []\n",
            encoding="utf-8",
        )
        _stub_agent_factory(monkeypatch)
        app_mod.OpenStrixApp(tmp_path)  # triggers _ensure_config_defaults

        import yaml as _yaml
        updated = _yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
        assert "home_channels" in updated
        assert updated["home_channels"] == []

    def test_ensure_config_defaults_does_not_overwrite_existing_home_channels(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An agent that already has home_channels set must not have it reset."""
        (tmp_path / "config.yaml").write_text(
            "model: claude-haiku-4-5-20251001\nhome_channels:\n  - archon\n  - mission-control\n",
            encoding="utf-8",
        )
        _stub_agent_factory(monkeypatch)
        app_mod.OpenStrixApp(tmp_path)

        import yaml as _yaml
        updated = _yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
        assert updated["home_channels"] == ["archon", "mission-control"]
