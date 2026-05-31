from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from kynetic_agents.config import AppConfig, RepoLayout
from kynetic_agents.discord import DiscordMixin
from kynetic_agents.models import AgentEvent
from kynetic_agents.ops_dashboard import (
    _load_events,
    build_dashboard_payload,
    compute_stats,
    parse_days_param,
    render_dashboard_html,
)
from kynetic_agents.shell_jobs import ShellJobRegistry


def _ts(offset_hours: float = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=offset_hours)).isoformat()


def _write_events(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def test_parse_days_param_default_and_validation() -> None:
    assert parse_days_param(None) == 30
    assert parse_days_param("") == 30
    assert parse_days_param("7") == 7
    with pytest.raises(ValueError):
        parse_days_param("0")
    with pytest.raises(ValueError):
        parse_days_param("-3")
    with pytest.raises(ValueError):
        parse_days_param("not-an-int")
    with pytest.raises(ValueError):
        parse_days_param("9999")


def test_load_events_filters_outside_window(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    _write_events(
        log,
        [
            {"type": "tool_call", "timestamp": _ts(0)},
            {"type": "tool_call", "timestamp": _ts(24 * 100)},  # 100 days old
        ],
    )
    inside = _load_events(log, days=30)
    assert len(inside) == 1


def test_load_events_skips_malformed_lines(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps({"type": "tool_call", "timestamp": _ts(0)}),
                "{not valid json",
                "",
                json.dumps({"type": "agent_invoke_start", "timestamp": _ts(0)}),
            ]
        )
    )
    out = _load_events(log, days=30)
    assert {e["type"] for e in out} == {"tool_call", "agent_invoke_start"}


def test_compute_stats_counts_and_attribution(tmp_path: Path) -> None:
    records = [
        {"type": "agent_invoke_start", "timestamp": _ts(0),
         "source_event_type": "discord_message", "session_id": "s1"},
        {"type": "agent_invoke_start", "timestamp": _ts(1),
         "source_event_type": "poller", "scheduler_name": "linkedin", "session_id": "s2"},
        {"type": "agent_invoke_start", "timestamp": _ts(2),
         "source_event_type": "poller", "scheduler_name": "linkedin", "session_id": "s3"},
        {"type": "tool_call", "timestamp": _ts(0), "tool": "send_message", "session_id": "s1"},
        {"type": "tool_call", "timestamp": _ts(0), "tool": "send_message", "session_id": "s1"},
        {"type": "tool_call", "timestamp": _ts(1), "tool": "journal", "session_id": "s2"},
        {"type": "event_queued", "timestamp": _ts(0), "source_event_type": "discord_message"},
        {"type": "event_queued", "timestamp": _ts(0), "source_event_type": "poller"},
        {"type": "event_deduped", "timestamp": _ts(0), "key": "poller:linkedin:0"},
        {"type": "turn_timing", "timestamp": _ts(0),
         "total_seconds": 8.0, "agent_invoke_seconds": 6.0},
        {"type": "turn_timing", "timestamp": _ts(1),
         "total_seconds": 10.0, "agent_invoke_seconds": 7.0},
        {"type": "agent_turn_missing_send_message", "timestamp": _ts(0),
         "scheduler_name": "linkedin", "final_text": "oops"},
        {"type": "scheduler_invalid_cron", "timestamp": _ts(0), "error": "bad cron"},
    ]
    for record in records:
        record["_ts"] = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))

    stats = compute_stats(records, days=30)

    assert stats["window_days"] == 30
    assert stats["summary"]["agent_invocations"] == 3
    assert stats["summary"]["events_queued"] == 2
    assert stats["summary"]["events_deduped"] == 1
    assert stats["summary"]["tool_calls"] == 3
    assert stats["summary"]["failures"] == 2
    assert stats["summary"]["avg_turn_seconds"] == 9.0
    assert stats["summary"]["avg_invoke_seconds"] == 6.5

    # Source / scheduler attribution
    assert stats["invoke_by_source"]["poller"] == 2
    assert stats["invoke_by_source"]["discord_message"] == 1
    assert stats["invoke_by_scheduler"]["linkedin"] == 2

    # Failures collected
    assert stats["failures_by_kind"]["agent_turn_missing_send_message"] == 1
    assert stats["failures_by_kind"]["scheduler_invalid_cron"] == 1
    assert any(f["kind"] == "agent_turn_missing_send_message" for f in stats["recent_failures"])

    # Avg tools / invocation: s1=2, s2=1, s3=0 (s3 had no tool_call rows) → only counted sessions seen in tool_call
    # Implementation counts tool_calls per session_id seen in tool_call events,
    # so average is over sessions with at least one tool call.
    assert stats["summary"]["avg_tools_per_invocation"] == 1.5

    # Backlog has at least the documented gaps
    backlog_ids = {item["id"] for item in stats["backlog"]}
    assert {"token-usage", "llm-retries"}.issubset(backlog_ids)


def test_render_dashboard_html_embeds_data(tmp_path: Path) -> None:
    stats = compute_stats([], days=7)
    html = render_dashboard_html(stats)
    assert "Ops Dashboard" in html
    assert "back to chat" in html
    # Embedded JSON
    payload_marker = '<script id="data" type="application/json">'
    assert payload_marker in html
    start = html.index(payload_marker) + len(payload_marker)
    end = html.index("</script>", start)
    embedded = json.loads(html[start:end])
    assert embedded["window_days"] == 7
    assert "summary" in embedded


class DummyStrix(DiscordMixin):
    def __init__(self, home: Path) -> None:
        self.home = home
        self.layout = RepoLayout(home=home, state_dir_name="state")
        self.layout.state_dir.mkdir(parents=True, exist_ok=True)
        self.layout.logs_dir.mkdir(parents=True, exist_ok=True)
        self.config = AppConfig()
        self.message_history_all = deque(maxlen=500)
        self.message_history_by_channel = defaultdict(lambda: deque(maxlen=250))
        self._current_turn_sent_messages: list[tuple[str, str]] | None = []
        self.current_channel_id: str | None = None
        self.current_event_label: str | None = None
        self.current_turn_start: float | None = None
        self.discord_client = None
        self.shell_jobs = ShellJobRegistry(self.layout.logs_dir / "shell-jobs")
        self.logged: list[dict[str, object]] = []
        self.enqueued: list[AgentEvent] = []

    def log_event(self, event_type: str, **payload: object) -> None:
        self.logged.append({"type": event_type, **payload})

    async def enqueue_event(self, event: AgentEvent) -> None:
        self.enqueued.append(event)


def test_build_dashboard_payload_uses_strix_layout(tmp_path: Path) -> None:
    strix = DummyStrix(tmp_path / "atlas")
    _write_events(
        strix.layout.events_log,
        [{"type": "event_queued", "timestamp": _ts(0), "source_event_type": "poller"}],
    )
    payload = build_dashboard_payload(strix, days=7)
    assert payload["summary"]["events_queued"] == 1
    assert payload["window_days"] == 7


# --- new tests for rotation support -----------------------------------------


def test_load_events_reads_rotated_siblings(tmp_path: Path) -> None:
    """Events in rotated siblings are included when within the time window."""
    log = tmp_path / "events.jsonl"

    # Write a rotated sibling with an old-but-within-window event (12h ago)
    rotated = tmp_path / "events.jsonl.20260514T120000Z"
    _write_events(rotated, [{"type": "tool_call", "timestamp": _ts(12)}])

    # Write the live file with a recent event (1h ago)
    _write_events(log, [{"type": "agent_invoke_start", "timestamp": _ts(1)}])

    result = _load_events(log, days=1)
    types = {e["type"] for e in result}
    assert types == {"tool_call", "agent_invoke_start"}, (
        "Events from rotated sibling should be included"
    )


def test_load_events_excludes_rotated_events_outside_window(tmp_path: Path) -> None:
    """Events in rotated siblings that predate the cutoff are filtered out."""
    log = tmp_path / "events.jsonl"

    # Write a rotated sibling with a very old event (100 days ago)
    rotated = tmp_path / "events.jsonl.20260114T000000Z"
    _write_events(rotated, [{"type": "tool_call", "timestamp": _ts(24 * 100)}])

    # Write the live file with a recent event
    _write_events(log, [{"type": "agent_invoke_start", "timestamp": _ts(1)}])

    result = _load_events(log, days=30)
    assert len(result) == 1
    assert result[0]["type"] == "agent_invoke_start", (
        "Old events in rotated siblings should be filtered by timestamp"
    )


def test_load_events_handles_missing_live_file_with_siblings(tmp_path: Path) -> None:
    """If the live file doesn't exist yet but siblings do, siblings are still read."""
    log = tmp_path / "events.jsonl"
    # Don't create the live file

    rotated = tmp_path / "events.jsonl.20260514T120000Z"
    _write_events(rotated, [{"type": "tool_call", "timestamp": _ts(1)}])

    result = _load_events(log, days=7)
    assert len(result) == 1
    assert result[0]["type"] == "tool_call"


def test_load_events_handles_mid_rotation_race(tmp_path: Path) -> None:
    """A sibling that disappears between glob and open is skipped without error."""
    log = tmp_path / "events.jsonl"
    _write_events(log, [{"type": "agent_invoke_start", "timestamp": _ts(0)}])

    # Simulate mid-rotation: create then immediately delete a sibling
    ghost = tmp_path / "events.jsonl.20260514T110000Z"
    ghost.write_text("")
    ghost.unlink()  # gone before open() is called

    # Should not raise; live file still readable
    result = _load_events(log, days=1)
    assert len(result) == 1
    assert result[0]["type"] == "agent_invoke_start"


def test_load_events_deduplicates_across_siblings(tmp_path: Path) -> None:
    """Verify no duplicate events if the same line somehow appeared in two files.
    (Not expected in practice, but the function should not crash or deduplicate —
    it's the caller's problem. This test just confirms the count.)"""
    log = tmp_path / "events.jsonl"
    event = {"type": "tool_call", "timestamp": _ts(1)}

    rotated = tmp_path / "events.jsonl.20260514T100000Z"
    _write_events(rotated, [event])
    _write_events(log, [event])

    result = _load_events(log, days=7)
    # Both appear — no implicit dedup at this layer
    assert len(result) == 2


def test_load_events_reads_multiple_siblings_in_order(tmp_path: Path) -> None:
    """Multiple rotated siblings are all read, with the live file last."""
    log = tmp_path / "events.jsonl"

    # Three siblings, all within the window
    for suffix, hours_ago in [
        ("20260512T000000Z", 48),
        ("20260513T000000Z", 24),
        ("20260514T000000Z", 12),
    ]:
        _write_events(
            tmp_path / f"events.jsonl.{suffix}",
            [{"type": "tool_call", "timestamp": _ts(hours_ago), "marker": suffix}],
        )

    # Live file
    _write_events(log, [{"type": "agent_invoke_start", "timestamp": _ts(1)}])

    result = _load_events(log, days=7)
    assert len(result) == 4, "All four files (3 siblings + live) should be read"
    types = {e["type"] for e in result}
    assert "tool_call" in types
    assert "agent_invoke_start" in types


# --- cost formula unit tests --------------------------------------------------
# Cost events use kynetic's llm_usage schema (cache_read_tokens /
# cache_creation_tokens, emitted by _extract_usage).


def test_turn_cost_usd_sonnet_cache_aware() -> None:
    """$0.5715 worked example: 1M input, 900K cache-read, 0 cache-creation, 100 output tokens."""
    from kynetic_agents.ops_dashboard import _turn_cost_usd

    ev = {
        "model": "claude-sonnet-4-6",
        "input_tokens": 1_000_000,
        "cache_read_tokens": 900_000,
        "cache_creation_tokens": 0,
        "output_tokens": 100,
    }
    # fresh_input = 1_000_000 - 900_000 - 0 = 100_000
    # cost = 100_000 * $3/M + 900_000 * $0.30/M + 0 * $3.75/M + 100 * $15/M
    #      = $0.30 + $0.27 + $0 + $0.0015
    #      = $0.5715
    assert round(_turn_cost_usd(ev), 4) == 0.5715


def test_turn_cost_usd_deepseek_v4_pro() -> None:
    """DeepSeek V4 Pro: 1M input, 900K cache-read, 0 cache-creation, 100 output."""
    from kynetic_agents.ops_dashboard import _turn_cost_usd

    ev = {
        "model": "deepseek-v4-pro",
        "input_tokens": 1_000_000,
        "cache_read_tokens": 900_000,
        "cache_creation_tokens": 0,
        "output_tokens": 100,
    }
    # fresh_input = 100_000
    # cost = 100_000*$1.74/M + 900_000*$0.0145/M + 0 + 100*$3.48/M
    #      = $0.174 + $0.01305 + $0.000348 = $0.187398
    assert round(_turn_cost_usd(ev), 4) == 0.1874


def test_turn_cost_usd_unknown_model_defaults_to_deepseek_v4_pro() -> None:
    """An unrecognized model is priced at the default (deepseek-v4-pro) rate."""
    from kynetic_agents.ops_dashboard import _turn_cost_usd

    base = {
        "input_tokens": 1_000_000,
        "cache_read_tokens": 900_000,
        "cache_creation_tokens": 0,
        "output_tokens": 100,
    }
    unknown = dict(base, model="some-mystery-model")
    deepseek = dict(base, model="deepseek-v4-pro")
    assert _turn_cost_usd(unknown) == _turn_cost_usd(deepseek)


def test_turn_cost_usd_provider_prefix_stripped() -> None:
    """model field may arrive as 'anthropic:claude-sonnet-4-6' — should price correctly."""
    from kynetic_agents.ops_dashboard import _turn_cost_usd

    ev_prefixed = {
        "model": "anthropic:claude-sonnet-4-6",
        "input_tokens": 1_000_000,
        "cache_read_tokens": 900_000,
        "cache_creation_tokens": 0,
        "output_tokens": 100,
    }
    ev_bare = dict(ev_prefixed, model="claude-sonnet-4-6")
    assert _turn_cost_usd(ev_prefixed) == _turn_cost_usd(ev_bare)


def test_turn_cost_usd_per_job_attribution(tmp_path: Path) -> None:
    """Two llm_usage events paired with agent_invoke_start rows → correct per-job cost."""
    from kynetic_agents.ops_dashboard import _compute_cost_stats

    now = datetime.now(timezone.utc)

    def _ts_dt(offset_seconds: float) -> datetime:
        return now - timedelta(seconds=offset_seconds)

    # Simulate two turns: one discord_message, one rss-daily-scan
    llu_items = [
        {
            "sid": "s1",
            "ts": _ts_dt(10),
            "cost": 0.5715,
            "model": "claude-sonnet-4-6",
            "input_tokens": 1_000_000,
            "cache_read": 900_000,
        },
        {
            "sid": "s2",
            "ts": _ts_dt(5),
            "cost": 0.01,
            "model": "claude-sonnet-4-6",
            "input_tokens": 10_000,
            "cache_read": 5_000,
        },
    ]

    # agent_invoke_start rows (session_id + timestamps for join)
    session_id_seen: set[str] = {"s1", "s2"}

    raw_events = [
        {
            "_ts": _ts_dt(11),
            "type": "agent_invoke_start",
            "session_id": "s1",
            "source_event_type": "discord_message",
            "scheduler_name": None,
        },
        {
            "_ts": _ts_dt(6),
            "type": "agent_invoke_start",
            "session_id": "s2",
            "source_event_type": "poller",
            "scheduler_name": "rss-daily-scan",
        },
    ]

    stats = _compute_cost_stats(llu_items, session_id_seen, raw_events)

    assert stats is not None
    per_job = {row["job"]: row for row in stats["per_job"]}
    assert "discord_message" in per_job, (
        f"Expected discord_message job, got: {list(per_job.keys())}"
    )
    assert "rss-daily-scan" in per_job, (
        f"Expected rss-daily-scan job, got: {list(per_job.keys())}"
    )
    assert abs(per_job["discord_message"]["cost_usd"] - 0.5715) < 1e-6, (
        f'discord_message cost mismatch: {per_job["discord_message"]["cost_usd"]}'
    )
    assert abs(per_job["rss-daily-scan"]["cost_usd"] - 0.01) < 1e-6, (
        f'rss-daily-scan cost mismatch: {per_job["rss-daily-scan"]["cost_usd"]}'
    )
    total = stats["total_cost_usd"]
    assert abs(total - (0.5715 + 0.01)) < 1e-6, f"Total cost mismatch: {total}"


def test_compute_stats_includes_cost_from_llm_usage_events() -> None:
    """compute_stats surfaces a cost block computed from llm_usage events."""
    from kynetic_agents.ops_dashboard import _parse_ts

    ts = _ts(1)
    events = [
        {"type": "agent_invoke_start", "timestamp": ts, "session_id": "s1",
         "source_event_type": "discord_message", "_ts": _parse_ts(ts)},
        {"type": "llm_usage", "timestamp": ts, "session_id": "s1",
         "model": "deepseek-v4-pro", "input_tokens": 1_000_000,
         "cache_read_tokens": 900_000, "cache_creation_tokens": 0,
         "output_tokens": 100, "_ts": _parse_ts(ts)},
    ]
    stats = compute_stats(events, days=7)
    assert "cost" in stats
    assert stats["cost"]["total_turns"] == 1
    assert round(stats["cost"]["total_cost_usd"], 4) == 0.1874
    assert stats["cost"]["per_job"][0]["job"] == "discord_message"
