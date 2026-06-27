from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import shutil
import subprocess
import tempfile
import time
from uuid import uuid4
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import discord
import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent
from deepagents.backends import FilesystemBackend
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import EditResult, FileUploadResponse, WriteResult
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .builtin_skills import BUILTIN_HOME_DIRNAME, sync_builtin_skills_home
from .config import (
    DEFAULT_CONFIG,
    DEFAULT_MODEL,
    DEFAULT_MODEL_PROVIDER,
    DEFAULT_PRE_COMMIT_SCRIPT,
    DEFAULT_SCHEDULER,
    STATE_DIR_NAME,
    AppConfig,
    DEFAULT_MODEL_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_MAX_RETRIES,
    DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS,
    RepoLayout,
    bootstrap_home_repo,
    load_config,
)
from .mcp_client import MCPManager, MCPServerConfig
from .phone_book import enrich_from_jsonl, load_phone_book, render_aliases_block
from .discord import (
    DISCORD_HISTORY_REFRESH_LIMIT,
    DISCORD_MESSAGE_CHAR_LIMIT,
    ERROR_REACTION_EMOJI,
    WARNING_REACTION_EMOJI,
    DiscordBridge,
    DiscordMixin,
    _chunk_discord_message,
)
from .hooks import HookManager
from .models import AgentEvent, MempalaceWriteItem
from .prompts import DEFAULT_CHECKPOINT, MEMPALACE_SECTION, SYSTEM_PROMPT, render_folders_section, render_turn_prompt
from .readonly_backend import (
    BUILTIN_SKILLS_ROUTE,
    LoggingWriteGuardBackend,
    WriteGuardBackend,
    build_builtin_skills_backend,
)
from .scheduler import SchedulerJob, SchedulerMixin
from .shell_jobs import ShellJobRegistry
from .supervisor import Supervisor
from .tools import (
    SEND_MESSAGE_LOOP_HARD_LIMIT,
    SEND_MESSAGE_LOOP_SIMILARITY_THRESHOLD,
    SEND_MESSAGE_LOOP_SOFT_LIMIT,
    SEND_MESSAGE_LOOP_WARN_LIMIT,
    SendMessageCircuitBreakerStop,
    ToolsMixin,
)

UTC = timezone.utc
LOG_ROLL_BYTES = 1_000_000
TRANSIENT_PROVIDER_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

MEMPALACE_READ_TOOLS = [
    "mempalace_search",
    "mempalace_get_drawer",
    "mempalace_kg_query",
    "mempalace_kg_timeline",
    "mempalace_get_taxonomy",
    "mempalace_list_drawers",
]

MEMPALACE_WRITE_TOOLS = [
    "mempalace_kg_add",
    "mempalace_kg_invalidate",
    "mempalace_diary_write",
]


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _roll_if_needed(path: Path, max_bytes: int = LOG_ROLL_BYTES) -> None:
    if not path.exists():
        return
    if path.stat().st_size <= max_bytes:
        return
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    path.rename(path.with_suffix(f"{path.suffix}.{stamp}"))


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    _roll_if_needed(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")


def _tail_jsonl(path: Path, count: int) -> list[dict[str, Any]]:
    if count <= 0 or not path.exists():
        return []
    lines: deque[dict[str, Any]] = deque(maxlen=count)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(lines)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "block"


def _model_for_deep_agents(model_name: str) -> str:
    cleaned = model_name.strip()
    if ":" in cleaned:
        return cleaned
    return f"{DEFAULT_MODEL_PROVIDER}:{cleaned}"


def _build_chat_model(
    model_name: str,
    *,
    max_retries: int = DEFAULT_MODEL_MAX_RETRIES,
    max_tokens: int = DEFAULT_MODEL_MAX_OUTPUT_TOKENS,
    request_timeout_seconds: int = DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS,
    thinking: bool = False,
    reasoning_effort: str | None = None,
) -> Any:
    # langchain-anthropic falls back to 4096 max output tokens for any model
    # not in its Claude-only profile table. MiniMax-M2.5 triggers that fallback,
    # which truncates tool_use args (e.g. write_file content) mid-stream. Pass
    # max_tokens explicitly so large tool calls fit.
    model_init_params: dict[str, Any] = {
        "max_retries": max(0, int(max_retries)),
        "max_tokens": max(1, int(max_tokens)),
        "timeout": max(1, int(request_timeout_seconds)),
    }
    is_deepseek = "deepseek" in model_name.lower()
    if thinking:
        # DeepSeek's Anthropic-compatible endpoint accepts the `thinking` param
        # but ignores `budget_tokens` (it self-scales reasoning depth). The value
        # is kept < max_tokens so langchain-anthropic's client-side validation
        # passes. `temperature` is supported on this Anthropic-compatible
        # endpoint (range 0.0-2.0); extended thinking pins it to 1.
        model_init_params["thinking"] = {"type": "enabled", "budget_tokens": 4096}
        model_init_params["temperature"] = 1
    elif is_deepseek:
        # DeepSeek V4's Anthropic-compatible endpoint defaults thinking to ON
        # (verified live), unlike real Anthropic where omitting the param means
        # off. Leaving it off would let the main agent reason on every turn,
        # contrary to the fast/cheap default we want. The Anthropic API accepts
        # `{"type": "disabled"}` (honored by DeepSeek) to force it off. Scoped to
        # DeepSeek: other providers (MiniMax, Moonshot, real Anthropic) already
        # default thinking off when the param is omitted, and we have not
        # verified they accept the disabled value.
        model_init_params["thinking"] = {"type": "disabled"}
    if reasoning_effort:
        # DeepSeek V4's Anthropic-compatible endpoint controls reasoning depth
        # via `output_config.effort` ("high" | "max"); it ignores `budget_tokens`.
        # langchain-anthropic forwards unknown `model_kwargs` in the request body,
        # so this reaches DeepSeek as a top-level `output_config` field. Only
        # meaningful when thinking is enabled.
        model_init_params.setdefault("model_kwargs", {})["output_config"] = {
            "effort": reasoning_effort,
        }
    return init_chat_model(model_name, **model_init_params)



def _exception_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    if isinstance(status_code, str):
        stripped = status_code.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _exception_request_id(exc: Exception) -> str | None:
    request_id = getattr(exc, "request_id", None)
    if request_id is None:
        return None
    text = str(request_id).strip()
    return text or None


def _error_log_fields(exc: Exception) -> dict[str, Any]:
    payload: dict[str, Any] = {"error_class": type(exc).__name__}
    status_code = _exception_status_code(exc)
    if status_code is not None:
        payload["error_status_code"] = status_code
    request_id = _exception_request_id(exc)
    if request_id is not None:
        payload["provider_request_id"] = request_id
    return payload


def _is_transient_provider_error(exc: Exception) -> bool:
    status_code = _exception_status_code(exc)
    if status_code in TRANSIENT_PROVIDER_STATUS_CODES:
        return True
    if status_code is not None and status_code >= 500:
        return True

    error_name = type(exc).__name__.lower()
    if error_name in {
        "apiconnectionerror",
        "apitimeouterror",
        "connecterror",
        "readtimeout",
        "timeoutexception",
    }:
        return True

    raw = str(exc).lower()
    return (
        "connection error" in raw
        or "timed out" in raw
        or "temporarily unavailable" in raw
    )


def _should_react_to_error(event: "AgentEvent") -> bool:
    return bool(event.channel_id) and bool(event.author)



def _skill_name_from_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return path.parent.name

    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return path.parent.name

    end_idx = -1
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx <= 1:
        return path.parent.name

    frontmatter = "\n".join(lines[1:end_idx])
    try:
        parsed = yaml.safe_load(frontmatter)
    except yaml.YAMLError:
        return path.parent.name
    if isinstance(parsed, dict):
        name = str(parsed.get("name", "")).strip()
        if name:
            return name
    return path.parent.name


def _git_sync(home: Path) -> str:
    git_dir = home / ".git"
    if not git_dir.exists():
        return "skip: not a git repo"
    try:
        add_proc = subprocess.run(
            ["git", "add", "-A"],
            cwd=home,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if add_proc.returncode != 0:
            return f"git add failed: {add_proc.stderr.strip()}"

        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=home,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if status_proc.returncode != 0:
            return f"git status failed: {status_proc.stderr.strip()}"
        if not status_proc.stdout.strip():
            # Working tree is clean — check for stranded commits from a prior push timeout.
            ahead_proc = subprocess.run(
                ["git", "rev-list", "@{u}..HEAD", "--count"],
                cwd=home,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if ahead_proc.returncode != 0 or ahead_proc.stdout.strip() in ("", "0"):
                return "clean: no changes"
            push_proc = subprocess.run(
                ["git", "push"],
                cwd=home,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            if push_proc.returncode != 0:
                return f"git push failed: {push_proc.stderr.strip()}"
            return "ok: pushed pending commits"

        commit_proc = subprocess.run(
            ["git", "commit", "-m", f"kynetic-agents auto-commit {utc_now_iso()}"],
            cwd=home,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if commit_proc.returncode != 0:
            return f"git commit failed: {commit_proc.stderr.strip()}"

        push_proc = subprocess.run(
            ["git", "push"],
            cwd=home,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if push_proc.returncode != 0:
            return f"git push failed: {push_proc.stderr.strip()}"

        return "ok: committed and pushed"
    except subprocess.TimeoutExpired as exc:
        cmd = exc.cmd[-1] if isinstance(exc.cmd, list) else str(exc.cmd)
        return f"git sync timeout: {cmd}"


def _extract_usage(result: dict[str, Any]) -> dict[str, int]:
    """Sum token usage across all AIMessage objects returned by agent.ainvoke.

    Checks usage_metadata (LangChain standard) first, then falls back to
    response_metadata["usage"] (older langchain-anthropic).
    """
    totals: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }
    for msg in (result.get("messages") or []):
        if not isinstance(msg, AIMessage):
            continue
        um = getattr(msg, "usage_metadata", None)
        if um:
            totals["input_tokens"]    += um.get("input_tokens",  0) or 0
            totals["output_tokens"]   += um.get("output_tokens", 0) or 0
            details = um.get("input_token_details") or {}
            totals["cache_read_tokens"]     += details.get("cache_read",     0) or 0
            totals["cache_creation_tokens"] += details.get("cache_creation", 0) or 0
        else:
            rm    = (getattr(msg, "response_metadata", None) or {})
            usage = rm.get("usage") or {}
            totals["input_tokens"]          += usage.get("input_tokens",                 0) or 0
            totals["output_tokens"]         += usage.get("output_tokens",                0) or 0
            totals["cache_read_tokens"]     += usage.get("cache_read_input_tokens",      0) or 0
            totals["cache_creation_tokens"] += usage.get("cache_creation_input_tokens",  0) or 0
    return {k: v for k, v in totals.items() if v > 0}


def _cleanup_old_sessions(sessions_dir: Path, retention_days: int) -> int:
    """Remove session log directories older than retention_days."""
    if not sessions_dir.exists():
        return 0
    cutoff = datetime.now(tz=UTC) - timedelta(days=retention_days)
    removed = 0
    for entry in sessions_dir.iterdir():
        if not entry.is_dir():
            continue
        # Session dirs are named like "20260226T013800Z-abcd1234".
        # Parse the timestamp prefix to determine age.
        name = entry.name
        timestamp_part = name.split("-")[0] if "-" in name else name
        try:
            dir_time = datetime.strptime(timestamp_part, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except ValueError:
            continue
        if dir_time < cutoff:
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    return removed


class OpenStrixApp(DiscordMixin, SchedulerMixin, ToolsMixin):
    def __init__(self, home: Path) -> None:
        self.home = home.resolve()
        self.layout = RepoLayout(home=self.home, state_dir_name=STATE_DIR_NAME)
        bootstrap_home_repo(self.layout, checkpoint_text=DEFAULT_CHECKPOINT)
        self.config = load_config(self.layout)
        if self.config.disable_builtin_skills:
            sync_builtin_skills_home(
                self.home, disabled_skills=self.config.disable_builtin_skills,
            )
        load_dotenv(dotenv_path=self.layout.env_file, override=False)
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
        self.tavily_search_url = os.getenv("TAVILY_SEARCH_URL", "").strip()
        self.web_search_enabled = bool(self.tavily_api_key)
        if not self.web_search_enabled:
            print(
                "[kynetic-agents] warning: TAVILY_API_KEY is not set; web_search tool is disabled.",
                flush=True,
            )

        self.queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self.scheduler = AsyncIOScheduler(timezone=UTC)
        self.pending_scheduler_keys: set[str] = set()
        self.current_channel_id: str | None = None
        self.current_event_label: str | None = None
        self.current_turn_start: float | None = None
        # Captured once run() starts the event loop; worker threads use this
        # to enqueue events from outside the loop (e.g. shell job waiters).
        self.loop: asyncio.AbstractEventLoop | None = None
        self.session_id = f"{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"

        self.message_history_all: deque[dict[str, Any]] = deque(maxlen=500)
        self.message_history_by_channel: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=250),
        )
        self._load_chat_history()
        # Session-scoped cache for fetched web content; cleaned up on shutdown.
        self.fetch_cache_dir = Path(tempfile.mkdtemp(prefix="fetch-cache-", dir=self.layout.logs_dir))
        self.shell_jobs = ShellJobRegistry(
            jobs_dir=self.layout.logs_dir / "shell-jobs",
        )
        self.hooks = HookManager(self)
        self.hooks.discover()

        self.discord_client: DiscordBridge | None = None
        self.api_runner: Any | None = None
        self.worker_task: asyncio.Task[Any] | None = None
        self._current_turn_sent_messages: list[tuple[str, str]] | None = None
        self.send_message_loop_soft_limit = SEND_MESSAGE_LOOP_SOFT_LIMIT
        self.send_message_loop_warn_limit = SEND_MESSAGE_LOOP_WARN_LIMIT
        self.send_message_loop_hard_limit = SEND_MESSAGE_LOOP_HARD_LIMIT
        self.send_message_loop_similarity_threshold = SEND_MESSAGE_LOOP_SIMILARITY_THRESHOLD
        self._send_message_last_text_normalized: str | None = None
        self._send_message_similarity_streak = 0
        self._send_message_circuit_breaker_active = False
        self._send_message_warning_reaction_sent = False
        self._last_turn_failure: str | None = None

        self.phone_book = load_phone_book(self.layout.phone_book_file)
        enrich_from_jsonl(
            self.phone_book, self.layout.people_jsonl, self.layout.channels_jsonl,
        )
        self.supervisor = Supervisor(self.layout.state_dir / "climbers")
        self._draining = False
        self.mcp_manager: MCPManager | None = None
        self._mempalace_write_queue: asyncio.Queue[MempalaceWriteItem] | None = None
        self._mempalace_session: Any | None = None
        # Tools/flags used to (re)build the main agent, stashed so per-job
        # scheduler agents (model overrides) can be built with the same toolset.
        self._agent_extra_tools: list[Any] = []
        self._agent_has_mempalace: bool = False
        self.agent = self._create_agent()

    def _load_chat_history(self) -> None:
        path = self.layout.chat_history_log
        if not path.exists():
            return

        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue

                record_type = str(record.get("type", "")).strip()
                if record_type == "message":
                    channel_id = str(record.get("channel_id", "")).strip()
                    author = str(record.get("author", "")).strip()
                    if not channel_id or not author:
                        continue
                    attachments = record.get("attachments")
                    self._remember_message(
                        channel_id=channel_id,
                        author=author,
                        content=str(record.get("content", "")),
                        attachment_names=[
                            str(item).strip()
                            for item in attachments
                            if str(item).strip()
                        ]
                        if isinstance(attachments, list)
                        else [],
                        message_id=str(record.get("message_id", "")).strip() or None,
                        is_bot=bool(record.get("is_bot")),
                        source=str(record.get("source", "discord")).strip() or "discord",
                        timestamp=str(record.get("timestamp", "")).strip() or None,
                        reactions=[
                            str(item).strip()
                            for item in record.get("reactions", [])
                            if str(item).strip()
                        ]
                        if isinstance(record.get("reactions"), list)
                        else [],
                        persist=False,
                    )
                    continue

                if record_type == "reaction":
                    channel_id = str(record.get("channel_id", "")).strip()
                    message_id = str(record.get("message_id", "")).strip()
                    emoji = str(record.get("emoji", "")).strip()
                    if not channel_id or not message_id or not emoji:
                        continue
                    self._apply_reaction_to_memory(
                        channel_id=channel_id,
                        message_id=message_id,
                        emoji=emoji,
                    )

    def _create_agent(
        self,
        extra_tools: list[Any] | None = None,
        has_mempalace: bool = False,
        model_override: str | None = None,
    ) -> Any:
        """Build the LangGraph agent with all tools.

        Args:
            extra_tools: Additional tools to register (e.g. MCP tools).
            has_mempalace: Whether mempalace tools are present.
            model_override: Provider-qualified model string to use instead of
                ``self.config.model``.  Scheduler jobs that set ``model:`` in
                ``scheduler.yaml`` use this to route low-cost tasks to a
                cheaper model class.
        """
        mutable_backend = LoggingWriteGuardBackend(
            root_dir=self.home,
            writable_dirs=self.config.writable_dirs,
            events_log_path=str(self.layout.events_log),
            session_id=self.session_id,
        )
        builtin_backend = build_builtin_skills_backend(root_dir=self.home / BUILTIN_HOME_DIRNAME)
        backend = CompositeBackend(
            default=mutable_backend,
            routes={BUILTIN_SKILLS_ROUTE: builtin_backend},
        )
        raw_model = model_override if model_override else self.config.model
        model_name = _model_for_deep_agents(raw_model)
        model = _build_chat_model(
            model_name,
            max_retries=self.config.model_max_retries,
            max_tokens=self.config.model_max_output_tokens,
            request_timeout_seconds=self.config.model_request_timeout_seconds,
        )
        skills_sources: list[str] = []
        if self.layout.skills_dir.exists():
            skills_sources.append("/skills")
        # Keep built-ins last so packaged defaults win on name collision.
        skills_sources.append(BUILTIN_SKILLS_ROUTE.rstrip("/"))
        skills = skills_sources or None
        self._log_loaded_skills(skills_sources)

        folders_text = render_folders_section(self.config.folders)
        system_prompt = SYSTEM_PROMPT
        if folders_text:
            system_prompt = system_prompt.replace(
                "Skills:", f"{folders_text}\n\nSkills:",
            )
        if has_mempalace:
            system_prompt = system_prompt + "\n\n" + MEMPALACE_SECTION

        tools = self._build_tools()
        if extra_tools:
            tools.extend(extra_tools)
        tools = self.hooks.wrap_tools(tools)

        subagents = self._build_subagents()
        subagents.extend(
            self._build_deep_thinker_subagents(
                model_name,
                max_tokens=self.config.model_max_output_tokens,
                request_timeout_seconds=self.config.model_request_timeout_seconds,
            )
        )

        return create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            backend=backend,
            skills=skills,
            subagents=subagents or None,
        )

    def _build_subagents(self) -> list[SubAgent]:
        """Build SubAgent specs from config.yaml subagents list."""
        if not self.config.subagents:
            return []

        specs: list[SubAgent] = []
        for cfg in self.config.subagents:
            spec: SubAgent = {
                "name": cfg.name,
                "description": cfg.description,
                "system_prompt": cfg.system_prompt or "You are a helpful assistant. Complete the task described below.",
            }
            if cfg.model:
                spec["model"] = _model_for_deep_agents(cfg.model)
            specs.append(spec)
        return specs

    def _build_deep_thinker_subagents(
        self,
        model_name: str,
        *,
        max_tokens: int,
        request_timeout_seconds: int,
    ) -> list[SubAgent]:
        """Build the opt-in deep-thinking subagents (thinking_enabled in config).

        Returns two SubAgent variants whose models have extended thinking
        enabled, differing only in DeepSeek reasoning effort: ``deep-thinker``
        (effort ``high``) and ``deep-thinker-max`` (effort ``max``). The main
        agent (which runs thinking-off) picks the effort per delegation by
        choosing the ``subagent_type`` — deepagents' ``task`` tool has no
        per-call effort argument, so variant selection is how effort is set
        "during the call". Returns an empty list when the flag is off.

        Each model is passed as a constructed instance (not a ``provider:model``
        string) so it can carry the ``thinking`` and ``output_config.effort``
        parameters; deepagents uses the instance directly. Tools are inherited
        from the main agent so the thinker can read files and run code on
        challenging coding problems.
        """
        if not self.config.thinking_enabled:
            return []
        system_prompt = (
            "You are a deep-reasoning specialist. Think rigorously and "
            "step by step about the problem before answering. For coding "
            "problems, reason about edge cases, complexity, and "
            "correctness, and inspect relevant files when useful. Return a "
            "clear, actionable final answer (plus the key reasoning that "
            "supports it) — not your entire scratch work."
        )
        base_description = (
            "Delegate here for problems that need careful, multi-step "
            "reasoning: tricky algorithm or systems design, debugging "
            "subtle logic, proofs, planning under ambiguity, or weighing "
            "non-obvious trade-offs. Use it for CHALLENGING coding "
            "problems; handle simple/routine coding inline yourself. "
            "Pass the full problem and relevant context; it returns a "
            "reasoned answer."
        )
        variants = [
            (
                "deep-thinker",
                "high",
                base_description + " Use this (effort: high) for most hard "
                "problems.",
            ),
            (
                "deep-thinker-max",
                "max",
                base_description + " Same as deep-thinker but with MAXIMUM "
                "reasoning effort — reserve it for the most demanding "
                "problems where deep-thinker's depth is not enough, as it is "
                "slower.",
            ),
        ]
        specs: list[SubAgent] = []
        for name, effort, description in variants:
            model = _build_chat_model(
                model_name,
                max_retries=self.config.model_max_retries,
                max_tokens=max_tokens,
                request_timeout_seconds=request_timeout_seconds,
                thinking=True,
                reasoning_effort=effort,
            )
            specs.append(
                {
                    "name": name,
                    "description": description,
                    "system_prompt": system_prompt,
                    "model": model,
                }
            )
        return specs

    def _skill_root_for_source(self, source: str) -> Path | None:
        if source == "/skills":
            return self.layout.skills_dir
        if source == BUILTIN_SKILLS_ROUTE.rstrip("/"):
            return self.home / BUILTIN_HOME_DIRNAME
        return None

    def _skills_for_source(self, source: str) -> list[tuple[str, str]]:
        root = self._skill_root_for_source(source)
        if root is None or not root.exists():
            return []

        rows: list[tuple[str, str]] = []
        for skill_file in sorted(root.rglob("SKILL.md")):
            rel_path = skill_file.relative_to(root).as_posix()
            virtual_path = f"{source}/{rel_path}"
            skill_name = _skill_name_from_file(skill_file)
            rows.append((skill_name, virtual_path))
        return rows

    def _log_loaded_skills(self, skills_sources: list[str]) -> None:
        print(
            f"[kynetic-agents] skills passed to deepagents: {skills_sources}",
            flush=True,
        )
        print("[kynetic-agents] discovered skill files:", flush=True)
        for source in skills_sources:
            rows = self._skills_for_source(source)
            if not rows:
                print(f"[kynetic-agents]   {source}: (no SKILL.md files)", flush=True)
                continue
            for skill_name, virtual_path in rows:
                print(f"[kynetic-agents]   {skill_name} -> {virtual_path}", flush=True)

    def log_event(self, event_type: str, **payload: Any) -> None:
        record = {
            "timestamp": utc_now_iso(),
            "type": event_type,
            "session_id": self.session_id,
            **payload,
        }
        _append_jsonl(self.layout.events_log, record)
        print(json.dumps(record, ensure_ascii=True, default=str), flush=True)

    def append_journal(
        self,
        user_wanted: str,
        agent_did: str,
        predictions: str,
        channel_id: str | None = None,
    ) -> None:
        entry = {
            "timestamp": utc_now_iso(),
            "session_id": self.session_id,
            "channel_id": channel_id if channel_id is not None else self.current_channel_id,
            "user_wanted": user_wanted,
            "agent_did": agent_did,
            "predictions": predictions,
        }
        _append_jsonl(self.layout.journal_log, entry)

    def should_respond_to_bot(self, author_id: str | int | None) -> bool:
        if author_id is None:
            return False
        return str(author_id) in self.config.always_respond_bot_ids

    def should_process_discord_message(
        self,
        *,
        author_is_bot: bool,
        author_id: str | int | None,
        channel_name: str = "",
        channel_id: str | int | None = None,
        mentions_bot: bool = False,
        author_addressed_bot: bool = False,
    ) -> bool:
        # First filter: ignore bots that aren't in the always-respond allowlist.
        is_always_respond = self.should_respond_to_bot(author_id)
        if author_is_bot and not is_always_respond:
            return False

        # Home-channels gate (only active when home_channels is configured).
        if self.config.home_channels:
            in_home_channel = (
                channel_name in self.config.home_channels
                or str(channel_id) in self.config.home_channels
            )
            if not (in_home_channel or mentions_bot or author_addressed_bot):
                return False

        return True

    def _iter_block_files(self) -> list[Path]:
        files = list(self.layout.blocks_dir.glob("*.yaml"))
        files.extend(self.layout.blocks_dir.glob("*.yml"))
        return sorted(files)

    def _load_memory_blocks(self) -> list[dict[str, Any]]:
        rows: list[tuple[int, str, dict[str, Any]]] = []
        for path in self._iter_block_files():
            try:
                loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue  # block deleted between glob and read (TOCTOU race)
            except yaml.YAMLError as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "Skipping corrupted block %s: %s", path.name, exc,
                )
                continue
            if not isinstance(loaded, dict):
                continue

            name = str(loaded.get("name", path.stem))
            text = str(loaded.get("text", ""))
            sort_raw = loaded.get("sort_order", loaded.get("sort", 0))
            try:
                sort_order = int(sort_raw)
            except (TypeError, ValueError):
                sort_order = 0

            block = {
                "id": path.stem,
                "name": name,
                "sort_order": sort_order,
                "text": text,
                "path": str(path.relative_to(self.home)),
            }
            rows.append((sort_order, name, block))

        rows.sort(key=lambda row: (row[0], row[1]))
        return [row[2] for row in rows]

    def _memory_block_path(self, block_id: str) -> Path:
        return self.layout.blocks_dir / f"{block_id}.yaml"

    def _find_memory_block_path(self, block_id: str) -> Path | None:
        candidates = [
            self.layout.blocks_dir / f"{block_id}.yaml",
            self.layout.blocks_dir / f"{block_id}.yml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _generate_block_id(self, preferred: str) -> str:
        block_id = _slugify(preferred)
        if self._find_memory_block_path(block_id) is None:
            return block_id
        idx = 2
        while self._find_memory_block_path(f"{block_id}-{idx}") is not None:
            idx += 1
        return f"{block_id}-{idx}"

    async def enqueue_event(self, event: AgentEvent) -> None:
        if event.dedupe_key:
            if event.dedupe_key in self.pending_scheduler_keys:
                self.log_event("event_deduped", key=event.dedupe_key)
                return
            self.pending_scheduler_keys.add(event.dedupe_key)

        await self.queue.put(event)
        self.log_event(
            "event_queued",
            source_event_type=event.event_type,
            channel_id=event.channel_id,
            scheduler_name=event.scheduler_name,
            queue_size=self.queue.qsize(),
            source_id=event.source_id,
        )

    def _handle_shell_job_complete(self, job: "ShellJob") -> None:
        """Thread-safe bridge: a shell job waiter thread invokes this when the
        subprocess exits. Schedules an async handler onto the main event loop
        so we can enqueue a shell_job_complete event.
        """
        if self.loop is None or self.loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._on_shell_job_complete(job),
                self.loop,
            )
        except Exception:
            # Never let a waiter-thread crash break the registry.
            pass

    async def _on_shell_job_complete(self, job: "ShellJob") -> None:
        """Enqueue a shell_job_complete event so the agent can react to a
        finished async shell job without re-hydration.
        """
        try:
            data = self.shell_jobs.read_output(
                job.job_id, tail_lines=100, stream="both"
            )
        except Exception:
            data = {"stdout_tail": "", "stderr_tail": ""}

        stdout_tail = (data.get("stdout_tail") or "").strip()
        stderr_tail = (data.get("stderr_tail") or "").strip()
        # Bound each stream so a runaway job doesn't produce a giant prompt.
        max_chars = 4000
        if len(stdout_tail) > max_chars:
            stdout_tail = stdout_tail[-max_chars:]
        if len(stderr_tail) > max_chars:
            stderr_tail = stderr_tail[-max_chars:]

        elapsed = round(job.elapsed_seconds, 1)
        status = job.status
        lines = [
            f"Shell job {job.job_id} complete (status={status}, exit_code={job.exit_code}, elapsed={elapsed}s).",
            f"Command: {job.command}",
            "",
            "--- stdout tail ---",
            stdout_tail or "(empty)",
            "",
            "--- stderr tail ---",
            stderr_tail or "(empty)",
        ]
        prompt = "\n".join(lines)

        event = AgentEvent(
            event_type="shell_job_complete",
            prompt=prompt,
            channel_id=job.channel_id,
            channel_name=job.channel_name,
            source_id=f"shell_job:{job.job_id}",
            dedupe_key=f"shell_job_complete:{job.job_id}",
        )
        try:
            await self.enqueue_event(event)
        except Exception as exc:
            self.log_event(
                "shell_job_complete_enqueue_failed",
                job_id=job.job_id,
                error=str(exc),
            )
        self.log_event(
            "shell_job_complete",
            job_id=job.job_id,
            command=job.command,
            exit_code=job.exit_code,
            elapsed=elapsed,
        )

    async def _run_post_turn_git_sync(self, event: AgentEvent) -> str:
        git_result = await asyncio.to_thread(_git_sync, self.home)
        self.log_event(
            "git_sync_after_turn",
            source_event_type=event.event_type,
            channel_id=event.channel_id,
            git_sync=git_result,
        )

        if "failed:" not in git_result:
            return git_result

        sent_messages = self._current_turn_sent_messages or []
        if not sent_messages:
            return git_result

        channel_id, message_id = sent_messages[-1]
        self.log_event(
            "warning",
            where="post_turn_git_sync",
            warning_type="git_sync_failed",
            git_sync=git_result,
            channel_id=channel_id,
            message_id=message_id,
        )
        await self._react_to_message(
            channel_id=channel_id,
            message_id=message_id,
            emoji=WARNING_REACTION_EMOJI,
        )
        return git_result

    def _render_prompt(self, event: AgentEvent) -> str:
        journal_entries = _tail_jsonl(
            self.layout.journal_log,
            self.config.journal_entries_in_prompt,
        )
        blocks = self._load_blocks_for_prompt()
        recent_candidates = [
            item
            for item in self.message_history_all
            if item.get("source") in {"discord", "web", "stdin"}
        ]
        if event.channel_id:
            channel_recent = [
                item
                for item in recent_candidates
                if item.get("channel_id") == event.channel_id
            ]
            if channel_recent:
                recent_candidates = channel_recent
        recent_messages = recent_candidates[-self.config.discord_messages_in_prompt :]

        return render_turn_prompt(
            journal_entries=journal_entries,
            memory_blocks=blocks,
            recent_messages=recent_messages,
            current_event={
                "event_type": event.event_type,
                "prompt": event.prompt,
                "channel_id": event.channel_id,
                "channel_name": event.channel_name,
                "channel_conversation_type": event.channel_conversation_type,
                "channel_visibility": event.channel_visibility,
                "author": event.author,
                "attachment_names": event.attachment_names,
                "scheduler_name": event.scheduler_name,
                "source_id": event.source_id,
                "source_platform": event.source_platform,
            },
            last_turn_failure=self._last_turn_failure,
            aliases_block=render_aliases_block(self.phone_book),
        )

    def _load_blocks_for_prompt(self) -> list[dict[str, Any]]:
        blocks = self._load_memory_blocks()
        return [
            {
                "id": block["id"],
                "name": block["name"],
                "sort_order": block["sort_order"],
                "text": block["text"],
            }
            for block in blocks
        ]

    async def _event_worker(self) -> None:
        while True:
            event = await self.queue.get()
            if self._draining:
                self.log_event("drain_skip_event", event_type=event.event_type)
                break
            self.current_channel_id = event.channel_id
            self.current_event_label = event.scheduler_name or event.event_type
            self.current_turn_start = time.monotonic()
            try:
                await self._process_event(event)
                self._last_turn_failure = None
            except SendMessageCircuitBreakerStop as exc:
                self._last_turn_failure = (
                    "Your previous turn was terminated by the send_message circuit breaker "
                    "(repeated near-duplicate messages). Before retrying, reflect on what "
                    "caused the loop. Consider using the five-whys skill to find the root "
                    "cause before attempting a different approach."
                )
                self.log_event(
                    "warning",
                    where="event_worker",
                    warning_type="send_message_loop_hard_stop",
                    source_event_type=event.event_type,
                    channel_id=event.channel_id,
                    error=str(exc),
                    **_error_log_fields(exc),
                )
            except Exception as exc:
                self._last_turn_failure = (
                    f"Your previous turn ended with an error: {type(exc).__name__}: {exc}. "
                    "Before retrying, reflect on what went wrong. If this is a recurring "
                    "failure, consider using the five-whys skill to find the structural cause."
                )
                reacted = False
                if _should_react_to_error(event):
                    reacted = await self._react_to_latest_message(
                        channel_id=event.channel_id,
                        emoji=ERROR_REACTION_EMOJI,
                        include_bot=False,
                    )
                error_message_sent = False
                self.log_event(
                    "error",
                    where="event_worker",
                    source_event_type=event.event_type,
                    error=str(exc),
                    reacted_to_last_user_message=reacted,
                    error_message_sent=error_message_sent,
                    **_error_log_fields(exc),
                )
            finally:
                if event.dedupe_key:
                    self.pending_scheduler_keys.discard(event.dedupe_key)
                if self._draining:
                    self.log_event("drain_complete", last_event=event.event_type)
                    break
                self.current_channel_id = None
                self.current_event_label = None
                self.current_turn_start = None
                self.queue.task_done()

    def _validate_memory_blocks(self) -> list[str]:
        """Check all block YAML files for parse errors. Returns list of error descriptions."""
        errors: list[str] = []
        for path in self._iter_block_files():
            try:
                loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            except yaml.YAMLError as exc:
                errors.append(f"{path.name}: {exc}")
                continue
            if not isinstance(loaded, dict):
                errors.append(f"{path.name}: expected a YAML mapping, got {type(loaded).__name__}")
        return errors

    def _get_agent_for_scheduler_model(self, model_name: str) -> Any:
        """Build a fresh agent for a per-job model override.

        A new agent is built on each scheduler firing — build cost is a few ms
        and is acceptable for jobs that fire no faster than once per minute.
        Caching is intentionally omitted: open-strix does not cache as a rule
        and the precedent cost of an idiom-of-one cache exceeds the build savings.

        The per-job agent reuses the same MCP tools and mempalace state as the
        main agent so a model override does not silently drop the toolset.
        """
        return self._create_agent(
            extra_tools=self._agent_extra_tools,
            has_mempalace=self._agent_has_mempalace,
            model_override=model_name,
        )

    def _is_mempalace_channel(self, channel_id: str) -> bool:
        channels = self.config.mempalace_channels
        if not channels:
            return False
        if channel_id in channels:
            return True
        entry = self.phone_book.entries.get(channel_id)
        return entry is not None and entry.name in channels

    async def _run_mempalace_writer(self) -> None:
        queue = self._mempalace_write_queue
        session = self._mempalace_session
        assert queue is not None and session is not None
        while True:
            item = await queue.get()
            try:
                wing = item.channel_name or item.channel_id
                room = item.timestamp[:10]  # YYYY-MM-DD — one room per day
                author_label = "bot" if item.is_bot else item.author
                content = f"[{item.timestamp}] {author_label}: {item.content}"
                await session.call_tool(
                    "mempalace_add_drawer",
                    {"wing": wing, "room": room, "content": content, "added_by": "kynetic-agents"},
                )
                self.log_event(
                    "mempalace_write",
                    channel_id=item.channel_id,
                    message_id=item.message_id,
                )
            except Exception as exc:
                self.log_event(
                    "mempalace_write_error",
                    channel_id=item.channel_id,
                    message_id=item.message_id,
                    error=str(exc),
                    **_error_log_fields(exc),
                )
            finally:
                queue.task_done()

    async def _process_event(self, event: AgentEvent) -> None:
        self._current_turn_sent_messages = []
        self._reset_send_message_circuit_breaker()
        # Turn-time instrumentation (#91): baseline measurement that lets the
        # eventual same-turn batching layer be evaluated against real numbers.
        turn_start = time.monotonic()
        timings: dict[str, float] = {
            "context_load_seconds": 0.0,
            "agent_invoke_seconds": 0.0,
            "block_validation_seconds": 0.0,
            "block_repair_invoke_seconds": 0.0,
            "git_sync_seconds": 0.0,
        }
        repair_invoke_count = 0

        prompt_start = time.monotonic()
        prompt = self._render_prompt(event)
        prompt_hook_event = await self.hooks.run_event(
            "pre_prompt",
            {
                "prompt": prompt,
                "source_event_type": event.event_type,
                "channel_id": event.channel_id,
                "channel_name": event.channel_name,
                "channel_conversation_type": event.channel_conversation_type,
                "channel_visibility": event.channel_visibility,
                "author": event.author,
                "attachment_names": event.attachment_names,
                "scheduler_name": event.scheduler_name,
                "source_id": event.source_id,
                "source_platform": event.source_platform,
            },
        )
        next_prompt = prompt_hook_event.get("prompt", prompt)
        if isinstance(next_prompt, str):
            prompt = next_prompt
        else:
            self.log_event(
                "hook_invalid_mutation",
                hook_event_type="pre_prompt",
                error="'prompt' must remain a string",
            )
        append_prompt = prompt_hook_event.get("append_prompt")
        if isinstance(append_prompt, str) and append_prompt:
            prompt = f"{prompt}\n\n{append_prompt}"
        timings["context_load_seconds"] = time.monotonic() - prompt_start
        self.log_event(
            "agent_invoke_start",
            source_event_type=event.event_type,
            channel_id=event.channel_id,
            scheduler_name=event.scheduler_name,
        )
        agent = (
            self._get_agent_for_scheduler_model(
                _model_for_deep_agents(event.scheduler_model)
            )
            if event.scheduler_model
            else self.agent
        )
        try:
            invoke_start = time.monotonic()
            async with self._typing_indicator(event):
                result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
            timings["agent_invoke_seconds"] = time.monotonic() - invoke_start
            self._log_agent_trace(result)
            _usage = _extract_usage(result)
            if _usage:
                self.log_event(
                    "llm_usage",
                    source_event_type=event.event_type,
                    scheduler_name=event.scheduler_name,
                    model=event.scheduler_model or self.config.model,
                    **_usage,
                )
            self._write_session_log(event, prompt, result)

            final_text = self._extract_final_text(result)
            sent_via_tool = bool(self._current_turn_sent_messages)
            if (
                not sent_via_tool
                and final_text.strip()
                and event.channel_id
                and event.scheduler_name is None
            ):
                await self._send_channel_message(
                    channel_id=event.channel_id,
                    text=final_text,
                )
                self.log_event(
                    "agent_final_message_fallback_sent",
                    source_event_type=event.event_type,
                    channel_id=event.channel_id,
                    final_text=final_text,
                )
            else:
                self.log_event(
                    "agent_final_message_discarded",
                    source_event_type=event.event_type,
                    channel_id=event.channel_id,
                    final_text=final_text,
                )

            tool_calls_in_turn = self._collect_tool_calls_in_turn(result)
            if final_text and "send_message" not in tool_calls_in_turn:
                self.log_event(
                    "agent_turn_missing_send_message",
                    source_event_type=event.event_type,
                    channel_id=event.channel_id,
                    final_text=final_text,
                    tool_calls_in_turn=tool_calls_in_turn,
                )

            # Post-turn hook: validate memory blocks and let agent self-correct
            validation_start = time.monotonic()
            block_errors = self._validate_memory_blocks()
            timings["block_validation_seconds"] = time.monotonic() - validation_start
            if block_errors:
                error_list = "\n".join(f"  - {e}" for e in block_errors)
                repair_prompt = (
                    "SYSTEM: Your turn just ended but some memory blocks have invalid YAML. "
                    "Fix them now using update_memory_block (which always produces valid YAML). "
                    "Do NOT use bash, write_file, or edit_file to fix block files.\n\n"
                    f"Broken blocks:\n{error_list}"
                )
                self.log_event(
                    "post_turn_block_validation_failed",
                    broken_blocks=[e.split(":")[0] for e in block_errors],
                    error_count=len(block_errors),
                )
                repair_start = time.monotonic()
                async with self._typing_indicator(event):
                    result = await agent.ainvoke(
                        {"messages": [HumanMessage(content=repair_prompt)]}
                    )
                timings["block_repair_invoke_seconds"] = time.monotonic() - repair_start
                repair_invoke_count = 1
                self._log_agent_trace(result)
                _repair_usage = _extract_usage(result)
                if _repair_usage:
                    self.log_event(
                        "llm_usage",
                        source_event_type="repair",
                        scheduler_name=event.scheduler_name,
                        model=event.scheduler_model or self.config.model,
                        **_repair_usage,
                    )
                # Check again — log but don't loop
                remaining_errors = self._validate_memory_blocks()
                if remaining_errors:
                    self.log_event(
                        "post_turn_block_validation_still_broken",
                        broken_blocks=[e.split(":")[0] for e in remaining_errors],
                    )

            git_start = time.monotonic()
            await self._run_post_turn_git_sync(event)
            timings["git_sync_seconds"] = time.monotonic() - git_start
        finally:
            self._reset_send_message_circuit_breaker()
            self._current_turn_sent_messages = None
            rounded = {key: round(value, 4) for key, value in timings.items()}
            self.log_event(
                "turn_timing",
                source_event_type=event.event_type,
                channel_id=event.channel_id,
                scheduler_name=event.scheduler_name,
                total_seconds=round(time.monotonic() - turn_start, 4),
                repair_invoke_count=repair_invoke_count,
                **rounded,
            )

    def _log_agent_trace(self, result: dict[str, Any]) -> None:
        messages = result.get("messages")
        if not isinstance(messages, list):
            return
        for message in messages:
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    self.log_event(
                        "tool_call",
                        tool=call.get("name"),
                        args=call.get("args"),
                    )

    def _collect_tool_calls_in_turn(self, result: dict[str, Any]) -> list[str]:
        messages = result.get("messages")
        if not isinstance(messages, list):
            return []
        names: list[str] = []
        for message in messages:
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    name = call.get("name")
                    if isinstance(name, str):
                        names.append(name)
        return names

    def _write_session_log(
        self,
        event: AgentEvent,
        prompt: str,
        result: dict[str, Any],
    ) -> None:
        session_dir = self.layout.sessions_dir / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        slug = re.sub(r"[^a-z0-9]+", "-", (event.event_type or "unknown").lower()).strip("-")
        filename = f"{timestamp}_{slug}.json"

        messages = result.get("messages")
        serialized_messages: list[dict[str, Any]] = []
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, BaseMessage):
                    try:
                        serialized_messages.append(msg.model_dump())
                    except Exception:
                        serialized_messages.append(
                            {"type": getattr(msg, "type", "unknown"), "content": str(getattr(msg, "content", ""))}
                        )
                elif isinstance(msg, dict):
                    serialized_messages.append(msg)

        record = {
            "session_id": self.session_id,
            "timestamp": utc_now_iso(),
            "event": {
                "event_type": event.event_type,
                "channel_id": event.channel_id,
                "channel_name": event.channel_name,
                "author": event.author,
                "scheduler_name": event.scheduler_name,
                "source_id": event.source_id,
            },
            "prompt": prompt,
            "messages": serialized_messages,
        }
        log_path = session_dir / filename
        try:
            log_path.write_text(
                json.dumps(record, ensure_ascii=True, default=str, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            self.log_event(
                "warning",
                where="write_session_log",
                warning_type="session_log_write_failed",
                error=str(exc),
            )

    def _extract_final_text(self, result: dict[str, Any]) -> str:
        messages = result.get("messages")
        if not isinstance(messages, list):
            return ""
        for message in reversed(messages):
            if not isinstance(message, AIMessage):
                continue
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts: list[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
                return "\n".join(text_parts).strip()
        return ""

    async def _stdin_mode(self) -> None:
        self.log_event("stdin_mode_start")
        print("No Discord token configured. Running in stdin mode.")
        while True:
            try:
                line = await asyncio.to_thread(input, "kynetic-agents> ")
            except EOFError:
                self.log_event("stdin_mode_eof")
                return
            prompt = line.strip()
            if not prompt:
                continue
            self._remember_message(
                channel_id="stdin",
                author="local_user",
                content=prompt,
                attachment_names=[],
                message_id=None,
                source="stdin",
            )
            await self.enqueue_event(
                AgentEvent(
                    event_type="stdin_message",
                    prompt=prompt,
                    channel_id="stdin",
                    author="local_user",
                ),
            )

    async def run(self) -> None:
        # Capture the running event loop for cross-thread scheduling
        # (e.g. shell job completion callbacks).
        self.loop = asyncio.get_running_loop()
        await self.hooks.run_event(
            "pre_startup",
            {"home": str(self.home)},
        )
        # Build effective MCP server list, prepending mempalace if configured.
        effective_mcp_servers = list(self.config.mcp_servers)
        if self.config.mempalace_path:
            mempalace_tools = MEMPALACE_READ_TOOLS + (MEMPALACE_WRITE_TOOLS if self.config.mempalace_writer else [])
            effective_mcp_servers = [
                MCPServerConfig(
                    name="mempalace",
                    command="python",
                    args=["-m", "mempalace.mcp_server", "--palace", self.config.mempalace_path],
                    allowed_tools=mempalace_tools,
                ),
                *effective_mcp_servers,
            ]

        if effective_mcp_servers:
            self.mcp_manager = MCPManager()
            mcp_tools = await self.mcp_manager.start_servers(
                effective_mcp_servers,
                log_fn=self.log_event,
            )
            if mcp_tools:
                self._agent_extra_tools = mcp_tools
                self._agent_has_mempalace = bool(self.config.mempalace_path)
                self.agent = self._create_agent(extra_tools=mcp_tools, has_mempalace=bool(self.config.mempalace_path))
                print(
                    f"[kynetic-agents] Agent recreated with {len(mcp_tools)} MCP tool(s)",
                    flush=True,
                )

        # Extract the raw mempalace session so the writer coroutine can call write tools.
        if self.config.mempalace_path and self.mcp_manager:
            for conn in self.mcp_manager.connections:
                if conn.config.name == "mempalace":
                    self._mempalace_session = conn.session
                    break

        # Start the singleton writer task only in the designated writer process.
        if self.config.mempalace_writer and self.config.mempalace_channels:
            if self._mempalace_session is not None:
                self._mempalace_write_queue = asyncio.Queue()
                asyncio.create_task(self._run_mempalace_writer())
            else:
                self.log_event(
                    "warning",
                    where="run",
                    warning_type="mempalace_writer_no_session",
                )

        self.worker_task = asyncio.create_task(self._event_worker())
        self._install_drain_handler()
        self.scheduler.start()
        self._reload_scheduler_jobs()
        self.supervisor.start_all()
        removed = _cleanup_old_sessions(
            self.layout.sessions_dir,
            self.config.session_log_retention_days,
        )
        self.log_event(
            "app_started",
            home=str(self.home),
            session_logs_cleaned=removed,
            mcp_servers=[c.config.name for c in (self.mcp_manager.connections if self.mcp_manager else [])],
            mempalace_writer=self.config.mempalace_writer and self._mempalace_write_queue is not None,
            mempalace_channels=self.config.mempalace_channels,
        )

        if self.config.api_port > 0:
            from .api import start_api

            self.api_runner = await start_api(self, self.config.api_port)

        await self.hooks.run_event(
            "post_startup",
            {
                "home": str(self.home),
                "api_port": self.config.api_port,
                "scheduler_running": self.scheduler.running,
            },
        )

        token = os.getenv(self.config.discord_token_env, "")
        if token:
            self.discord_client = DiscordBridge(self)
            self.log_event("discord_connecting")
            await self.discord_client.start(token)
            return

        if self.api_runner is not None:
            print(
                "No Discord token configured. API-only mode is active.",
                flush=True,
            )
            await asyncio.Event().wait()
            return

        await self._stdin_mode()

    def _install_drain_handler(self) -> None:
        """Register SIGQUIT to initiate graceful drain on Unix systems."""
        if not hasattr(signal, "SIGQUIT"):
            return
        loop = asyncio.get_running_loop()

        def _on_sigquit() -> None:
            self.log_event("drain_signal_received")
            print("[kynetic-agents] SIGQUIT received — draining after current turn", flush=True)
            self._draining = True
            # If the worker is idle (waiting on queue.get), unblock it
            # by pushing a sentinel; the drain check will skip + break.
            self.queue.put_nowait(AgentEvent(event_type="drain_sentinel", prompt="", channel_id=""))
            # Schedule shutdown after the worker has had time to finish
            loop.create_task(self._drain_then_stop())

        loop.add_signal_handler(signal.SIGQUIT, _on_sigquit)

    async def _drain_then_stop(self) -> None:
        """Wait for the event worker to finish its current turn, then stop."""
        if self.worker_task is not None:
            try:
                await asyncio.wait_for(self.worker_task, timeout=300)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        # Close Discord to unblock run()
        if self.discord_client is not None and not self.discord_client.is_closed():
            await self.discord_client.close()

    async def shutdown(self) -> None:
        self.log_event("app_shutdown_start")
        await self.hooks.run_event(
            "pre_shutdown",
            {"home": str(self.home)},
        )
        self.supervisor.stop_all()
        if self._mempalace_write_queue is not None:
            try:
                await asyncio.wait_for(self._mempalace_write_queue.join(), timeout=10.0)
            except asyncio.TimeoutError:
                self.log_event("warning", where="shutdown", warning_type="mempalace_drain_timeout")
        if self.mcp_manager is not None:
            await self.mcp_manager.shutdown()
        if self.api_runner is not None:
            await self.api_runner.cleanup()
        if self.discord_client is not None and not self.discord_client.is_closed():
            await self.discord_client.close()
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        if self.worker_task is not None:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        if self.fetch_cache_dir.exists():
            shutil.rmtree(self.fetch_cache_dir, ignore_errors=True)
        await self.hooks.run_event(
            "post_shutdown",
            {"home": str(self.home)},
        )
        self.log_event("app_shutdown_complete")


def run_kynetic_agents(home: Path | None = None) -> None:
    app = OpenStrixApp(home=home or Path.cwd())

    async def _runner() -> None:
        try:
            await app.run()
        finally:
            await app.shutdown()

    try:
        asyncio.run(_runner())
    except KeyboardInterrupt:
        pass
