"""Verify every @tool()-decorated function in _build_tools is returned in the tools list.

Catches the bug where a tool is defined but never added to the list that
_build_tools() returns, making it invisible to agents at runtime.
"""

from __future__ import annotations

import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

import kynetic_agents.app as app_mod
import kynetic_agents.tools as tools_mod
from kynetic_agents.config import AppConfig, RepoLayout, load_config


TOOLS_PY = Path(__file__).resolve().parent.parent / "kynetic_agents" / "tools.py"


def test_all_tools_registered():
    """Every @tool() function inside _build_tools must appear in the returned list."""
    source = TOOLS_PY.read_text()

    # Extract function names defined with @tool("...") decorator.
    # Pattern: @tool("name") followed by [async] def func_name(
    tool_funcs: set[str] = set()
    for m in re.finditer(
        r'@tool\([^)]+\)\s+(?:async\s+)?def\s+(\w+)\s*\(', source
    ):
        tool_funcs.add(m.group(1))

    assert tool_funcs, "Found no @tool() definitions — regex may need updating"

    # Extract variable names from the `tools: list[Any] = [...]` block.
    list_match = re.search(
        r'tools:\s*list\[Any\]\s*=\s*\[(.*?)\]', source, re.DOTALL
    )
    assert list_match, "Could not find `tools: list[Any] = [...]` in tools.py"
    listed = set(re.findall(r'\b(\w+)\b', list_match.group(1)))

    # Also pick up tools added conditionally via tools.insert(...) or tools.append(...)
    for m in re.finditer(r'tools\.(?:insert|append)\([^,]*,?\s*(\w+)', source):
        listed.add(m.group(1))

    # These tools are wired through deepagents' FilesystemBackend (LoggingWriteGuardBackend)
    # rather than the returned tools list — they are intentionally excluded.
    backend_provided = {"read_file", "write_file", "edit_file", "glob_files"}

    missing = tool_funcs - listed - backend_provided
    assert not missing, (
        f"Tools defined with @tool() but missing from the returned list: {missing}. "
        f"Add them to the `tools` list in _build_tools()."
    )


def test_run_shell_replaces_invalid_utf8_output() -> None:
    if os.name == "nt":
        command = "$stdout = [Console]::OpenStandardOutput(); $stdout.WriteByte(0x96)"
    else:
        command = "printf '\\226'"

    result = tools_mod._run_shell(command, timeout_seconds=5)

    assert result.returncode == 0
    assert result.stdout == "\ufffd"


# ---------------------------------------------------------------------------
# fetch_url_max_bytes config wiring
# ---------------------------------------------------------------------------

def _make_app(tmp_path: Path, config_yaml: str) -> app_mod.OpenStrixApp:
    (tmp_path / "config.yaml").write_text(config_yaml, encoding="utf-8")
    with patch.object(app_mod, "create_deep_agent", return_value=object()):
        return app_mod.OpenStrixApp(tmp_path)


def test_fetch_url_max_bytes_default_is_two_mb() -> None:
    assert AppConfig().fetch_url_max_bytes == 2_000_000


def test_load_config_parses_fetch_url_max_bytes(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"model": "test", "fetch_url_max_bytes": 10_000_000}),
        encoding="utf-8",
    )
    layout = RepoLayout(home=tmp_path, state_dir_name="state")
    config = load_config(layout)
    assert config.fetch_url_max_bytes == 10_000_000


def test_load_config_defaults_fetch_url_max_bytes_when_missing(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"model": "test"}),
        encoding="utf-8",
    )
    layout = RepoLayout(home=tmp_path, state_dir_name="state")
    config = load_config(layout)
    assert config.fetch_url_max_bytes == 2_000_000


@pytest.mark.asyncio
async def test_fetch_url_tool_default_honours_config_max_bytes(tmp_path: Path) -> None:
    """fetch_url must use config.fetch_url_max_bytes as the default, not a hardcoded constant."""
    app = _make_app(tmp_path, "model: test\nfetch_url_max_bytes: 500\n")
    body = b"x" * 600  # exceeds the 500-byte config limit

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: Any) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        tools = {t.name: t for t in app._build_tools()}
        result = await tools["fetch_url"].ainvoke(
            {"url": f"http://127.0.0.1:{server.server_port}/big.bin"},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert "max_bytes" in result or "exceeded" in result
