"""Tests for _git_sync: post-turn commit-and-push behaviour.

Intent under test:
  1. Nothing happens when the working tree is clean and there is nothing to push.
  2. Uncommitted changes are committed and land in the remote on the same call.
  3. A push that times out does not block — the function returns an error and
     the event worker can continue processing messages.
  4. A commit stranded by a prior push failure is pushed on the next successful call.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kynetic_agents.app import _git_sync


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)


def _init_repo(tmp_path: Path) -> Path:
    """Bare repo with one commit and no remote — nothing to push."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["git", "init", "-b", "main"], repo)
    _git(["git", "config", "user.email", "test@kynetic.ai"], repo)
    _git(["git", "config", "user.name", "Test"], repo)
    (repo / "README.md").write_text("init")
    _git(["git", "add", "-A"], repo)
    _git(["git", "commit", "-m", "init"], repo)
    return repo


def _init_repo_with_remote(tmp_path: Path) -> tuple[Path, Path]:
    """Local repo + bare remote, initial commit already pushed.

    Returns (local, remote).
    """
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", "-b", "main", str(remote)],
        capture_output=True,
        check=True,
    )
    local = tmp_path / "local"
    local.mkdir()
    _git(["git", "init", "-b", "main"], local)
    _git(["git", "config", "user.email", "test@kynetic.ai"], local)
    _git(["git", "config", "user.name", "Test"], local)
    _git(["git", "remote", "add", "origin", str(remote)], local)
    (local / "README.md").write_text("init")
    _git(["git", "add", "-A"], local)
    _git(["git", "commit", "-m", "init"], local)
    _git(["git", "push", "-u", "origin", "main"], local)
    return local, remote


def _commits_in(repo: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "log", "--format=%s"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip().splitlines()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGitSync:
    def test_nothing_to_do_when_tree_is_clean_and_nothing_to_push(
        self, tmp_path: Path
    ) -> None:
        """Clean state: no error, no commit created."""
        local, remote = _init_repo_with_remote(tmp_path)
        before = _commits_in(remote)

        _git_sync(local)

        assert _commits_in(remote) == before

    def test_uncommitted_changes_reach_the_remote(self, tmp_path: Path) -> None:
        """New file is committed and visible in the remote after one call."""
        local, remote = _init_repo_with_remote(tmp_path)
        (local / "notes.txt").write_text("hello")

        _git_sync(local)

        remote_commits = _commits_in(remote)
        assert any("kynetic-agents auto-commit" in c for c in remote_commits)

    def test_push_timeout_does_not_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hanging push is not fatal — the function returns an error and moves on."""
        local, _ = _init_repo_with_remote(tmp_path)
        (local / "notes.txt").write_text("hello")

        original_run = subprocess.run

        def _run_with_slow_push(cmd, **kwargs):
            if isinstance(cmd, list) and cmd[:2] == ["git", "push"]:
                raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 30))
            return original_run(cmd, **kwargs)

        monkeypatch.setattr(subprocess, "run", _run_with_slow_push)

        result = _git_sync(local)

        assert "timeout" in result

    def test_stranded_commit_is_pushed_on_next_call(self, tmp_path: Path) -> None:
        """A commit that was not pushed (simulating a prior timeout) is pushed on the
        next call, even though the working tree is clean at that point."""
        local, remote = _init_repo_with_remote(tmp_path)

        # Simulate the state left behind by a push timeout: commit exists locally
        # but was never pushed.
        (local / "notes.txt").write_text("stranded")
        _git(["git", "add", "-A"], local)
        _git(["git", "commit", "-m", "stranded work"], local)

        # Sanity check: working tree is clean but remote doesn't have the commit yet.
        assert "stranded work" not in "\n".join(_commits_in(remote))

        _git_sync(local)

        assert "stranded work" in "\n".join(_commits_in(remote))
