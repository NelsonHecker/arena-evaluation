from __future__ import annotations

import pathlib
import subprocess

import pytest

pytest.importorskip("rclpy")

from arena_evaluation.benchmark.state import capture_git_sha


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _workspace(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """A colcon workspace whose install dir sits beside src/Arena, a git checkout with one commit."""
    repo = tmp_path / "ws" / "src" / "Arena"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    (repo / "README.md").write_text("arena\n")
    _git(repo, "add", "README.md")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init")
    install = tmp_path / "ws" / "install"
    install.mkdir()
    return install, repo


def test_capture_git_sha_finds_the_arena_checkout_from_the_install_dir(tmp_path: pathlib.Path) -> None:
    install, repo = _workspace(tmp_path)
    assert capture_git_sha(install) == (_git(repo, "rev-parse", "HEAD"), False)


def test_capture_git_sha_reports_a_dirty_checkout(tmp_path: pathlib.Path) -> None:
    install, repo = _workspace(tmp_path)
    (repo / "README.md").write_text("changed\n")
    assert capture_git_sha(install) == (_git(repo, "rev-parse", "HEAD"), True)
