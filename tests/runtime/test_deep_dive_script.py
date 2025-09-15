from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def test_deep_dive_script_creates_summary(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Demo\n", encoding="utf-8")

    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-m", "init")

    script = Path(__file__).resolve().parents[2] / "tools" / "deep-dive.sh"

    env = os.environ.copy()
    env["DEEP_DIVE_SKIP_INSTALLS"] = "1"
    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = repo / "docs" / "deep-dive" / "summary.txt"
    assert summary.exists()
    assert "Deep Dive" in summary.read_text(encoding="utf-8")
