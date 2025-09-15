from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest  # type: ignore[import-not-found]

SCRIPT_PATH = (
    Path("extensions/alita-language-tools") / "scripts" / "check-task-prerequisites.sh"
)
DEFAULT_BRANCH = "001-test-feature"
EXPECTED_MISSING_JSON = '{"ok":false,"missing":["feature-spec.md","plan.md"]}'
EXPECTED_OK_JSON = '{"ok":true,"missing":[]}'


def _run_git_command(repo_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )


def _write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _init_feature_repo(
    tmp_path: Path, branch: str = DEFAULT_BRANCH
) -> tuple[Path, Path]:
    repo_dir = tmp_path / "feature-repo"
    repo_dir.mkdir()

    _run_git_command(repo_dir, "init")
    _run_git_command(repo_dir, "config", "user.email", "test@example.com")
    _run_git_command(repo_dir, "config", "user.name", "Test User")
    _run_git_command(repo_dir, "commit", "--allow-empty", "-m", "init")
    _run_git_command(repo_dir, "checkout", "-b", branch)

    feature_dir = repo_dir / "specs" / branch
    feature_dir.mkdir(parents=True)
    return repo_dir, feature_dir


def _run_script(repo_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    if shutil.which("jq") is None:
        pytest.skip("jq is required to run check-task-prerequisites.sh")

    env = os.environ.copy() | {
        "GIT_DIR": str(repo_dir / ".git"),
        "GIT_WORK_TREE": str(repo_dir),
    }
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *args],
        cwd=SCRIPT_PATH.parent,
        env=env,
        capture_output=True,
        text=True,
    )


def test_reports_missing_required_documents(tmp_path: Path) -> None:
    repo_dir, _ = _init_feature_repo(tmp_path)

    result = _run_script(repo_dir, "--json")

    assert result.returncode == 1
    assert result.stdout.strip() == EXPECTED_MISSING_JSON


def test_succeeds_when_required_documents_present(tmp_path: Path) -> None:
    repo_dir, feature_dir = _init_feature_repo(tmp_path)
    _write_file(feature_dir / "feature-spec.md", "# Feature spec\n")
    _write_file(feature_dir / "plan.md", "# Plan\n")

    result = _run_script(repo_dir, "--json")

    assert result.returncode == 0
    assert result.stdout.strip() == EXPECTED_OK_JSON


def test_supports_legacy_file_names(tmp_path: Path) -> None:
    repo_dir, feature_dir = _init_feature_repo(tmp_path)
    _write_file(feature_dir / "spec.md", "# Legacy spec\n")
    _write_file(feature_dir / "implementation-plan.md", "# Legacy plan\n")

    result = _run_script(repo_dir, "--json")

    assert result.returncode == 0
    assert result.stdout.strip() == EXPECTED_OK_JSON
