"""Tests for syncing and integrating the GitHub spec-kit repository."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import spec_kit


class _DummyResult:
    """Simple stand-in for subprocess.CompletedProcess."""

    def __init__(self) -> None:
        self.stdout = ""
        self.stderr = ""
        self.returncode = 0


def _install_fake_run(monkeypatch: pytest.MonkeyPatch, bucket: list[list[str]]) -> None:
    """Install a fake subprocess.run that records commands."""

    def _fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> _DummyResult:
        bucket.append(cmd)
        return _DummyResult()

    monkeypatch.setattr(spec_kit.subprocess, "run", _fake_run)


def test_sync_clones_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A missing repo should trigger a git clone."""

    commands: list[list[str]] = []
    _install_fake_run(monkeypatch, commands)

    architect = spec_kit.SpecKitArchitect(workspace_root=tmp_path)
    repo_path = architect.sync_github_spec_kit("https://example.com/spec-kit.git")

    assert repo_path == tmp_path / "spec-kit"
    assert commands == [["git", "clone", "https://example.com/spec-kit.git", str(repo_path)]]


def test_sync_updates_existing_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An existing repo should fetch and reset to origin/main."""

    commands: list[list[str]] = []
    _install_fake_run(monkeypatch, commands)

    repo_path = tmp_path / "spec-kit"
    (repo_path / ".git").mkdir(parents=True)

    architect = spec_kit.SpecKitArchitect(workspace_root=tmp_path)
    result_path = architect.sync_github_spec_kit()

    assert result_path == repo_path
    assert commands == [
        ["git", "-C", str(repo_path), "fetch", "--all"],
        ["git", "-C", str(repo_path), "reset", "--hard", "origin/main"],
    ]


def test_integrate_copies_templates(tmp_path: Path) -> None:
    """Templates are copied into the configured destination."""

    repo_path = tmp_path / "spec-kit"
    templates_source = repo_path / "templates"
    templates_source.mkdir(parents=True)
    (templates_source / "spec-template.md").write_text("spec template", encoding="utf-8")

    architect = spec_kit.SpecKitArchitect(workspace_root=tmp_path)
    destination = architect.integrate_github_templates(repo_path)

    expected_file = destination / "spec-template.md"
    assert expected_file.is_file()
    assert expected_file.read_text(encoding="utf-8") == "spec template"

