"""Tests for the constitutional-gates shell utility."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "lib" / "constitutional-gates.sh"
)


def run_script(tmp_path: Path, spec_text: str, plan_text: str) -> dict[str, object]:
    """Write artifacts to disk and execute the gate script."""
    spec_file = tmp_path / "spec.md"
    plan_file = tmp_path / "plan.md"
    spec_file.write_text(spec_text)
    plan_file.write_text(plan_text)

    completed = subprocess.run(
        [str(SCRIPT_PATH), "--spec", str(spec_file), "--plan", str(plan_file)],
        capture_output=True,
        text=True,
        check=True,
    )

    return json.loads(completed.stdout)


def test_reports_missing_sections(tmp_path: Path) -> None:
    """The script should surface missing Feature ID and DoD requirements."""
    result = run_script(tmp_path, "## Overview\n", "## Plan Outline\n")

    assert result == {
        "ok": False,
        "messages": [
            "Spec missing Feature ID (Article II)",
            "Plan missing DoD (Article XV)",
        ],
    }


def test_reports_success_when_requirements_present(tmp_path: Path) -> None:
    """When all sections are present the script should return ok=true."""
    spec_content = """Feature ID: feat-123\nSummary of capability."""
    plan_content = """Definition of Done:\n- Tests validated\n"""

    result = run_script(tmp_path, spec_content, plan_content)

    assert result == {"ok": True, "messages": []}


def test_fails_with_clear_error_when_jq_missing(tmp_path: Path) -> None:
    """The script should emit a helpful error when jq is unavailable."""
    spec_file = tmp_path / "spec.md"
    plan_file = tmp_path / "plan.md"
    spec_file.write_text("## Spec content\n")
    plan_file.write_text("## Plan content\n")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    for command_name in ("bash", "cat", "grep"):
        command_path = shutil.which(command_name)
        assert command_path is not None, f"{command_name} command not found"
        (bin_dir / command_name).symlink_to(Path(command_path))

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)

    completed = subprocess.run(
        [
            str(SCRIPT_PATH),
            "--spec",
            str(spec_file),
            "--plan",
            str(plan_file),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert completed.returncode != 0
    stderr_lower = completed.stderr.lower()
    assert "jq" in stderr_lower
    assert "install" in stderr_lower
