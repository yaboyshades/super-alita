"""Tests for the constitutional-gates shell utility."""

from __future__ import annotations

import json
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
