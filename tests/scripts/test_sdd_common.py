"""Integration tests for SDD shell helpers."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path


def test_log_json_reuses_cached_python_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "lib" / "sdd-common.sh"
    command = (
        "set -euo pipefail; "
        "PS4='+TRACE:'; "
        "set -x; "
        f"source {shlex.quote(str(script_path))}; "
        "log_json info 'first message'; "
        "log_json info 'second message'"
    )

    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        check=True,
    )

    output_lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(output_lines) == 2

    parsed = [json.loads(line) for line in output_lines]
    assert [entry["message"] for entry in parsed] == ["first message", "second message"]
    assert all(entry["level"] == "info" for entry in parsed)

    trace_lines: list[str] = []
    for raw_line in result.stderr.splitlines():
        stripped = raw_line.lstrip("+")
        if stripped.startswith("TRACE:"):
            trace_lines.append(stripped[len("TRACE:") :])

    first_log_index: int | None = None
    for index, line in enumerate(trace_lines):
        if "log_json info" in line:
            first_log_index = index
            break

    assert first_log_index is not None, "log_json invocation was not traced"

    detection_after_first_log = any(
        "command -v python" in line for line in trace_lines[first_log_index + 1 :]
    )
    assert (
        not detection_after_first_log
    ), "python interpreter lookup should not run after the first log_json call"

    detection_before_first_log = [
        line
        for line in trace_lines[: first_log_index + 1]
        if "command -v python" in line
    ]
    assert detection_before_first_log, "expected at least one interpreter detection"
