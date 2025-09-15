"""Integration test for collect_validation_messages.sh."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "collect_validation_messages.sh"


def test_collect_validation_messages_outputs_distinct_entries() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["count"] >= 3
    messages = payload["messages"]
    assert len(messages) == payload["count"]

    texts = [entry["message"] for entry in messages]
    assert len(set(texts)) == len(texts)

    statuses = {entry["status"] for entry in messages}
    assert statuses.issubset({"info", "warning", "error"})
