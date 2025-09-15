from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def test_sdd_common_self_test_exits_when_log_json_fails() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "lib" / "sdd-common.sh"

    script_command = "\n".join(
        [
            "log_json() { return 17; }",
            "export -f log_json",
            f"bash {shlex.quote(str(script_path))}",
        ]
    )

    result = subprocess.run(
        ["bash", "-c", script_command],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 17
    assert "log_json self-test failed" in result.stderr
