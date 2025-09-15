import json
import shlex
import subprocess
from pathlib import Path


def _run_log_json(*pairs: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    command = "source scripts/lib/sdd-common.sh && log_json " + " ".join(
        shlex.quote(pair) for pair in pairs
    )
    return subprocess.run(
        ["bash", "-c", command],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )


def test_log_json_warns_when_message_key_is_present() -> None:
    result = _run_log_json(
        "action=create",
        "message=hello world",
        "status=ok",
    )

    payload = json.loads(result.stdout)
    assert payload == {"action": "create", "status": "ok"}

    stderr = result.stderr.strip()
    assert "WARN: log_json" in stderr
    assert "message" in stderr
