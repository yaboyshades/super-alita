import json
import subprocess
from typing import Any

from tests.runtime.fakes import fake_grpc_server


def _run_client(method: str, payload: Any | None = None) -> dict[str, Any]:
    arg = json.dumps(payload) if payload is not None else ""
    script = f"""
import('./extensions/copilot-agent/src/grpcClient.ts').then(async (c) => {{
  try {{
    const res = await c.{method}({arg});
    process.stdout.write(JSON.stringify({{'ok': true, 'res': res}}));
  }} catch (e) {{
    process.stdout.write(JSON.stringify({{'ok': false, 'error': String(e.message || e)}}));
  }}
}});
"""
    result = subprocess.run(
        [
            "node",
            "--no-warnings",
            "--import",
            "./extensions/copilot-agent/node_modules/tsx/dist/loader.mjs",
            "-e",
            script,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout, result.stderr
    return json.loads(result.stdout.strip())


def test_get_health_success() -> None:
    with fake_grpc_server():
        out = _run_client("getHealth")
    assert out["ok"] is True
    assert out["res"]["message"] == "ok"


def test_get_health_error() -> None:
    with fake_grpc_server(["health"]):
        out = _run_client("getHealth")
    assert out["ok"] is False
    assert "health fail" in out["error"]


def test_get_status_success() -> None:
    with fake_grpc_server():
        out = _run_client("getStatus")
    assert out["ok"] is True
    assert out["res"]["version"] == "1.0"


def test_get_status_error() -> None:
    with fake_grpc_server(["status"]):
        out = _run_client("getStatus")
    assert out["ok"] is False
    assert "status fail" in out["error"]


def _task_payload() -> dict[str, Any]:
    return {
        "task_id": "t1",
        "content": "do",
        "session_id": "s1",
        "user_id": "u1",
        "workspace": "w1",
        "metadata": {},
    }


def test_process_task_success() -> None:
    with fake_grpc_server():
        out = _run_client("processTask", _task_payload())
    assert out["ok"] is True
    assert out["res"]["task_id"] == "t1"


def test_process_task_error() -> None:
    with fake_grpc_server(["process"]):
        out = _run_client("processTask", _task_payload())
    assert out["ok"] is False
    assert "process fail" in out["error"]


def test_kg_query_success() -> None:
    with fake_grpc_server():
        out = _run_client("kgQuery", {"query": "q", "limit": 1})
    assert out["ok"] is True
    assert out["res"]["total_results"] == 0


def test_kg_query_error() -> None:
    with fake_grpc_server(["kg"]):
        out = _run_client("kgQuery", {"query": "q", "limit": 1})
    assert out["ok"] is False
    assert "kg fail" in out["error"]
