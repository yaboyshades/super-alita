"""CLI integration tests for startup modes."""

import json
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_cli_no_chat_exits_with_health():
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "--no-chat"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout.strip())
    assert data["event_bus"]
    assert data["ability_registry"]
    assert data["kg"]
    assert data["llm_model"]


def test_cli_runs_uvicorn():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.main", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(40):
            try:
                resp = urlopen(f"http://127.0.0.1:{port}/healthz")
                if resp.status == 200:
                    break
            except Exception:
                time.sleep(0.25)
        else:
            raise AssertionError("server did not start")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    assert proc.returncode is not None
