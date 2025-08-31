from __future__ import annotations

from typing import Any

import requests

DEFAULT_BASE = "http://127.0.0.1:8080"


class LocalAPI:
    """Thin client for local HTTP endpoints (DeepCode, pytest, secure scan)."""

    def __init__(self, base: str = DEFAULT_BASE, timeout: int = 60):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def deepcode_request(
        self,
        task_kind: str,
        requirements: str,
        repo_path: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        r = requests.post(
            f"{self.base}/deepcode/request",
            json={
                "task_kind": task_kind,
                "requirements": requirements,
                "repo_path": repo_path,
                "conversation_id": conversation_id,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def deepcode_latest(self) -> dict[str, Any]:
        r = requests.get(f"{self.base}/deepcode/latest", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def deepcode_apply(self, paths: list[str] | None = None) -> dict[str, Any]:
        r = requests.post(
            f"{self.base}/deepcode/apply",
            json={"paths": paths},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def pytest_run(self, args: list[str] | None = None) -> dict[str, Any]:
        r = requests.post(
            f"{self.base}/tools/pytest_run",
            json={"args": args or ["-q"]},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def secure_scan_code(self, code: str) -> dict[str, Any]:
        r = requests.post(
            f"{self.base}/ability/execute/secure_scan_code",
            json={"code": code},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
