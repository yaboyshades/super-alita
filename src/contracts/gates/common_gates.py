from __future__ import annotations

import re
from re import Pattern
from typing import Any

from .base_gate import Gate

RE_EVAL = re.compile(r"\beval\s*\(")
RE_OS_SYSTEM = re.compile(r"\bos\.system\s*\(")
RE_SHELL_TRUE = re.compile(r"subprocess\.(?:Popen|run|call)\(.*shell\s*=\s*True", re.S)


class SafetyGate(Gate):
    """Static safety scan + server secure_scan_code."""

    def __init__(self, api_client):
        self.api = api_client

    def validate_latest(self, latest: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        diffs = list(latest.get("diffs") or [])
        reasons: list[str] = []
        for d in diffs:
            path = d.get("path", "")
            code = d.get("new_content") or ""
            if path.endswith((".py", ".md", ".json", ".yml", ".yaml", ".rst", ".txt")):
                if RE_EVAL.search(code):
                    reasons.append(f"{path}: eval() detected")
                if RE_OS_SYSTEM.search(code):
                    reasons.append(f"{path}: os.system() detected")
                if RE_SHELL_TRUE.search(code):
                    reasons.append(f"{path}: subprocess(..., shell=True) detected")
                res = self.api.secure_scan_code(code)
                for it in res.get("result", {}).get("issues", []):
                    reasons.append(
                        f"{path}: {it.get('severity')} - {it.get('message')}"
                    )
    return (not reasons, {"reasons": reasons})


class RequiredPathsGate(Gate):
    """Ensure artifacts exist before applying."""

    def __init__(
        self,
        required_paths: list[Pattern],
        required_docs: list[Pattern] | None = None,
    ):
        self.required_paths = required_paths
        self.required_docs = required_docs or []

    def validate_latest(self, latest: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        diffs = list(latest.get("diffs") or [])
        paths = [d.get("path") for d in diffs if d.get("path")]
        reasons: list[str] = []
        if self.required_paths and not any(
            rx.search(p) for p in paths for rx in self.required_paths
        ):
            reasons.append("required implementation/tests not found")
        if self.required_docs and not any(
            rx.search(p) for p in paths for rx in self.required_docs
        ):
            reasons.append("required docs not found")
    return (not reasons, {"paths": paths, "reasons": reasons})


class PytestGate(Gate):
    """Run pytest; pass only if green."""

    def __init__(self, api_client, args: list[str] | None = None):
        self.api = api_client
        self.args = args or ["-q"]

    def validate_latest(self, latest: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        # We could analyze latest diffs to determine which tests to run,
        # but for now we run all tests as configured
        _ = latest  # Acknowledged unused for now
        res = self.api.pytest_run(self.args)
        s = str(res).lower()
        ok = ("error" not in s) and ("failed" not in s)
        return ok, {"pytest_result": res}


class CombinedGate(Gate):
    """Compose multiple gates."""

    def __init__(self, *gates: Gate):
        self.gates = list(gates)

    def validate_latest(self, latest: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        all_ok = True
        info: dict[str, Any] = {"reasons": []}
        paths = None
        for g in self.gates:
            ok, details = g.validate_latest(latest)
            all_ok = all_ok and ok
            if "reasons" in details:
                info["reasons"].extend(details["reasons"])
            if "paths" in details and not paths:
                paths = details["paths"]
            for k, v in details.items():
                if k not in ("reasons", "paths"):
                    info[k] = v
        if paths:
            info["paths"] = paths
        return all_ok, info
