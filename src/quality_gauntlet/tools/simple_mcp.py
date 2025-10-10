"""Simple MCP-compatible tool wrappers used in the MVP pipeline."""

from __future__ import annotations

import json
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path

from . import CommandResult
from ._runner import DEFAULT_RUNNER
from ..schemas import ToolResult


class SimpleMCPTool(ABC):
    """Base class for simple MCP tool wrappers."""

    def __init__(self, name: str, timeout: int = 120) -> None:
        self._name = name
        self._timeout = timeout

    @property
    def name(self) -> str:
        """Return the human readable tool name."""

        return self._name

    async def _run_command(
        self,
        argv: list[str],
        *,
        cwd: Path | None = None,
    ) -> tuple[CommandResult, float]:
        """Execute an external command and record execution time."""

        start = time.perf_counter()
        result = await DEFAULT_RUNNER(
            argv,
            cwd=str(cwd) if cwd else None,
            timeout=float(self._timeout),
        )
        duration_ms = (time.perf_counter() - start) * 1000
        return result, duration_ms

    @abstractmethod
    async def execute(self, *, code: str) -> ToolResult:
        """Execute the tool against provided code and return a result."""


class BanditTool(SimpleMCPTool):
    """Bandit security scanner wrapper."""

    def __init__(self) -> None:
        super().__init__("bandit", timeout=60)

    async def execute(self, *, code: str) -> ToolResult:
        start = time.perf_counter()
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write(code)
            temp_path = Path(handle.name)

        try:
            command = ["bandit", "-f", "json", "-ll", str(temp_path)]
            result, duration_ms = await self._run_command(command, cwd=temp_path.parent)
            payload = json.loads(result.stdout or "{}")
            findings = payload.get("results", [])
            vulnerabilities = [
                {
                    "issue_text": item.get("issue_text", ""),
                    "issue_severity": item.get("issue_severity", ""),
                    "line_number": int(item.get("line_number", 0) or 0),
                    "test_id": item.get("test_id", ""),
                }
                for item in findings
                if isinstance(item, dict)
            ]
            summary = {
                "critical": 0,
                "high": sum(1 for item in vulnerabilities if item["issue_severity"].upper() == "HIGH"),
                "medium": sum(1 for item in vulnerabilities if item["issue_severity"].upper() == "MEDIUM"),
                "low": sum(1 for item in vulnerabilities if item["issue_severity"].upper() == "LOW"),
            }
            return ToolResult(
                success=True,
                output={"vulnerabilities": vulnerabilities, "summary": summary},
                execution_time_ms=duration_ms,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(success=False, output={}, error=str(exc), execution_time_ms=duration_ms)
        finally:
            temp_path.unlink(missing_ok=True)

class RuffTool(SimpleMCPTool):
    """Ruff linting wrapper."""

    def __init__(self) -> None:
        super().__init__("ruff", timeout=30)

    async def execute(self, *, code: str) -> ToolResult:
        start = time.perf_counter()
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write(code)
            temp_path = Path(handle.name)

        try:
            command = ["ruff", "check", "--output-format=json", str(temp_path)]
            result, duration_ms = await self._run_command(command, cwd=temp_path.parent)
            payload = json.loads(result.stdout or "[]")
            violations = [
                {
                    "code": item.get("code", ""),
                    "message": item.get("message", ""),
                    "line": int(item.get("location", {}).get("row", 0) or 0),
                    "column": int(item.get("location", {}).get("column", 0) or 0),
                }
                for item in payload
                if isinstance(item, dict)
            ]
            return ToolResult(
                success=True,
                output={"violations": violations, "count": len(violations)},
                execution_time_ms=duration_ms,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(success=False, output={}, error=str(exc), execution_time_ms=duration_ms)
        finally:
            temp_path.unlink(missing_ok=True)


class MypyTool(SimpleMCPTool):
    """Mypy static typing wrapper."""

    def __init__(self) -> None:
        super().__init__("mypy", timeout=60)

    async def execute(self, *, code: str) -> ToolResult:
        start = time.perf_counter()
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write(code)
            temp_path = Path(handle.name)

        try:
            command = [
                "mypy",
                "--show-error-codes",
                "--no-color-output",
                str(temp_path),
            ]
            result, duration_ms = await self._run_command(command, cwd=temp_path.parent)
            combined_output = "\n".join(
                line for line in (result.stdout + "\n" + result.stderr).splitlines() if line
            )
            errors = [
                line.replace(str(temp_path), "<tmp>")
                for line in combined_output.splitlines()
                if temp_path.name in line
            ]
            return ToolResult(
                success=True,
                output={"errors": errors, "error_count": len(errors)},
                execution_time_ms=duration_ms,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(success=False, output={}, error=str(exc), execution_time_ms=duration_ms)
        finally:
            temp_path.unlink(missing_ok=True)
