"""MCP-compatible wrapper for invoking Snyk Code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field  # type: ignore[import-not-found]

from . import CommandResult, CommandRunner
from ._runner import DEFAULT_RUNNER


class SecurityScanResult(BaseModel):
    """Container for Snyk security findings."""

    vulnerabilities: list[dict[str, str]] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)


class SnykScanError(RuntimeError):
    """Raised when the Snyk CLI returns malformed output."""


class SnykMCPTool:
    """Thin wrapper over the Snyk CLI returning structured output."""

    def __init__(
        self,
        project_path: Path,
        *,
        runner: CommandRunner | None = None,
    ) -> None:
        self._project_path = project_path
        self._runner = runner or DEFAULT_RUNNER

    async def scan_code(self, code: str, language: str) -> SecurityScanResult:
        """Scan the provided code and return normalized results."""

        temp_file = self._project_path / f".snyk_tmp.{language}"
        temp_file.write_text(code, encoding="utf-8")

        try:
            argv = [
                "snyk",
                "code",
                "test",
                str(temp_file),
                "--json",
            ]
            result = await self._runner(argv, cwd=str(self._project_path))
            payload = self._parse_payload(result)
            vulns_data = payload.get("vulnerabilities", [])
            if not isinstance(vulns_data, list):
                vulns_data = []
            vulns = [dict(entry) for entry in vulns_data if isinstance(entry, dict)]
            summary = {
                "critical": sum(1 for v in vulns if v.get("severity") == "critical"),
                "high": sum(1 for v in vulns if v.get("severity") == "high"),
                "medium": sum(1 for v in vulns if v.get("severity") == "medium"),
                "low": sum(1 for v in vulns if v.get("severity") == "low"),
            }
            return SecurityScanResult(vulnerabilities=vulns, summary=summary)
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def score(self, scan_result: SecurityScanResult) -> float:
        """Convert a scan result into a normalized security score."""

        weights = {
            "critical": 0.4,
            "high": 0.3,
            "medium": 0.2,
            "low": 0.1,
        }
        penalty = sum(
            scan_result.summary[level] * weight for level, weight in weights.items()
        )
        return max(0.0, 1.0 - min(penalty, 1.0))

    @staticmethod
    def _parse_payload(result: CommandResult) -> dict[str, Any]:
        """Parse JSON payload from Snyk output."""

        raw = result.stdout.strip() or result.stderr.strip()
        if not raw:
            raise SnykScanError("Snyk produced no JSON output")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise SnykScanError("Malformed Snyk JSON payload") from exc
