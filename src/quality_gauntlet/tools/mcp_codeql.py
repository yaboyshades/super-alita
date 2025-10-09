"""MCP wrapper for CodeQL static analysis."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field  # type: ignore[import-not-found]

from . import CommandRunner
from ._runner import DEFAULT_RUNNER


class StaticAnalysisFinding(BaseModel):
    """Single static analysis finding from SARIF."""

    rule_id: str
    level: str
    message: str


class StaticAnalysisResult(BaseModel):
    """Aggregated static analysis output."""

    findings: list[StaticAnalysisFinding] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)


class CodeQLAnalysisError(RuntimeError):
    """Raised when CodeQL returns an invalid SARIF payload."""


class CodeQLMCPTool:
    """Execute CodeQL against snippets and return structured results."""

    def __init__(
        self,
        database_language: str,
        *,
        runner: CommandRunner | None = None,
    ) -> None:
        self._language = database_language
        self._runner = runner or DEFAULT_RUNNER

    async def analyze_code(self, code: str) -> StaticAnalysisResult:
        """Execute CodeQL analysis for the provided code snippet."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_file = tmp_path / f"code.{self._language}"
            source_file.write_text(code, encoding="utf-8")

            db_path = tmp_path / "db"
            sarif_path = tmp_path / "result.sarif"

            await self._runner(
                [
                    "codeql",
                    "database",
                    "create",
                    str(db_path),
                    f"--language={self._language}",
                    f"--source-root={tmp_path}",
                ],
                cwd=tmp_dir,
            )

            await self._runner(
                [
                    "codeql",
                    "database",
                    "analyze",
                    str(db_path),
                    f"{self._language}-security-and-quality.qls",
                    "--format=sarif-latest",
                    f"--output={sarif_path}",
                ],
                cwd=tmp_dir,
            )

            sarif = self._load_sarif(sarif_path)
            findings: list[StaticAnalysisFinding] = []
            runs = sarif.get("runs", [])
            if not isinstance(runs, list):
                runs = []
            for run in runs:
                if not isinstance(run, dict):
                    continue
                results = run.get("results", [])
                if not isinstance(results, list):
                    continue
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    findings.append(
                        StaticAnalysisFinding(
                            rule_id=item.get("ruleId", "unknown"),
                            level=item.get("level", "note"),
                            message=item.get("message", {}).get("text", ""),
                        )
                    )
            summary = {
                "error": sum(1 for f in findings if f.level == "error"),
                "warning": sum(1 for f in findings if f.level == "warning"),
                "note": sum(1 for f in findings if f.level == "note"),
            }
            return StaticAnalysisResult(findings=findings, summary=summary)

    def score(self, analysis: StaticAnalysisResult) -> float:
        """Compute a normalized quality score from analysis summary."""

        weights = {"error": 0.5, "warning": 0.3, "note": 0.1}
        penalty = sum(
            analysis.summary[level] * weight for level, weight in weights.items()
        )
        normalized = min(penalty / 10.0, 1.0)
        return max(0.0, 1.0 - normalized)

    @staticmethod
    def _load_sarif(path: Path) -> dict[str, Any]:
        raw = path.read_text(encoding="utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise CodeQLAnalysisError("Malformed SARIF output") from exc
