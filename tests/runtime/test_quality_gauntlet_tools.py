"""Unit tests for Quality Gauntlet tool adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest  # type: ignore[import-not-found]

from src.quality_gauntlet.tools import CommandResult
from src.quality_gauntlet.tools.mcp_codeql import CodeQLMCPTool, StaticAnalysisResult
from src.quality_gauntlet.tools.mcp_snyk import SnykMCPTool


@pytest.mark.asyncio
async def test_snyk_tool_parses_summary_and_scores(tmp_path: Path) -> None:
    """Ensure the Snyk adapter parses vulnerabilities and computes a score."""

    payload = {
        "vulnerabilities": [
            {"severity": "high"},
            {"severity": "medium"},
        ]
    }

    class FakeRunner:
        async def __call__(
            self,
            _argv: list[str],
            *,
            cwd: str | None = None,
            timeout: float | None = None,
        ) -> CommandResult:
            del cwd, timeout
            return CommandResult(returncode=0, stdout=json.dumps(payload), stderr="")

    tool = SnykMCPTool(tmp_path, runner=FakeRunner())
    result = await tool.scan_code("print('hello')", language="py")

    assert result.summary == {"critical": 0, "high": 1, "medium": 1, "low": 0}
    assert 0.3 < tool.score(result) < 1.0


@pytest.mark.asyncio
async def test_codeql_tool_generates_summary_and_score() -> None:
    """CodeQL adapter should write SARIF and compute scores."""

    sarif_payload = {
        "runs": [
            {
                "results": [
                    {"ruleId": "R1", "level": "error", "message": {"text": "issue"}},
                    {"ruleId": "R2", "level": "warning", "message": {"text": "warn"}},
                ]
            }
        ]
    }

    class FakeCodeQLRunner:
        async def __call__(
            self,
            argv: list[str],
            *,
            cwd: str | None = None,
            timeout: float | None = None,
        ) -> CommandResult:
            del cwd, timeout
            if "--output=" in argv[-1]:
                output = argv[-1].split("=", 1)[1]
                Path(output).write_text(json.dumps(sarif_payload), encoding="utf-8")
            return CommandResult(returncode=0, stdout="", stderr="")

    tool = CodeQLMCPTool("python", runner=FakeCodeQLRunner())
    analysis = await tool.analyze_code("print('hi')")

    assert isinstance(analysis, StaticAnalysisResult)
    assert analysis.summary == {"error": 1, "warning": 1, "note": 0}
    score = tool.score(analysis)
    assert 0.0 <= score <= 1.0
