"""Integration tests for the Quality Gauntlet orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest  # type: ignore[import-not-found]

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from src.quality_gauntlet.tools.mcp_codeql import StaticAnalysisResult
    from src.quality_gauntlet.tools.mcp_snyk import SecurityScanResult

from src.quality_gauntlet.config import GauntletConfig, QualityThresholds
from src.quality_gauntlet.orchestrator import (
    QualityGauntletOrchestrator,
    QualityVerdict,
)
from src.quality_gauntlet.schemas import IterationTelemetry


@dataclass
class FakeCoder:
    """Deterministic coder returning predefined implementations."""

    outputs: list[str]
    call_count: int = 0

    def generate_code(self, *, feedback: str | None = None) -> str:  # noqa: D401
        _ = feedback
        index = min(self.call_count, len(self.outputs) - 1)
        self.call_count += 1
        return self.outputs[index]


class CleanSecurityScanner:
    async def scan(self, _code: str) -> SecurityScanResult:  # noqa: D401
        from src.quality_gauntlet.tools.mcp_snyk import SecurityScanResult

        return SecurityScanResult(
            vulnerabilities=[],
            summary={"critical": 0, "high": 0, "medium": 0, "low": 0},
        )

    def score(self, _result: SecurityScanResult) -> float:  # noqa: D401
        return 1.0


class CleanStaticAnalyzer:
    async def analyze(self, _code: str) -> StaticAnalysisResult:  # noqa: D401
        from src.quality_gauntlet.tools.mcp_codeql import StaticAnalysisResult

        return StaticAnalysisResult(
            findings=[], summary={"error": 0, "warning": 0, "note": 0}
        )

    def score(self, _result: StaticAnalysisResult) -> float:  # noqa: D401
        return 1.0


class StrictEvaluator:
    def __init__(self, config: GauntletConfig) -> None:
        self._config = config

    def score(
        self,
        *,
        code: str,
        peer_gaps: list[str],
        security_score: float,
        quality_score: float,
        constitutional_score: float,
    ) -> QualityVerdict:
        _ = code
        compliance = 1.0 if not peer_gaps else 0.0
        passed = (
            compliance >= self._config.thresholds.compliance
            and security_score >= self._config.thresholds.security
            and quality_score >= self._config.thresholds.quality
            and constitutional_score >= self._config.thresholds.constitutional
        )
        return QualityVerdict(
            passed=passed,
            security_score=security_score,
            quality_score=quality_score,
            compliance_score=compliance,
            constitutional_score=constitutional_score,
            remediation_plan=["Satisfy acceptance criteria."] if peer_gaps else [],
            reasoning="strict evaluation",
        )


@pytest.mark.asyncio
async def test_orchestrator_passes_with_clean_code() -> None:
    task = SimpleTask(
        id="T-1",
        title="Add numbers",
        description="Implement add function",
        acceptance_criteria=["returns sum", "two parameters"],
    )

    coder = FakeCoder(
        outputs=[
            (
                "def add(a, b):\n"
                "    return a + b  # returns sum and uses two parameters\n"
            )
        ]
    )
    config = GauntletConfig(
        max_iterations=1,
        thresholds=QualityThresholds(),
        enable_bandit=False,
        enable_ruff=False,
        enable_mypy=False,
        enable_codeql=True,
    )

    orchestrator = QualityGauntletOrchestrator(
        task,
        coder=coder,
        security_scanner=CleanSecurityScanner(),
        static_analyzer=CleanStaticAnalyzer(),
        evaluator=StrictEvaluator(config),
        config=config,
    )

    result = await orchestrator.run()

    assert result.passed is True
    assert result.scores["security"] == 1.0
    assert result.scores["constitutional"] >= config.thresholds.constitutional
    iteration_entries: list[IterationTelemetry] = list(result.iterations)
    assert any(isinstance(entry, IterationTelemetry) for entry in iteration_entries)
    for entry in iteration_entries:
        assert entry.constitutional_report.score >= config.thresholds.constitutional


@pytest.mark.asyncio
async def test_orchestrator_requires_refinement_cycle() -> None:
    task = SimpleTask(
        id="T-2",
        title="Greeter",
        description="Return greeting",
        acceptance_criteria=["return greeting", "accept name"],
    )

    coder = FakeCoder(
        outputs=[
            "def greet():\n    return 'hi'\n",
            (
                "def greet(name):\n"
                "    return f'hello {name}'  # return greeting accept name\n"
            ),
        ]
    )

    config = GauntletConfig(
        max_iterations=2,
        thresholds=QualityThresholds(),
        enable_bandit=False,
        enable_ruff=False,
        enable_mypy=False,
        enable_codeql=True,
    )

    orchestrator = QualityGauntletOrchestrator(
        task,
        coder=coder,
        security_scanner=CleanSecurityScanner(),
        static_analyzer=CleanStaticAnalyzer(),
        evaluator=StrictEvaluator(config),
        config=config,
    )

    result = await orchestrator.run()

    assert result.passed is True
    assert coder.call_count == 2
    assert len(result.iterations.iterations) == 2
    # First iteration should have remediation plan
    assert result.iterations.iterations[0].remediation_plan
    assert (
        result.iterations.iterations[-1].constitutional_report.score
        >= config.thresholds.constitutional
    )


@dataclass
class SimpleTask:
    """Minimal task container matching GauntletTask protocol."""

    id: str
    title: str
    description: str
    acceptance_criteria: list[str]
