"""Hierarchical orchestrator implementing the Quality Gauntlet loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

from .agents import (
    CoderAgentProtocol,
    PeerReviewerProtocol,
    QualityEvaluatorProtocol,
    SecurityScannerProtocol,
    StaticAnalyzerProtocol,
)
from .config import GauntletConfig
from .schemas import (
    IterationTelemetry,
    PeerReviewReport,
    QualityGauntletResult,
    RefinementHistory,
)
from .tools.mcp_codeql import StaticAnalysisResult
from .tools.mcp_snyk import SecurityScanResult


class GauntletTask(Protocol):
    """Minimum task contract consumed by the gauntlet."""

    title: str
    description: str
    acceptance_criteria: list[str]


@dataclass(slots=True)
class QualityVerdict:
    """Verdict emitted by the quality evaluator."""

    passed: bool
    security_score: float
    quality_score: float
    compliance_score: float
    constitutional_score: float
    remediation_plan: list[str]
    reasoning: str


class KeywordPeerReviewer:
    """Lightweight peer reviewer using keyword heuristics."""

    def __init__(self, task: GauntletTask) -> None:
        self._criteria = list(task.acceptance_criteria)

    def review(self, code: str) -> list[str]:
        gaps: list[str] = []
        lowered = code.lower()
        for criterion in self._criteria:
            tokens = [token.strip().lower() for token in criterion.split() if token]
            if not tokens:
                continue
            if not any(token in lowered for token in tokens):
                gaps.append(f"Missing coverage for criterion: {criterion}")
        return gaps


class NullSecurityScanner:
    """Fallback security scanner returning an empty result."""

    async def scan(self, _code: str) -> SecurityScanResult:  # noqa: D401
        return SecurityScanResult(
            vulnerabilities=[],
            summary={"critical": 0, "high": 0, "medium": 0, "low": 0},
        )


class NullStaticAnalyzer:
    """Fallback static analyzer returning a clean result."""

    async def analyze(self, _code: str) -> StaticAnalysisResult:  # noqa: D401
        return StaticAnalysisResult(
            findings=[], summary={"error": 0, "warning": 0, "note": 0}
        )


class DeterministicQualityEvaluator:
    """Simple evaluator combining scores against configured thresholds."""

    def __init__(self, config: GauntletConfig) -> None:
        self._thresholds = config.thresholds

    def score(
        self,
        *,
        code: str,
        peer_gaps: list[str],
        security_score: float,
        quality_score: float,
    ) -> QualityVerdict:
        _ = code
        compliance_score = max(0.0, 1.0 - min(len(peer_gaps) * 0.1, 1.0))
        constitutional_score = min(security_score, quality_score, compliance_score)
        remediation_plan: list[str] = []

        if compliance_score < self._thresholds.compliance:
            remediation_plan.append(
                "Address peer review gaps and align with acceptance criteria."
            )
        if security_score < self._thresholds.security:
            remediation_plan.append(
                "Resolve reported security vulnerabilities before merging."
            )
        if quality_score < self._thresholds.quality:
            remediation_plan.append(
                "Fix static analysis findings flagged by CodeQL/Ruff."
            )

        passed = (
            security_score >= self._thresholds.security
            and quality_score >= self._thresholds.quality
            and compliance_score >= self._thresholds.compliance
            and constitutional_score >= self._thresholds.constitutional
        )

        reasoning = (
            "All thresholds satisfied." if passed else "Quality thresholds not yet met."
        )
        return QualityVerdict(
            passed=passed,
            security_score=round(security_score, 4),
            quality_score=round(quality_score, 4),
            compliance_score=round(compliance_score, 4),
            constitutional_score=round(constitutional_score, 4),
            remediation_plan=remediation_plan,
            reasoning=reasoning,
        )


class QualityGauntletOrchestrator:
    """Drive iterative refinement using injected agents and tools."""

    def __init__(
        self,
        task: GauntletTask,
        *,
        coder: CoderAgentProtocol,
        peer_reviewer: PeerReviewerProtocol | None = None,
        security_scanner: SecurityScannerProtocol | None = None,
        static_analyzer: StaticAnalyzerProtocol | None = None,
        evaluator: QualityEvaluatorProtocol | None = None,
        config: GauntletConfig | None = None,
    ) -> None:
        self._task = task
        self._coder = coder
        self._peer_reviewer = peer_reviewer or KeywordPeerReviewer(task)
        self._security_scanner = security_scanner or NullSecurityScanner()
        self._static_analyzer = static_analyzer or NullStaticAnalyzer()
        self._config = config or GauntletConfig()
        self._evaluator = evaluator or DeterministicQualityEvaluator(self._config)

    async def run(self) -> QualityGauntletResult:
        history = RefinementHistory()
        feedback: str | None = None
        latest_code = ""
        latest_verdict: QualityVerdict | None = None

        for iteration in range(1, self._config.max_iterations + 1):
            latest_code = self._coder.generate_code(feedback=feedback)
            peer_gaps = self._peer_reviewer.review(latest_code)
            peer_report = PeerReviewReport(gaps=peer_gaps)

            security_result = await self._security_scanner.scan(latest_code)
            quality_result = await self._static_analyzer.analyze(latest_code)

            security_value = 1.0 if not security_result.vulnerabilities else 0.5
            security_score = getattr(self._security_scanner, "score", None)
            if callable(security_score):
                score_callable = cast(
                    Callable[[SecurityScanResult], float], security_score
                )
                security_value = score_callable(security_result)

            quality_value = 1.0 if not quality_result.findings else 0.5
            quality_score = getattr(self._static_analyzer, "score", None)
            if callable(quality_score):
                quality_callable = cast(
                    Callable[[StaticAnalysisResult], float], quality_score
                )
                quality_value = quality_callable(quality_result)

            verdict = self._evaluator.score(
                code=latest_code,
                peer_gaps=peer_gaps,
                security_score=security_value,
                quality_score=quality_value,
            )
            latest_verdict = verdict

            history.add(
                IterationTelemetry(
                    iteration=iteration,
                    code_snapshot=latest_code,
                    peer_review=peer_report,
                    security_summary=security_result.summary,
                    quality_summary=quality_result.summary,
                    remediation_plan=verdict.remediation_plan,
                    verdict_passed=verdict.passed,
                )
            )

            if verdict.passed:
                break

            feedback = "\n".join(
                verdict.remediation_plan or ["Improve coverage for missing criteria."]
            )

        if latest_verdict is None:
            raise RuntimeError("Quality Gauntlet did not produce a verdict")

        scores = {
            "security": latest_verdict.security_score,
            "quality": latest_verdict.quality_score,
            "compliance": latest_verdict.compliance_score,
            "constitutional": latest_verdict.constitutional_score,
        }

        return QualityGauntletResult(
            final_code=latest_code,
            iterations=history,
            passed=latest_verdict.passed,
            scores=scores,
            remediation_plan=latest_verdict.remediation_plan,
        )
