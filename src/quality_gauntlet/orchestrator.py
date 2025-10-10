"""Quality Gauntlet orchestration and scoring pipeline."""

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
from .constitutional import ASTConstitutionalValidator
from .schemas import (
    ConstitutionalReport,
    IterationTelemetry,
    PeerReviewReport,
    QualityGauntletResult,
    RefinementHistory,
    ToolResult,
)
from .tools.mcp_codeql import StaticAnalysisResult
from .tools.mcp_snyk import SecurityScanResult
from .tools.simple_mcp import BanditTool, MypyTool, RuffTool


class GauntletTask(Protocol):
    """Minimum task contract consumed by the gauntlet."""

    title: str
    description: str
    acceptance_criteria: list[str]


@dataclass(slots=True)
class QualityScores:
    """Aggregated scorecard for a gauntlet iteration."""

    security: float
    quality: float
    compliance: float
    constitutional: float

    @property
    def overall(self) -> float:
        """Weighted overall score emphasising security and compliance."""

        return round(
            (self.security * 0.3)
            + (self.quality * 0.25)
            + (self.constitutional * 0.25)
            + (self.compliance * 0.20),
            4,
        )

    def passes_thresholds(self, config: GauntletConfig) -> bool:
        """Check if all configured thresholds are satisfied."""

        thresholds = config.thresholds
        return (
            self.security >= thresholds.security
            and self.quality >= thresholds.quality
            and self.compliance >= thresholds.compliance
            and self.constitutional >= thresholds.constitutional
        )


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

    @property
    def scores(self) -> QualityScores:
        """Expose the verdict scores as a structured object."""

        return QualityScores(
            security=self.security_score,
            quality=self.quality_score,
            compliance=self.compliance_score,
            constitutional=self.constitutional_score,
        )


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

    def score(self, _result: SecurityScanResult) -> float:  # noqa: D401
        return 1.0


class NullStaticAnalyzer:
    """Fallback static analyzer returning a clean result."""

    async def analyze(self, _code: str) -> StaticAnalysisResult:  # noqa: D401
        return StaticAnalysisResult(
            findings=[], summary={"error": 0, "warning": 0, "note": 0}
        )

    def score(self, _result: StaticAnalysisResult) -> float:  # noqa: D401
        return 1.0


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
        constitutional_score: float,
    ) -> QualityVerdict:
        _ = code
        compliance_score = max(0.0, 1.0 - min(len(peer_gaps) * 0.1, 1.0))
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
        if constitutional_score < self._thresholds.constitutional:
            remediation_plan.append(
                "Resolve constitutional violations flagged by the validator."
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

        self._constitution = ASTConstitutionalValidator(self._config.constitution_path)
        self._bandit_tool = BanditTool() if self._config.enable_bandit else None
        self._ruff_tool = RuffTool() if self._config.enable_ruff else None
        self._mypy_tool = MypyTool() if self._config.enable_mypy else None

    async def run(self) -> QualityGauntletResult:
        history = RefinementHistory()
        feedback: str | None = None
        latest_code = ""
        latest_verdict: QualityVerdict | None = None
        latest_scores: QualityScores | None = None
        latest_reports: dict[str, ToolResult] = {}

        for iteration in range(1, self._config.max_iterations + 1):
            latest_code = self._coder.generate_code(feedback=feedback)
            peer_report = PeerReviewReport(gaps=self._peer_reviewer.review(latest_code))

            tool_reports, constitutional_report = await self._run_analysis_tools(
                latest_code
            )

            security_value, security_summary = await self._evaluate_security(
                latest_code, tool_reports
            )
            quality_value, quality_summary = await self._evaluate_quality(
                latest_code, tool_reports
            )

            scores = QualityScores(
                security=round(security_value, 4),
                quality=round(quality_value, 4),
                compliance=round(peer_report.compliance_score, 4),
                constitutional=round(constitutional_report.score, 4),
            )

            verdict = self._evaluator.score(
                code=latest_code,
                peer_gaps=peer_report.gaps,
                security_score=scores.security,
                quality_score=scores.quality,
                constitutional_score=scores.constitutional,
            )
            verdict.security_score = scores.security
            verdict.quality_score = scores.quality
            verdict.compliance_score = scores.compliance
            verdict.constitutional_score = scores.constitutional

            latest_verdict = verdict
            latest_scores = scores
            latest_reports = tool_reports

            history.add(
                IterationTelemetry(
                    iteration=iteration,
                    code_snapshot=latest_code,
                    peer_review=peer_report,
                    security_summary=security_summary,
                    quality_summary=quality_summary,
                    constitutional_report=constitutional_report,
                    tool_reports=tool_reports,
                    remediation_plan=verdict.remediation_plan,
                    verdict_passed=verdict.passed,
                )
            )

            if verdict.passed or scores.passes_thresholds(self._config):
                break

            feedback = self._generate_feedback(
                scores=scores,
                peer_report=peer_report,
                tool_reports=tool_reports,
                constitutional_report=constitutional_report,
            )

        if latest_verdict is None or latest_scores is None:
            raise RuntimeError("Quality Gauntlet did not produce a verdict")

        scores_payload = {
            "security": latest_scores.security,
            "quality": latest_scores.quality,
            "compliance": latest_scores.compliance,
            "constitutional": latest_scores.constitutional,
        }

        return QualityGauntletResult(
            final_code=latest_code,
            iterations=history,
            passed=latest_verdict.passed,
            scores=scores_payload,
            remediation_plan=latest_verdict.remediation_plan,
        )

    async def _run_analysis_tools(
        self, code: str
    ) -> tuple[dict[str, ToolResult], ConstitutionalReport]:
        """Execute optional analysis tools for the supplied code."""

        tool_reports: dict[str, ToolResult] = {}

        if self._bandit_tool is not None:
            tool_reports["bandit"] = await self._bandit_tool.execute(code=code)
        if self._ruff_tool is not None:
            tool_reports["ruff"] = await self._ruff_tool.execute(code=code)
        if self._mypy_tool is not None:
            tool_reports["mypy"] = await self._mypy_tool.execute(code=code)

        violations = self._constitution.validate(code)
        constitutional_report = ConstitutionalReport(
            violations=violations,
            score=self._constitution.calculate_constitutional_score(),
        )
        return tool_reports, constitutional_report

    async def _evaluate_security(
        self, code: str, tool_reports: dict[str, ToolResult]
    ) -> tuple[float, dict[str, int]]:
        """Compute security score from either injected scanner or Bandit."""

        if (
            isinstance(self._security_scanner, NullSecurityScanner)
            or not self._config.enable_snyk
        ):
            return self._score_bandit(tool_reports.get("bandit"))

        result = await self._security_scanner.scan(code)
        tool_reports["security_scanner"] = ToolResult(
            success=True,
            output=result.dict(),
            execution_time_ms=0.0,
        )
        scorer = getattr(self._security_scanner, "score", None)
        if callable(scorer):
            score_callable = cast(Callable[[SecurityScanResult], float], scorer)
            score = score_callable(result)
        else:
            score = 1.0 if not result.vulnerabilities else 0.5
        return score, result.summary

    async def _evaluate_quality(
        self, code: str, tool_reports: dict[str, ToolResult]
    ) -> tuple[float, dict[str, int]]:
        """Compute quality score from static analyzer or lint/type tools."""

        if (
            isinstance(self._static_analyzer, NullStaticAnalyzer)
            or not self._config.enable_codeql
        ):
            return self._score_quality(
                tool_reports.get("ruff"), tool_reports.get("mypy")
            )

        result = await self._static_analyzer.analyze(code)
        tool_reports["static_analyzer"] = ToolResult(
            success=True,
            output={
                "summary": result.summary,
                "findings": [finding.dict() for finding in result.findings],
            },
            execution_time_ms=0.0,
        )
        scorer = getattr(self._static_analyzer, "score", None)
        if callable(scorer):
            score_callable = cast(Callable[[StaticAnalysisResult], float], scorer)
            score = score_callable(result)
        else:
            score = 1.0 if not result.findings else 0.5
        return score, result.summary

    def _score_bandit(self, report: ToolResult | None) -> tuple[float, dict[str, int]]:
        """Convert a Bandit tool result into a score and summary."""

        summary = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        if report is None or not report.success:
            return 1.0, summary

        raw_summary = report.output.get("summary", {})
        for key in summary:
            summary[key] = int(raw_summary.get(key, 0) or 0)

        penalty = (
            summary["critical"] * 0.4
            + summary["high"] * 0.3
            + summary["medium"] * 0.2
            + summary["low"] * 0.1
        )
        return max(0.0, 1.0 - min(penalty, 1.0)), summary

    def _score_quality(
        self,
        ruff_report: ToolResult | None,
        mypy_report: ToolResult | None,
    ) -> tuple[float, dict[str, int]]:
        """Derive a quality score from Ruff and Mypy findings."""

        lint_violations = 0
        type_errors = 0

        if ruff_report and ruff_report.success:
            lint_violations = int(ruff_report.output.get("count", 0) or 0)
        if mypy_report and mypy_report.success:
            type_errors = int(mypy_report.output.get("error_count", 0) or 0)

        penalty = (lint_violations * 0.05) + (type_errors * 0.1)
        summary = {"lint": lint_violations, "type": type_errors}
        return max(0.0, 1.0 - min(penalty, 1.0)), summary

    def _generate_feedback(
        self,
        *,
        scores: QualityScores,
        peer_report: PeerReviewReport,
        tool_reports: dict[str, ToolResult],
        constitutional_report: ConstitutionalReport,
    ) -> str:
        """Construct actionable feedback for the next refinement cycle."""

        lines = ["REMEDIATION REQUIRED:"]

        if peer_report.gaps:
            lines.append("\n🔸 Acceptance Criteria:")
            lines.extend(f"  - {gap}" for gap in peer_report.gaps)

        if scores.security < self._config.thresholds.security:
            lines.append("\n🔸 Security Findings:")
            bandit_report = tool_reports.get("bandit")
            if bandit_report and bandit_report.output.get("vulnerabilities"):
                for vuln in bandit_report.output["vulnerabilities"][:3]:
                    lines.append(
                        "  - "
                        + f"Line {vuln.get('line_number', '?')}: "
                        + vuln.get("issue_text", "Resolve vulnerability")
                    )
            else:
                lines.append(
                    "  - Investigate security tooling configuration and rerun scans."
                )

        if scores.quality < self._config.thresholds.quality:
            lines.append("\n🔸 Quality Issues:")
            ruff_report = tool_reports.get("ruff")
            if ruff_report and ruff_report.output.get("violations"):
                for violation in ruff_report.output["violations"][:3]:
                    lines.append(
                        "  - "
                        + f"Line {violation.get('line', '?')}: "
                        + violation.get("message", "Resolve lint warning")
                    )
            mypy_report = tool_reports.get("mypy")
            if mypy_report and mypy_report.output.get("errors"):
                lines.extend(
                    f"  - {error}" for error in mypy_report.output["errors"][:3]
                )

        if scores.constitutional < self._config.thresholds.constitutional:
            lines.append("\n🔸 Constitutional Compliance:")
            for violation in constitutional_report.violations:
                if violation.severity == "error":
                    lines.append(
                        "  - "
                        + f"{violation.article}: {violation.message} (line {violation.line_number})"
                    )

        if len(lines) == 1:
            lines.append(
                "No actionable items detected; review acceptance criteria manually."
            )

        return "\n".join(lines)
