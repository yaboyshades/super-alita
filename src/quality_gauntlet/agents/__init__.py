"""Agent interfaces used by the Quality Gauntlet."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from ..orchestrator import QualityVerdict
    from ..tools.mcp_codeql import StaticAnalysisResult
    from ..tools.mcp_snyk import SecurityScanResult


class CoderAgentProtocol(Protocol):
    """Minimal protocol required for code generation agents."""

    def generate_code(self, *, feedback: str | None = None) -> str:
        """Return source code for the current iteration."""


class PeerReviewerProtocol(Protocol):
    """Protocol for peer review implementations."""

    def review(self, code: str) -> list[str]:
        """Return list of spec compliance gaps."""


class SecurityScannerProtocol(Protocol):
    """Protocol for security tooling (e.g., Snyk)."""

    async def scan(
        self, code: str
    ) -> SecurityScanResult:  # pragma: no cover - runtime type check
        """Perform an async security scan and return structured findings."""


class StaticAnalyzerProtocol(Protocol):
    """Protocol for static analyzers (CodeQL, Ruff)."""

    async def analyze(self, code: str) -> StaticAnalysisResult:  # pragma: no cover
        """Run analysis and return structured findings."""


class QualityEvaluatorProtocol(Protocol):
    """Protocol for synthesizing final verdicts."""

    def score(
        self,
        *,
        code: str,
        peer_gaps: list[str],
        security_score: float,
        quality_score: float,
        constitutional_score: float,
    ) -> QualityVerdict:  # pragma: no cover
        """Return a verdict object describing pass/fail and remediation plan."""
