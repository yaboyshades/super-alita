"""Shared Pydantic schemas used across the Quality Gauntlet."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field  # type: ignore[import-not-found]


class PeerReviewReport(BaseModel):
    """Outcome of a peer review pass."""

    gaps: list[str] = Field(default_factory=list)

    @property
    def compliance_score(self) -> float:
        """Compute naive compliance score based on number of gaps."""

        if not self.gaps:
            return 1.0
        penalty = min(len(self.gaps) * 0.1, 1.0)
        return round(1.0 - penalty, 4)


class IterationTelemetry(BaseModel):
    """Telemetry captured for a single gauntlet iteration."""

    iteration: int
    code_snapshot: str
    peer_review: PeerReviewReport
    security_summary: dict[str, int]
    quality_summary: dict[str, int]
    remediation_plan: list[str]
    verdict_passed: bool


class RefinementHistory(BaseModel):
    """Collection wrapper for iteration telemetry."""

    iterations: list[IterationTelemetry] = Field(default_factory=list)

    def add(self, entry: IterationTelemetry) -> None:
        self.iterations.append(entry)

    def __iter__(self) -> Iterable[IterationTelemetry]:
        return iter(self.iterations)


class QualityGauntletResult(BaseModel):
    """Serialized return payload for orchestrator runs."""

    final_code: str
    iterations: RefinementHistory
    passed: bool
    scores: dict[str, float]
    remediation_plan: list[str]
