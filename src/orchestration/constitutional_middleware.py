"""Constitutional compliance middleware with auto-remediation.

Intercepts all events and operations, validates against constitutional articles,
triggers auto-remediation workflows, and blocks non-compliant operations.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.contracts import UnifiedEvent, Violation


class InterceptAction(Enum):
    """Action taken by middleware on an event."""

    ALLOW = "allow"  # Pass event through
    BLOCK = "block"  # Block event from propagating
    TRANSFORM = "transform"  # Modify event before passing
    DEFER = "defer"  # Queue for later processing


@dataclass
class InterceptResult:
    """Result of middleware interception.

    Attributes:
        action: Action to take
        violations: List of constitutional violations found
        transformed_event: Modified event (if action=TRANSFORM)
        remediation_applied: Whether auto-remediation was attempted
        remediation_success: Whether remediation succeeded
        message: Human-readable explanation
    """

    action: InterceptAction
    violations: list[Violation]
    transformed_event: UnifiedEvent | None = None
    remediation_applied: bool = False
    remediation_success: bool = False
    message: str = ""

    @classmethod
    def ALLOW(cls, message: str = "Compliant") -> InterceptResult:
        """Create ALLOW result."""
        return cls(
            action=InterceptAction.ALLOW, violations=[], message=message
        )

    @classmethod
    def BLOCK(
        cls, violations: list[Violation], message: str = ""
    ) -> InterceptResult:
        """Create BLOCK result."""
        return cls(
            action=InterceptAction.BLOCK,
            violations=violations,
            message=message or f"Blocked: {len(violations)} violations",
        )

    @classmethod
    def TRANSFORM(
        cls, transformed: UnifiedEvent, message: str = "Transformed"
    ) -> InterceptResult:
        """Create TRANSFORM result."""
        return cls(
            action=InterceptAction.TRANSFORM,
            violations=[],
            transformed_event=transformed,
            message=message,
        )


@dataclass
class ConstitutionalArticle:
    """A constitutional article with validation logic.

    Attributes:
        id: Article identifier (e.g., "Article II")
        name: Human-readable name
        description: Article description
        validate: Async validation function returning score 0.0-1.0
        threshold: Minimum passing score
        auto_remediate: Optional remediation function
    """

    id: str
    name: str
    description: str
    validate: Callable[[UnifiedEvent], Coroutine[Any, Any, float]]
    threshold: float = 0.75
    auto_remediate: (
        Callable[[UnifiedEvent, float], Coroutine[Any, Any, UnifiedEvent]]
        | None
    ) = None


class Constitution:
    """Constitutional framework with articles and governance rules."""

    def __init__(self):
        """Initialize constitution with default articles."""
        self.articles: dict[str, ConstitutionalArticle] = {}
        self.global_threshold = 0.75

    def add_article(self, article: ConstitutionalArticle) -> None:
        """Register a constitutional article.

        Args:
            article: Article to register
        """
        self.articles[article.id] = article

    def remove_article(self, article_id: str) -> None:
        """Remove a constitutional article.

        Args:
            article_id: Article ID to remove
        """
        if article_id in self.articles:
            del self.articles[article_id]

    async def validate_all(
        self, evt: UnifiedEvent
    ) -> list[tuple[ConstitutionalArticle, float]]:
        """Validate event against all articles.

        Args:
            evt: Event to validate

        Returns:
            List of (article, score) tuples
        """
        results = []
        for article in self.articles.values():
            try:
                score = await article.validate(evt)
                results.append((article, score))
            except Exception:
                # Validation failure = score 0
                results.append((article, 0.0))
        return results

    def calculate_overall_score(
        self, article_scores: list[tuple[ConstitutionalArticle, float]]
    ) -> float:
        """Calculate weighted overall compliance score.

        Args:
            article_scores: List of (article, score) tuples

        Returns:
            Overall score 0.0-1.0
        """
        if not article_scores:
            return 1.0

        # Simple average for now - can be weighted in future
        return sum(score for _, score in article_scores) / len(article_scores)


class ConstitutionalMiddleware:
    """Middleware that validates all events against constitutional articles.

    Events flow through:
    1. Validate against all articles
    2. Identify violations (score < threshold)
    3. Attempt auto-remediation if available
    4. Allow/Block/Transform based on results
    """

    def __init__(self, constitution: Constitution, strict_mode: bool = False):
        """Initialize middleware.

        Args:
            constitution: Constitutional framework
            strict_mode: If True, block on any violation. If False, allow
                        with warnings if overall score >= threshold
        """
        self.constitution = constitution
        self.strict_mode = strict_mode
        self.violation_history: list[Violation] = []
        self.remediation_cache: dict[str, UnifiedEvent] = {}

    async def intercept(self, evt: UnifiedEvent) -> InterceptResult:
        """Intercept and validate an event.

        Args:
            evt: Event to validate

        Returns:
            InterceptResult with action and violations
        """
        # Validate against all articles
        article_scores = await self.constitution.validate_all(evt)

        # Find violations
        violations = []
        for article, score in article_scores:
            if score < article.threshold:
                violation = Violation(
                    violation_type="constitutional",
                    severity=self._score_to_severity(score),
                    article=article.id,
                    description=f"{article.name} compliance: {score:.2f} < {article.threshold:.2f}",
                    artifact=f"{evt.event_type} from {evt.source}",
                    recommendation=f"Review {article.name} requirements",
                    corr_id=evt.corr_id,
                )
                violations.append(violation)

        # No violations - allow
        if not violations:
            return InterceptResult.ALLOW(evt, "All articles satisfied")

        # Attempt auto-remediation
        remediation_attempted = False
        remediation_success = False
        remediated_event = evt

        for article, score in article_scores:
            if score < article.threshold and article.auto_remediate:
                remediation_attempted = True
                try:
                    remediated_event = await article.auto_remediate(
                        remediated_event, score
                    )
                    # Re-validate after remediation
                    new_score = await article.validate(remediated_event)
                    if new_score >= article.threshold:
                        remediation_success = True
                        # Remove this violation
                        violations = [
                            v for v in violations if v.article != article.id
                        ]
                except Exception:
                    pass

        # After remediation, check if we can proceed
        if not violations:
            return InterceptResult(
                action=InterceptAction.TRANSFORM,
                violations=[],
                transformed_event=remediated_event,
                remediation_applied=remediation_attempted,
                remediation_success=remediation_success,
                message="Remediated to compliance",
            )

        # Calculate overall score
        overall_score = self.constitution.calculate_overall_score(
            article_scores
        )

        # Store violations for audit
        self.violation_history.extend(violations)

        # Strict mode: block on any violation
        if self.strict_mode:
            return InterceptResult.BLOCK(
                violations, "Strict mode: blocking on violations"
            )

        # Non-strict: allow if overall score acceptable
        if overall_score >= self.constitution.global_threshold:
            return InterceptResult(
                action=InterceptAction.ALLOW,
                violations=violations,
                message=f"Allowed with warnings (score={overall_score:.2f})",
            )

        # Overall score too low - block
        return InterceptResult.BLOCK(
            violations, f"Overall score {overall_score:.2f} below threshold"
        )

    def _score_to_severity(self, score: float) -> str:
        """Convert compliance score to severity level.

        Args:
            score: Compliance score 0.0-1.0

        Returns:
            Severity: "critical", "high", "medium", or "low"
        """
        if score < 0.25:
            return "critical"
        elif score < 0.5:
            return "high"
        elif score < 0.75:
            return "medium"
        else:
            return "low"

    def get_violation_summary(self) -> dict[str, Any]:
        """Get summary of violation history.

        Returns:
            Dict with violation counts by type, severity, article
        """
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_article: dict[str, int] = {}

        for violation in self.violation_history:
            by_type[violation.violation_type] = (
                by_type.get(violation.violation_type, 0) + 1
            )
            by_severity[violation.severity] = (
                by_severity.get(violation.severity, 0) + 1
            )
            if violation.article:
                by_article[violation.article] = (
                    by_article.get(violation.article, 0) + 1
                )

        return {
            "total_violations": len(self.violation_history),
            "by_type": by_type,
            "by_severity": by_severity,
            "by_article": by_article,
            "recent": self.violation_history[-10:],
        }


# Example article validators


async def validate_test_first(evt: UnifiedEvent) -> float:
    """Validate Article II: Test-First principle.

    Args:
        evt: Event to validate

    Returns:
        Score 0.0-1.0
    """
    # Check if code generation events have test requirements
    if evt.event_type == "code_generate":
        payload = evt.payload
        if "test_requirements" in payload and payload["test_requirements"]:
            return 1.0
        return 0.3  # Partial credit for having payload structure
    return 1.0  # Non-code events pass


async def validate_simplicity(evt: UnifiedEvent) -> float:
    """Validate Article III: Simplicity Gate (≤50 lines, ≤10 complexity).

    Args:
        evt: Event to validate

    Returns:
        Score 0.0-1.0
    """
    if evt.event_type == "code_generate":
        payload = evt.payload
        if "complexity" in payload:
            complexity = payload["complexity"]
            if complexity <= 10:
                return 1.0
            elif complexity <= 15:
                return 0.7
            else:
                return 0.3
    return 1.0


# Auto-remediation example


async def remediate_test_first(
    evt: UnifiedEvent, score: float
) -> UnifiedEvent:
    """Auto-remediate Test-First violations.

    Args:
        evt: Non-compliant event
        score: Current compliance score

    Returns:
        Remediated event
    """
    if evt.event_type == "code_generate":
        # Add test requirements if missing
        if "test_requirements" not in evt.payload:
            evt.payload["test_requirements"] = [
                "Unit tests for all public functions",
                "Integration tests for API endpoints",
            ]
    return evt
