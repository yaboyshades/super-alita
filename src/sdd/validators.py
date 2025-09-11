"""
SDD Validation Utilities
Provides validation helpers for Spec-Driven Development workflow
"""

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationResult:
    """Result of validation check."""

    passed: bool
    score: float
    message: str
    details: dict[str, Any]


class SDDValidator:
    """Validates SDD artifacts against constitutional framework."""

    def __init__(self, constitutional_threshold: float = 0.75):
        self.constitutional_threshold = constitutional_threshold

    async def validate_specification(self, content: str) -> dict[str, Any]:
        """Validate specification content against SDD rules.

        Returns a dict with:
            checks: mapping of check name -> ValidationResult as dict
            overall_score: float
            passed: bool
        """
        checks: dict[str, ValidationResult] = {}
        checks["has_user_stories"] = self._check_user_stories(content)
        checks["has_acceptance_criteria"] = self._check_acceptance_criteria(content)
        checks["constitutional_compliance"] = self._check_constitutional_elements(
            content
        )
        scores = [c.score for c in checks.values()]
        overall = sum(scores) / len(scores) if scores else 0.0
        return {
            "checks": {k: vars(v) for k, v in checks.items()},
            "overall_score": overall,
            "passed": overall >= self.constitutional_threshold,
        }

    async def validate_plan(self, content: str) -> dict[str, Any]:
        """Validate plan content against SDD rules."""
        checks: dict[str, ValidationResult] = {}
        checks["has_architecture"] = self._check_architecture_section(content)
        checks["has_dependencies"] = self._check_dependencies_section(content)
        checks["has_testing_strategy"] = self._check_testing_strategy(content)
        scores = [c.score for c in checks.values()]
        overall = sum(scores) / len(scores) if scores else 0.0
        return {
            "checks": {k: vars(v) for k, v in checks.items()},
            "overall_score": overall,
            "passed": overall >= self.constitutional_threshold,
        }

    async def validate_tasks(self, content: str) -> dict[str, Any]:
        """Validate tasks content against SDD rules."""
        checks: dict[str, ValidationResult] = {}
        checks["has_task_breakdown"] = self._check_task_breakdown(content)
        checks["has_dependencies"] = self._check_task_dependencies(content)
        checks["has_estimates"] = self._check_task_estimates(content)
        scores = [c.score for c in checks.values()]
        overall = sum(scores) / len(scores) if scores else 0.0
        return {
            "checks": {k: vars(v) for k, v in checks.items()},
            "overall_score": overall,
            "passed": overall >= self.constitutional_threshold,
        }

    def _check_user_stories(self, content: str) -> ValidationResult:
        """Check for user stories in content."""
        pattern = r"(?i)(?:as a|i want|so that)"
        matches = len(re.findall(pattern, content))

        if matches >= 3:
            return ValidationResult(True, 1.0, "Good user stories", {"count": matches})
        elif matches >= 1:
            return ValidationResult(True, 0.6, "Some user stories", {"count": matches})
        else:
            return ValidationResult(False, 0.1, "No user stories", {"count": 0})

    def _check_acceptance_criteria(self, content: str) -> ValidationResult:
        """Check for acceptance criteria in content."""
        pattern = r"(?i)(?:given|when|then|acceptance criteria)"
        matches = len(re.findall(pattern, content))

        if matches >= 3:
            return ValidationResult(
                True, 1.0, "Good acceptance criteria", {"count": matches}
            )
        elif matches >= 1:
            return ValidationResult(
                True, 0.6, "Some acceptance criteria", {"count": matches}
            )
        else:
            return ValidationResult(False, 0.1, "No acceptance criteria", {"count": 0})

    def _check_constitutional_elements(self, content: str) -> ValidationResult:
        """Check for constitutional framework compliance."""
        elements = [
            "library-first",
            "test-first",
            "simplicity",
            "integration",
            "clarity",
            "counterfactual",
        ]

        found_elements = sum(
            1 for element in elements if element.lower() in content.lower()
        )

        score = found_elements / len(elements)

        if score >= 0.8:
            return ValidationResult(
                True,
                1.0,
                "Strong constitutional compliance",
                {"elements": found_elements},
            )
        elif score >= 0.5:
            return ValidationResult(
                True,
                0.7,
                "Moderate constitutional compliance",
                {"elements": found_elements},
            )
        else:
            return ValidationResult(
                False,
                0.3,
                "Weak constitutional compliance",
                {"elements": found_elements},
            )

    def _check_architecture_section(self, content: str) -> ValidationResult:
        """Check for architecture section in plan."""
        pattern = r"(?i)(?:architecture|design|structure)"
        matches = len(re.findall(pattern, content))

        return ValidationResult(
            matches >= 1,
            1.0 if matches >= 1 else 0.2,
            f"Architecture {'defined' if matches >= 1 else 'missing'}",
            {"matches": matches},
        )

    def _check_dependencies_section(self, content: str) -> ValidationResult:
        """Check for dependencies section in plan."""
        pattern = r"(?i)(?:dependencies|libraries|packages)"
        matches = len(re.findall(pattern, content))

        return ValidationResult(
            matches >= 1,
            1.0 if matches >= 1 else 0.2,
            f"Dependencies {'defined' if matches >= 1 else 'missing'}",
            {"matches": matches},
        )

    def _check_testing_strategy(self, content: str) -> ValidationResult:
        """Check for testing strategy in plan."""
        pattern = r"(?i)(?:test|testing|coverage)"
        matches = len(re.findall(pattern, content))

        return ValidationResult(
            matches >= 2,
            1.0 if matches >= 2 else 0.5 if matches >= 1 else 0.1,
            f"Testing strategy {'strong' if matches >= 2 else 'basic' if matches >= 1 else 'missing'}",
            {"matches": matches},
        )

    def _check_task_breakdown(self, content: str) -> ValidationResult:
        """Check for task breakdown in tasks."""
        pattern = r"(?i)(?:task|step|\d+\.)"
        matches = len(re.findall(pattern, content))

        return ValidationResult(
            matches >= 3,
            1.0 if matches >= 5 else 0.7 if matches >= 3 else 0.3,
            f"Task breakdown {'good' if matches >= 5 else 'adequate' if matches >= 3 else 'insufficient'}",
            {"matches": matches},
        )

    def _check_task_dependencies(self, content: str) -> ValidationResult:
        """Check for task dependencies."""
        pattern = r"(?i)(?:depends|dependency|prerequisite)"
        matches = len(re.findall(pattern, content))

        return ValidationResult(
            matches >= 1,
            1.0 if matches >= 1 else 0.3,
            f"Dependencies {'defined' if matches >= 1 else 'missing'}",
            {"matches": matches},
        )

    def _check_task_estimates(self, content: str) -> ValidationResult:
        """Check for task estimates."""
        pattern = r"(?i)(?:estimate|effort|hours|days)"
        matches = len(re.findall(pattern, content))

        return ValidationResult(
            matches >= 1,
            1.0 if matches >= 1 else 0.3,
            f"Estimates {'provided' if matches >= 1 else 'missing'}",
            {"matches": matches},
        )
