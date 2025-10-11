"""LLM output validation utilities used by runtime pipelines."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "CheckOutcome",
    "LLMOutputValidator",
    "OutputValidationError",
    "ValidationSummary",
]


class OutputValidationError(RuntimeError):
    """Raised when the language model output fails validation."""


@dataclass(slots=True)
class CheckOutcome:
    """Represents the result of a single validation check."""

    name: str
    passed: bool
    score: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the outcome for telemetry payloads."""

        payload: dict[str, Any] = {"name": self.name, "passed": self.passed}
        if self.score is not None:
            payload["score"] = self.score
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass(slots=True)
class ValidationSummary:
    """Aggregated validation state across all configured checks."""

    passed: bool
    checks: dict[str, CheckOutcome]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the summary into a plain dictionary."""

        return {
            "passed": self.passed,
            "checks": {name: outcome.to_dict() for name, outcome in self.checks.items()},
        }


class CheckCallable(Protocol):
    """Protocol describing the signature of validation check callables."""

    async def __call__(
        self, agent_output: str, context: dict[str, Any] | None = None
    ) -> CheckOutcome | bool | tuple[bool, dict[str, Any]] | dict[str, Any]:
        """Validate the given output and return a check result."""


class LLMOutputValidator:
    """Aggregates multiple validation checks for agent outputs."""

    def __init__(
        self,
        bias_check: CheckCallable | None = None,
        factual_accuracy_check: CheckCallable | None = None,
        reasoning_check: CheckCallable | None = None,
        hallucination_check: CheckCallable | None = None,
    ) -> None:
        self._checks: dict[str, CheckCallable | None] = {
            "bias": bias_check,
            "factual_accuracy": factual_accuracy_check,
            "reasoning": reasoning_check,
            "hallucination": hallucination_check,
        }

    async def _run_check(
        self,
        name: str,
        checker: CheckCallable | None,
        agent_output: str,
        context: dict[str, Any] | None,
    ) -> CheckOutcome:
        if checker is None:
            return CheckOutcome(name=name, passed=True, details={"skipped": True})
        try:
            result = await checker(agent_output, context)
        except Exception as exc:  # noqa: BLE001
            return CheckOutcome(
                name=name,
                passed=False,
                details={"error": str(exc)},
            )
        return self._coerce_outcome(name, result)

    @staticmethod
    def _coerce_outcome(
        name: str,
        result: CheckOutcome | bool | tuple[bool, dict[str, Any]] | dict[str, Any],
    ) -> CheckOutcome:
        if isinstance(result, CheckOutcome):
            if not result.name:
                return CheckOutcome(
                    name=name,
                    passed=result.passed,
                    score=result.score,
                    details=result.details,
                )
            return result
        if isinstance(result, tuple) and len(result) == 2:
            passed, details = result
            details_dict = dict(details) if isinstance(details, dict) else {"details": details}
            return CheckOutcome(name=name, passed=bool(passed), details=details_dict)
        if isinstance(result, dict):
            passed = bool(result.get("passed", False))
            score = result.get("score")
            details = {k: v for k, v in result.items() if k not in {"passed", "score"}}
            return CheckOutcome(name=name, passed=passed, score=score, details=details)
        if isinstance(result, bool):
            return CheckOutcome(name=name, passed=result)
        return CheckOutcome(name=name, passed=False, details={"unexpected_result": result})

    async def validate_agent_output(
        self, agent_output: str, context: dict[str, Any] | None = None
    ) -> ValidationSummary:
        """Run all configured checks concurrently for a given output."""

        context = context or {}
        outcomes = await asyncio.gather(
            *(
                self._run_check(name, checker, agent_output, context)
                for name, checker in self._checks.items()
            )
        )
        checks = {outcome.name: outcome for outcome in outcomes}
        summary = ValidationSummary(
            passed=all(outcome.passed for outcome in outcomes), checks=checks
        )
        if not summary.passed:
            raise OutputValidationError(summary.to_dict())
        return summary
