"""Constitutional alignment checks for runtime decisions.

The design follows the Constitutional AI methodology (Bai et al., 2022), using a
small set of normative principles to gate proposed actions before execution.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, List, Tuple


@dataclass(slots=True)
class PrincipleEvaluation:
    """Structured evaluation for a single constitutional principle."""

    principle: str
    approved: bool
    rationale: str


class ConstitutionalViolationError(RuntimeError):
    """Raised when an action fails constitutional review."""


class ConstitutionalReasoner:
    """Evaluate actions against a curated set of constitutional principles."""

    def __init__(
        self,
        *,
        principles_loader: Callable[[], Awaitable[Iterable[str]]] | None = None,
    ) -> None:
        self._principles_loader = principles_loader
        self._principles_cache: list[str] | None = None
        self._lock = asyncio.Lock()

    async def evaluate_action(
        self,
        proposed_action: dict[str, Any],
        current_context: dict[str, Any] | None = None,
    ) -> Tuple[bool, str]:
        """Check whether a proposed action aligns with the constitution.

        Args:
            proposed_action: Structured description of the action under review.
            current_context: Additional context such as session metadata or
                triggering user request.

        Returns:
            Tuple of ``(approved, reasoning)`` where ``approved`` indicates
            whether the action satisfies all principles and ``reasoning`` is a
            natural language justification summarising the evaluation.

        Raises:
            ValueError: If ``proposed_action`` is not a dictionary.
        """

        if not isinstance(proposed_action, dict):
            raise ValueError("proposed_action must be a dictionary")
        context = current_context or {}
        principles = await self._load_constitutional_principles()
        evaluations: list[PrincipleEvaluation] = []
        for principle in principles:
            evaluation = await self._evaluate_against_principle(
                action=proposed_action, principle=principle, context=context
            )
            evaluations.append(evaluation)
        approved = all(ev.approved for ev in evaluations)
        reasoning = self._synthesize_reasoning(evaluations)
        return approved, reasoning

    async def _load_constitutional_principles(self) -> list[str]:
        async with self._lock:
            if self._principles_cache is not None:
                return list(self._principles_cache)
            if self._principles_loader is not None:
                loaded = await self._principles_loader()
                self._principles_cache = [str(p) for p in loaded]
            else:
                self._principles_cache = [
                    "Transparency: All actions must be explainable",
                    "Beneficence: Must benefit user without harm",
                    "Autonomy: Must respect user agency",
                    "Justice: Must be fair and unbiased",
                    "Non-maleficence: Must not cause harm",
                ]
            return list(self._principles_cache)

    async def _evaluate_against_principle(
        self,
        *,
        action: dict[str, Any],
        principle: str,
        context: dict[str, Any],
    ) -> PrincipleEvaluation:
        label = principle.split(":", 1)[0].strip().lower()
        rationale: str
        approved = True
        if label == "transparency":
            approved, rationale = self._check_transparency(action)
        elif label == "beneficence":
            approved, rationale = self._check_beneficence(action, context)
        elif label == "autonomy":
            approved, rationale = self._check_autonomy(action, context)
        elif label == "justice":
            approved, rationale = self._check_justice(action, context)
        elif label == "non-maleficence":
            approved, rationale = self._check_non_maleficence(action, context)
        else:
            approved, rationale = True, "No specialised rule; default approval."
        return PrincipleEvaluation(
            principle=principle,
            approved=approved,
            rationale=rationale,
        )

    def _synthesize_reasoning(self, evaluations: Iterable[PrincipleEvaluation]) -> str:
        fragments = []
        for evaluation in evaluations:
            emoji = "✅" if evaluation.approved else "⚠️"
            fragments.append(f"{emoji} {evaluation.principle}: {evaluation.rationale}")
        return " \n".join(fragments)

    def _check_transparency(self, action: dict[str, Any]) -> tuple[bool, str]:
        ability = action.get("ability") or action.get("name")
        if not ability:
            return False, "Missing explicit ability name for auditability."
        args = action.get("args", {})
        if not isinstance(args, dict):
            return False, "Arguments must be serialisable for inspection."
        return True, f"Ability '{ability}' provides explicit metadata."

    def _check_beneficence(
        self, action: dict[str, Any], context: dict[str, Any]
    ) -> tuple[bool, str]:
        if action.get("risk_level") == "high":
            return False, "Risk level reported as high; requires manual review."
        user_goal = context.get("user_goal") or context.get("goal")
        if user_goal:
            return True, f"Action supports articulated goal: {user_goal!r}."
        return True, "No conflicting incentives detected."

    def _check_autonomy(
        self, action: dict[str, Any], context: dict[str, Any]
    ) -> tuple[bool, str]:
        if context.get("requires_confirmation") and not action.get("confirmed"):
            return False, "User confirmation required but not recorded."
        if action.get("override_user_choice"):
            return False, "Action overrides an explicit user decision."
        return True, "User agency preserved."

    def _check_justice(
        self, action: dict[str, Any], context: dict[str, Any]
    ) -> tuple[bool, str]:
        segments = context.get("affected_segments")
        if isinstance(segments, list) and len(set(segments)) != len(segments):
            return False, "Duplicate segments hint at unfair targeting."
        return True, "No bias indicators detected."

    def _check_non_maleficence(
        self, action: dict[str, Any], context: dict[str, Any]
    ) -> tuple[bool, str]:
        dangerous_flags = {
            "unsafe",
            "delete_system_files",
            "exfiltrate",
            "self_harm",
        }
        action_text = str(action)
        if any(flag in action_text.lower() for flag in dangerous_flags):
            return False, "Action description matches prohibited patterns."
        if context.get("safety_block"):
            return False, "Context flagged the action as unsafe."
        return True, "No harm vectors detected."
