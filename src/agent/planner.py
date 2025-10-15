"""Planner responsible for cost/risk optimised agent task decomposition."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .abilities.registry import AbilityRegistry, Ability


@dataclass(slots=True)
class PlanStep:
    """Structured representation of a planning step."""

    action: str
    parameters: Dict[str, Any]
    cost_estimate: float
    risk_level: str
    dependencies: List[str]
    fallback_action: Optional[str] = None
    timeout_seconds: int = 30


class Planner:
    """Task decomposition with knowledge graph awareness and risk management."""

    def __init__(
        self,
        knowledge_graph: Any,
        llm_orchestrator: Any,
        ability_registry: AbilityRegistry,
    ) -> None:
        self.knowledge_graph = knowledge_graph
        self.llm_orchestrator = llm_orchestrator
        self.ability_registry = ability_registry
        self.logger = logging.getLogger(__name__)

    async def generate_plan(
        self, goal: str, constraints: List[str], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate an execution plan with cost and risk annotations."""
        self.logger.info("Generating plan for goal: %s", goal)
        try:
            await self._find_similar_plans(goal)
            subtasks = await self._decompose_goal(goal, constraints, context)
            annotated_steps = await self._annotate_steps_with_cost_risk(subtasks)
            optimized_plan = await self._optimize_plan_sequence(annotated_steps)
            robust_plan = await self._add_fallbacks_and_error_handling(optimized_plan)
            self.logger.info("Generated plan with %s steps", len(robust_plan))
            return robust_plan
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.error("Plan generation failed: %s", exc)
            return await self._generate_fallback_plan(goal)

    async def _find_similar_plans(self, goal: str) -> List[Dict[str, Any]]:
        try:
            sessions = await self.knowledge_graph.semantic_search(
                goal,
                limit=5,
                filters={"type": "agent_session", "metadata.success": True},
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.warning("Failed to find similar plans: %s", exc)
            return []

        plans: List[Dict[str, Any]] = []
        for session in sessions:
            content = session.get("content", {})
            if "current_plan" not in content:
                continue
            plans.append(
                {
                    "goal": content.get("goal"),
                    "steps": content.get("current_plan", []),
                    "success_rate": await self._calculate_session_success_rate(session),
                }
            )
        return plans

    async def _decompose_goal(
        self, goal: str, constraints: List[str], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if await self._is_complex_goal(goal):
            return await self._llm_decompose_goal(goal, constraints, context)
        return await self._rule_based_decomposition(goal, constraints)

    async def _is_complex_goal(self, goal: str) -> bool:
        complex_indicators = [
            "multiple",
            "several",
            "complex",
            "advanced",
            "integrate",
            "orchestrate",
            "coordinate",
        ]
        goal_lower = goal.lower()
        return len(goal.split()) > 15 or any(
            indicator in goal_lower for indicator in complex_indicators
        )

    async def _llm_decompose_goal(
        self, goal: str, constraints: List[str], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        prompt = (
            "Decompose the following goal into executable steps as JSON.\n\n"
            f"Goal: {goal}\n"
            f"Constraints: {', '.join(constraints)}\n"
            f"Context: {context.get('relevant_patterns', [])}\n"
            "Return an array with action, parameters, and description."
        )
        try:
            response = await self.llm_orchestrator.generate(prompt)
            payload = getattr(response, "content", response)
            if not isinstance(payload, str):
                payload = json.dumps(payload)
            return self._parse_llm_response(payload)
        except Exception as exc:
            self.logger.warning("LLM decomposition failed, falling back: %s", exc)
            return await self._rule_based_decomposition(goal, constraints)

    async def _rule_based_decomposition(
        self, goal: str, constraints: List[str]
    ) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        goal_lower = goal.lower()
        if any(word in goal_lower for word in ["analyze", "review", "check", "inspect"]):
            steps.append(
                {
                    "action": "code_analysis",
                    "parameters": {"target": goal},
                    "description": "Analyze the code or system",
                }
            )
        if any(word in goal_lower for word in ["find", "search", "locate"]):
            steps.append(
                {
                    "action": "search_code",
                    "parameters": {"query": goal},
                    "description": "Search for relevant code",
                }
            )
        if any(word in goal_lower for word in ["create", "write", "generate", "implement"]):
            steps.append(
                {
                    "action": "code_generation",
                    "parameters": {"requirement": goal},
                    "description": "Generate code based on requirements",
                }
            )
        if not steps:
            steps.append(
                {
                    "action": "analyze_requirements",
                    "parameters": {"goal": goal, "constraints": constraints},
                    "description": "Analyze requirements and determine approach",
                }
            )
        return steps

    async def _annotate_steps_with_cost_risk(
        self, steps: List[Dict[str, Any]]
    ) -> List[PlanStep]:
        annotated: List[PlanStep] = []
        for step in steps:
            ability: Ability | None = self.ability_registry.get_ability(step.get("action", ""))
            if ability:
                cost_estimate = ability.metadata.cost_estimate
                risk_level = ability.metadata.risk_level
            else:
                cost_estimate = 1.0
                risk_level = "high"
            annotated.append(
                PlanStep(
                    action=step.get("action", "unknown"),
                    parameters=step.get("parameters", {}),
                    cost_estimate=cost_estimate,
                    risk_level=risk_level,
                    dependencies=[],
                    timeout_seconds=ability.metadata.timeout_seconds if ability else 30,
                )
            )
        return annotated

    async def _optimize_plan_sequence(self, steps: List[PlanStep]) -> List[Dict[str, Any]]:
        low_risk = sorted([s for s in steps if s.risk_level == "low"], key=lambda x: x.cost_estimate)
        medium_risk = sorted(
            [s for s in steps if s.risk_level == "medium"], key=lambda x: x.cost_estimate
        )
        high_risk = sorted(
            [s for s in steps if s.risk_level == "high"], key=lambda x: x.cost_estimate
        )
        ordered = low_risk + medium_risk + high_risk
        return [
            {
                "action": step.action,
                "parameters": step.parameters,
                "cost_estimate": step.cost_estimate,
                "risk_level": step.risk_level,
                "timeout_seconds": step.timeout_seconds,
            }
            for step in ordered
        ]

    async def _add_fallbacks_and_error_handling(
        self, plan: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        robust_plan: List[Dict[str, Any]] = []
        for step in plan:
            updated = dict(step)
            if step.get("risk_level") == "high":
                updated["fallback_action"] = await self._get_fallback_action(step["action"])
                updated["max_retries"] = 3
            else:
                updated["max_retries"] = 1
            updated["retry_delay_seconds"] = 2
            robust_plan.append(updated)
        return robust_plan

    async def _get_fallback_action(self, action: str) -> str:
        mapping = {
            "code_execution": "code_review",
            "api_call": "local_processing",
            "file_operation": "file_analysis",
        }
        return mapping.get(action, "analyze_requirements")

    async def _generate_fallback_plan(self, goal: str) -> List[Dict[str, Any]]:
        return [
            {
                "action": "analyze_requirements",
                "parameters": {"goal": goal},
                "description": "Analyze requirements as fallback",
                "cost_estimate": 1.0,
                "risk_level": "low",
                "timeout_seconds": 30,
                "max_retries": 1,
                "retry_delay_seconds": 2,
            }
        ]

    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        try:
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1
            if start_idx == -1 or end_idx <= start_idx:
                return []
            return json.loads(response[start_idx:end_idx])
        except Exception as exc:
            self.logger.warning("Failed to parse LLM response: %s", exc)
            return []

    async def _calculate_session_success_rate(self, session: Dict[str, Any]) -> float:
        try:
            executed = session.get("content", {}).get("executed_steps", [])
            if not executed:
                return 0.0
            successes = len(
                [step for step in executed if step.get("result", {}).get("success")]
            )
            return successes / len(executed)
        except Exception:  # pragma: no cover - defensive
            return 0.0


def create_planner(
    knowledge_graph: Any, llm_orchestrator: Any, ability_registry: AbilityRegistry
) -> Planner:
    """Factory helper for creating planners."""
    return Planner(knowledge_graph, llm_orchestrator, ability_registry)
