"""Deterministic agent loop implementing observe→deliberate→plan→act→reflect."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from .planner import Planner
from .abilities.registry import AbilityRegistry, Ability


@dataclass(slots=True)
class AgentState:
    """Represents the evolving state of an agent session."""

    current_goal: str
    constraints: List[str]
    context: Dict[str, Any]
    current_plan: List[Dict[str, Any]]
    executed_steps: List[Dict[str, Any]]
    reflections: List[str]
    artifacts: Dict[str, Any]


@dataclass(slots=True)
class StepResult:
    """Container for step execution results."""

    success: bool
    output: Any
    error: Optional[str] = None
    constitutional_approved: bool = True
    reasoning: Optional[str] = None
    step_type: str = "action"


class CoreAgentLoop:
    """Deterministic agent loop orchestrating core execution phases."""

    def __init__(
        self,
        constitutional_reasoner: Any,
        event_bus: Any,
        knowledge_graph: Any,
        planner: Planner,
        ability_registry: AbilityRegistry,
    ) -> None:
        self.constitutional_reasoner = constitutional_reasoner
        self.event_bus = event_bus
        self.knowledge_graph = knowledge_graph
        self.planner = planner
        self.ability_registry = ability_registry
        self.logger = logging.getLogger(__name__)
        self.current_state: AgentState | None = None
        self.session_id: Optional[str] = None

    async def execute_task(
        self, task: str, context: Dict[str, Any], max_iterations: int = 10
    ) -> Dict[str, Any]:
        self.session_id = f"session_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        state_context = dict(context)
        self.current_state = AgentState(
            current_goal=task,
            constraints=context.get("constraints", []),
            context=state_context,
            current_plan=[],
            executed_steps=[],
            reflections=[],
            artifacts={},
        )

        await self._emit_event(
            "session.started",
            {
                "task": task,
                "context": state_context,
                "session_id": self.session_id,
            },
        )

        iteration = 0
        final_result: Optional[Dict[str, Any]] = None
        while iteration < max_iterations and not await self._is_goal_achieved():
            self.logger.info("Iteration %s for task: %s", iteration + 1, task)
            try:
                await self._observe()
                await self._deliberate()
                await self._plan()
                await self._act()
                await self._reflect()
                if await self._should_terminate():
                    final_result = await self._compile_final_result()
                    break
            except Exception as exc:
                self.logger.error("Iteration %s failed: %s", iteration, exc)
                await self._emit_event(
                    "iteration.failed",
                    {
                        "iteration": iteration,
                        "error": str(exc),
                        "state": asdict(self.current_state),
                    },
                )
                break
            iteration += 1

        if not final_result:
            final_result = await self._compile_final_result()

        await self._emit_event(
            "session.completed",
            {
                "session_id": self.session_id,
                "final_result": final_result,
                "iterations": iteration,
                "reflections": self.current_state.reflections if self.current_state else [],
            },
        )

        await self._persist_session_to_kg(final_result)
        return final_result

    async def _observe(self) -> StepResult:
        await self._emit_event(
            "step.started",
            {"step": "observe", "session_id": self.session_id},
        )
        try:
            relevant_context = await self.knowledge_graph.semantic_search(
                self.current_state.current_goal,
                limit=5,
                filters={"type": "success_pattern"},
            )
            avoid_patterns = await self.knowledge_graph.semantic_search(
                self.current_state.current_goal,
                limit=3,
                filters={"type": "failure_pattern"},
            )
            self.current_state.context.update(
                {
                    "relevant_patterns": relevant_context or [],
                    "avoid_patterns": avoid_patterns or [],
                }
            )
            result = StepResult(
                success=True,
                output={
                    "relevant_patterns": len(relevant_context or []),
                    "avoid_patterns": len(avoid_patterns or []),
                },
                step_type="observation",
            )
            await self._emit_event(
                "step.completed",
                {
                    "step": "observe",
                    "result": asdict(result),
                    "session_id": self.session_id,
                },
            )
            return result
        except Exception as exc:
            result = StepResult(success=False, output=None, error=str(exc), step_type="observation")
            await self._emit_event(
                "step.failed",
                {
                    "step": "observe",
                    "result": asdict(result),
                    "session_id": self.session_id,
                },
            )
            return result

    async def _deliberate(self) -> StepResult:
        await self._emit_event(
            "step.started", {"step": "deliberate", "session_id": self.session_id}
        )
        try:
            goal_analysis = {
                "complexity": await self._assess_complexity(self.current_state.current_goal),
                "constraints": len(self.current_state.constraints),
                "available_context": len(
                    self.current_state.context.get("relevant_patterns", [])
                ),
                "risk_level": await self._assess_risk_level(),
            }
            self.current_state.artifacts["goal_analysis"] = goal_analysis
            result = StepResult(success=True, output=goal_analysis, step_type="deliberation")
            await self._emit_event(
                "step.completed",
                {
                    "step": "deliberate",
                    "result": asdict(result),
                    "session_id": self.session_id,
                },
            )
            return result
        except Exception as exc:
            result = StepResult(success=False, output=None, error=str(exc), step_type="deliberation")
            await self._emit_event(
                "step.failed",
                {
                    "step": "deliberate",
                    "result": asdict(result),
                    "session_id": self.session_id,
                },
            )
            return result

    async def _plan(self) -> StepResult:
        await self._emit_event(
            "step.started", {"step": "plan", "session_id": self.session_id}
        )
        try:
            plan = await self.planner.generate_plan(
                goal=self.current_state.current_goal,
                constraints=self.current_state.constraints,
                context=self.current_state.context,
            )
            approved_steps: List[Dict[str, Any]] = []
            revised_steps: List[Dict[str, Any]] = []
            for step in plan:
                approved, reasoning = await self.constitutional_reasoner.evaluate_action(
                    action=step,
                    context={
                        "user_intent": self.current_state.current_goal,
                        "step_context": self.current_state.context,
                        "risk_level": self.current_state.artifacts.get("goal_analysis", {}).get(
                            "risk_level", "medium"
                        ),
                    },
                )
                if approved:
                    approved_steps.append(step)
                    continue
                revised_step = await self._revise_step(step, reasoning)
                if revised_step:
                    revised_steps.append(revised_step)
                else:
                    self.logger.warning("Step rejected and not revised: %s", step)
            final_plan = approved_steps + revised_steps
            self.current_state.current_plan = final_plan
            result = StepResult(
                success=bool(final_plan),
                output={
                    "plan_steps": len(final_plan),
                    "revised_steps": len(revised_steps),
                },
                step_type="planning",
            )
            await self._emit_event(
                "step.completed",
                {
                    "step": "plan",
                    "result": asdict(result),
                    "session_id": self.session_id,
                    "plan_details": {
                        "total_steps": len(final_plan),
                        "revised_count": len(revised_steps),
                    },
                },
            )
            return result
        except Exception as exc:
            result = StepResult(success=False, output=None, error=str(exc), step_type="planning")
            await self._emit_event(
                "step.failed",
                {
                    "step": "plan",
                    "result": asdict(result),
                    "session_id": self.session_id,
                },
            )
            return result

    async def _act(self) -> StepResult:
        await self._emit_event(
            "step.started", {"step": "act", "session_id": self.session_id}
        )
        try:
            executed: List[StepResult] = []
            for step in self.current_state.current_plan:
                step_result = await self._execute_step(step)
                executed.append(step_result)
                self.current_state.executed_steps.append(
                    {
                        "step": step,
                        "result": asdict(step_result) if step_result else None,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )
                if step_result and not step_result.success:
                    self.logger.warning("Step failed: %s", step.get("action"))
                    break
            success = all(result.success for result in executed if result)
            result = StepResult(
                success=success,
                output={
                    "executed_steps": len(executed),
                    "results": [asdict(res) for res in executed if res],
                },
                step_type="action",
            )
            await self._emit_event(
                "step.completed",
                {
                    "step": "act",
                    "result": asdict(result),
                    "session_id": self.session_id,
                },
            )
            return result
        except Exception as exc:
            result = StepResult(success=False, output=None, error=str(exc), step_type="action")
            await self._emit_event(
                "step.failed",
                {
                    "step": "act",
                    "result": asdict(result),
                    "session_id": self.session_id,
                },
            )
            return result

    async def _reflect(self) -> StepResult:
        await self._emit_event(
            "step.started", {"step": "reflect", "session_id": self.session_id}
        )
        try:
            successes = [
                step
                for step in self.current_state.executed_steps
                if step.get("result", {}).get("success")
            ]
            failures = [
                step
                for step in self.current_state.executed_steps
                if not step.get("result", {}).get("success")
            ]
            progress = await self._calculate_goal_progress()
            insights = [
                f"Completed {len(successes)} steps successfully",
                f"Encountered {len(failures)} failures",
                f"Goal progress: {progress:.1f}%",
            ]
            for failure in failures:
                result = failure.get("result", {})
                insights.append(
                    f"Step {failure['step'].get('action')} failed: {result.get('error')}"
                )
            self.current_state.reflections.extend(insights)
            result = StepResult(success=True, output={"insights": insights}, step_type="reflection")
            await self._emit_event(
                "step.completed",
                {
                    "step": "reflect",
                    "result": asdict(result),
                    "session_id": self.session_id,
                },
            )
            return result
        except Exception as exc:
            result = StepResult(success=False, output=None, error=str(exc), step_type="reflection")
            await self._emit_event(
                "step.failed",
                {
                    "step": "reflect",
                    "result": asdict(result),
                    "session_id": self.session_id,
                },
            )
            return result

    async def _execute_step(self, step: Dict[str, Any]) -> StepResult:
        ability: Ability | None = self.ability_registry.get_ability(step.get("action", ""))
        if not ability:
            return StepResult(success=False, output=None, error=f"Unknown ability: {step.get('action')}")
        try:
            approved, reasoning = await self.constitutional_reasoner.evaluate_action(
                action=step,
                context={
                    "user_intent": self.current_state.current_goal,
                    "ability_metadata": asdict(ability.metadata),
                    "step_context": self.current_state.context,
                },
            )
            if not approved:
                return StepResult(
                    success=False,
                    output=None,
                    error=f"Constitutional violation: {reasoning}",
                    constitutional_approved=False,
                    reasoning=reasoning,
                )
            output = await ability.execute(step.get("parameters", {}), self.current_state.context)
            return StepResult(success=True, output=output, reasoning="Step executed successfully")
        except Exception as exc:
            return StepResult(success=False, output=None, error=str(exc))

    async def _revise_step(self, step: Dict[str, Any], rejection_reason: str) -> Optional[Dict[str, Any]]:
        revised_step = dict(step)
        if "code_execution" in rejection_reason.lower():
            parameters = step.get("parameters", {})
            revised_step["parameters"] = {
                key: value
                for key, value in parameters.items()
                if "exec" not in key.lower() and "eval" not in key.lower()
            }
        approved, _ = await self.constitutional_reasoner.evaluate_action(
            action=revised_step,
            context={
                "user_intent": self.current_state.current_goal,
                "step_context": self.current_state.context,
            },
        )
        return revised_step if approved else None

    async def _is_goal_achieved(self) -> bool:
        if not self.current_state or not self.current_state.executed_steps:
            return False
        recent = self.current_state.executed_steps[-3:]
        return all(step.get("result", {}).get("success") for step in recent)

    async def _should_terminate(self) -> bool:
        if not self.current_state:
            return True
        if len(self.current_state.executed_steps) >= 20:
            return True
        return await self._is_goal_achieved()

    async def _calculate_goal_progress(self) -> float:
        if not self.current_state or not self.current_state.current_plan:
            return 0.0
        completed = len(
            [step for step in self.current_state.executed_steps if step.get("result", {}).get("success")]
        )
        total = max(len(self.current_state.current_plan), 1)
        return (completed / total) * 100

    async def _compile_final_result(self) -> Dict[str, Any]:
        if not self.current_state:
            return {
                "session_id": self.session_id,
                "goal": None,
                "success": False,
                "iterations": 0,
                "reflections": [],
                "artifacts": {},
                "final_state": {},
            }
        return {
            "session_id": self.session_id,
            "goal": self.current_state.current_goal,
            "success": await self._is_goal_achieved(),
            "iterations": len(self.current_state.executed_steps),
            "reflections": self.current_state.reflections,
            "artifacts": self.current_state.artifacts,
            "final_state": asdict(self.current_state),
        }

    async def _persist_session_to_kg(self, final_result: Dict[str, Any]) -> None:
        try:
            session_atom = await self.knowledge_graph.create_atom(
                "agent_session",
                final_result,
                metadata={
                    "session_id": self.session_id,
                    "goal": self.current_state.current_goal if self.current_state else None,
                    "success": final_result.get("success"),
                    "iterations": final_result.get("iterations"),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            for index, step in enumerate(self.current_state.executed_steps if self.current_state else []):
                step_atom = await self.knowledge_graph.create_atom(
                    "execution_step",
                    step,
                    metadata={
                        "session_id": self.session_id,
                        "step_index": index,
                        "success": step.get("result", {}).get("success", False),
                    },
                )
                await self.knowledge_graph.create_bond(
                    source_id=session_atom["id"],
                    target_id=step_atom["id"],
                    bond_type="contains",
                    strength=1.0,
                )
        except Exception as exc:
            self.logger.error("Failed to persist session to KG: %s", exc)

    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        if hasattr(self.event_bus, "emit"):
            await self.event_bus.emit(
                event_type,
                data=data,
                source_plugin="core_agent_loop",
                session_id=self.session_id,
                timestamp=datetime.now(UTC),
            )

    async def _assess_complexity(self, goal: str) -> str:
        word_count = len(goal.split())
        if word_count < 10:
            return "simple"
        if word_count < 25:
            return "medium"
        return "complex"

    async def _assess_risk_level(self) -> str:
        risky_keywords = ["delete", "modify", "execute", "update", "deploy"]
        goal_lower = self.current_state.current_goal.lower()
        if any(keyword in goal_lower for keyword in risky_keywords):
            return "high"
        if "create" in goal_lower or "add" in goal_lower:
            return "medium"
        return "low"


async def create_core_agent_loop(
    constitutional_reasoner: Any,
    event_bus: Any,
    knowledge_graph: Any,
    planner: Planner,
    ability_registry: AbilityRegistry,
) -> CoreAgentLoop:
    """Factory helper for creating an agent loop."""
    return CoreAgentLoop(
        constitutional_reasoner, event_bus, knowledge_graph, planner, ability_registry
    )
