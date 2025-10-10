"""LADDER EventBus integration adapter with KG enhancement."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.core.event_bus import EventBus
from src.core.events import BaseEvent
from src.ladder.graph.task_graph import TaskGraph
from src.ladder.planner import ExecutionResult, LadderPlanner


class PlanningMode(Enum):
    """Planning mode for LADDER integration."""

    SHADOW = "shadow"  # Planning without execution
    ACTIVE = "active"  # Full planning and execution


@dataclass
class LadderIntegrationConfig:
    """Configuration for LADDER integration."""

    max_concurrent_tasks: int = 3
    task_timeout: float = 300.0
    success_reward: float = 1.0
    failure_penalty: float = -0.5
    planning_mode: PlanningMode = PlanningMode.SHADOW


@dataclass
class IntegrationMetrics:
    """Metrics for LADDER integration performance."""

    total_plans: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_planning_time: float = 0.0
    average_execution_time: float = 0.0


class LadderAdapter:
    """
    LADDER EventBus integration adapter.

    This adapter provides LADDER planning capabilities with full EventBus
    integration for event-driven orchestration.
    """

    def __init__(
        self,
        planner: LadderPlanner,
        event_bus: EventBus,
        source_plugin: str = "ladder_adapter",
        config: LadderIntegrationConfig | None = None,
    ):
        """Initialize the LADDER adapter.

        Args:
            planner: LADDER planner instance
            event_bus: Event bus for system communication
            source_plugin: Plugin identifier for events
            config: Integration configuration
        """
        self.planner = planner
        self.event_bus = event_bus
        self.source_plugin = source_plugin
        self.config = config or LadderIntegrationConfig()

        # State management
        self.active_plans: dict[str, TaskGraph] = {}
        self.execution_contexts: dict[str, dict[str, Any]] = {}
        self.metrics = IntegrationMetrics()
        self._initialized = False

    async def setup(self) -> None:
        """Async setup for event subscriptions."""
        if self._initialized:
            return

        # Subscribe to relevant events
        await self._setup_event_subscriptions()
        self._initialized = True

    async def _setup_event_subscriptions(self) -> None:
        """Setup event bus subscriptions."""
        await self.event_bus.subscribe(
            "planning_request", self._handle_planning_request
        )

    async def handle_request(
        self, query: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Handle a planning request and return results."""
        context = context or {}
        session_id = f"session_{int(time.time())}"

        try:
            # Emit planning start event
            await self.event_bus.emit(
                "planning_started",
                source_plugin=self.source_plugin,
                goal=query,
                session_id=session_id,
                context=context,
            )

            start_time = time.time()

            # Create plan using LADDER
            plan = await self.planner.create_plan(
                goal=query,
                context=context,
            )

            # Store plan
            self.active_plans[session_id] = plan

            # Execute plan
            result = await self.planner.execute_plan(plan)

            planning_time = time.time() - start_time
            self._update_metrics(result, planning_time)

            # Emit planning completion event
            await self.event_bus.emit(
                "planning_completed",
                source_plugin=self.source_plugin,
                session_id=session_id,
                success=result.success,
                execution_time=result.execution_time,
                tasks_completed=len(plan.get_all_task_ids()),
            )

            return {
                "status": "success" if result.success else "failed",
                "answer": str(result.result) if result.result else "No result",
                "tasks_created": len(plan.get_all_task_ids()),
                "execution_time_ms": int(planning_time * 1000),
                "task_graph": plan,
                "session_id": session_id,
            }

        except Exception as e:
            # Handle planning/execution errors
            await self.event_bus.emit(
                "planning_error",
                source_plugin=self.source_plugin,
                session_id=session_id,
                error=str(e),
                goal=query,
            )

            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id,
            }

    def _update_metrics(
        self, result: ExecutionResult, planning_time: float
    ) -> None:
        """Update integration metrics."""
        self.metrics.total_plans += 1

        if result.success:
            self.metrics.successful_executions += 1
        else:
            self.metrics.failed_executions += 1

        # Update rolling averages
        total_plans = self.metrics.total_plans
        self.metrics.average_planning_time = (
            self.metrics.average_planning_time * (total_plans - 1)
            + planning_time
        ) / total_plans

        if result.execution_time:
            self.metrics.average_execution_time = (
                self.metrics.average_execution_time * (total_plans - 1)
                + result.execution_time
            ) / total_plans

    async def _handle_planning_request(self, event: BaseEvent) -> None:
        """Handle planning requests from the event bus."""
        # Extract data from BaseEvent
        event_data = event.model_dump() if hasattr(event, "model_dump") else {}

        goal = event_data.get("goal", "")
        session_id = event_data.get("session_id", f"req_{int(time.time())}")
        context = event_data.get("context", {})

        if goal:
            result = await self.handle_request(goal, context)

            # Emit response
            await self.event_bus.emit(
                "planning_result",
                source_plugin=self.source_plugin,
                session_id=session_id,
                success=result.get("status") == "success",
                result=result,
            )

    def get_metrics(self) -> dict[str, Any]:
        """Get current integration metrics."""
        success_rate = 0.0
        if self.metrics.total_plans > 0:
            success_rate = (
                self.metrics.successful_executions / self.metrics.total_plans
            )

        return {
            "total_plans": self.metrics.total_plans,
            "success_rate": success_rate,
            "avg_planning_time": self.metrics.average_planning_time,
            "avg_execution_time": self.metrics.average_execution_time,
            "active_plans": len(self.active_plans),
            "planning_mode": self.config.planning_mode.value,
        }
