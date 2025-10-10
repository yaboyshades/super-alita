"""KG-enhanced LADDER EventBus integration adapter."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.core.event_bus import EventBus
from src.core.events import BaseEvent
from src.knowledge_graph import KnowledgeGraphAdapter, KnowledgeGraphInterface
from src.ladder.graph.task_graph import TaskGraph
from src.ladder.kg_enhanced_planner import KGEnhancedLadderPlanner
from src.ladder.planner import ExecutionResult, PlannerConfig


class PlanningMode(Enum):
    """Planning mode for LADDER integration."""

    SHADOW = "shadow"  # Planning without execution
    ACTIVE = "active"  # Full planning and execution


@dataclass
class KGLadderIntegrationConfig:
    """Configuration for KG-enhanced LADDER integration."""

    max_concurrent_tasks: int = 3
    task_timeout: float = 300.0
    success_reward: float = 1.0
    failure_penalty: float = -0.5
    planning_mode: PlanningMode = PlanningMode.SHADOW
    enable_kg_learning: bool = True
    kg_query_timeout: float = 5.0


@dataclass
class IntegrationMetrics:
    """Metrics for LADDER integration performance."""

    total_plans: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_planning_time: float = 0.0
    average_execution_time: float = 0.0
    kg_queries_made: int = 0
    kg_patterns_used: int = 0


class KGEnhancedLadderAdapter:
    """
    KG-enhanced LADDER EventBus integration adapter.

    This adapter provides LADDER planning capabilities with Knowledge Graph
    enhancement for better planning context and learning.
    """

    def __init__(
        self,
        kg_interface: KnowledgeGraphInterface,
        event_bus: EventBus,
        source_plugin: str = "kg_ladder_adapter",
        config: KGLadderIntegrationConfig | None = None,
    ):
        """Initialize the KG-enhanced LADDER adapter.

        Args:
            kg_interface: Knowledge graph interface
            event_bus: Event bus for system communication
            source_plugin: Plugin identifier for events
            config: Integration configuration
        """
        self.kg_interface = kg_interface
        self.event_bus = event_bus
        self.source_plugin = source_plugin
        self.config = config or KGLadderIntegrationConfig()

        # Setup KG adapter
        self.kg_adapter = KnowledgeGraphAdapter(
            kg_interface=kg_interface,
            event_bus=event_bus,
            source_plugin=f"{source_plugin}_kg",
        )

        # Setup KG-enhanced planner
        planner_config = PlannerConfig(
            shadow_mode=(self.config.planning_mode == PlanningMode.SHADOW),
            enable_knowledge_graph=True,
            execution_timeout=self.config.task_timeout,
        )

        self.planner = KGEnhancedLadderPlanner(
            kg_adapter=self.kg_adapter, config=planner_config
        )

        # State management
        self.active_plans: dict[str, TaskGraph] = {}
        self.execution_contexts: dict[str, dict[str, Any]] = {}
        self.metrics = IntegrationMetrics()
        self._initialized = False

    async def setup(self) -> None:
        """Async setup for event subscriptions."""
        if self._initialized:
            return

        # Setup KG adapter first
        await self.kg_adapter.setup()

        # Subscribe to relevant events
        await self._setup_event_subscriptions()
        self._initialized = True

    async def _setup_event_subscriptions(self) -> None:
        """Setup event bus subscriptions."""
        await self.event_bus.subscribe(
            "planning_request", self._handle_planning_request
        )
        await self.event_bus.subscribe("kg_learned", self._handle_kg_learned)

    async def handle_request(
        self, query: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Handle a planning request with KG enhancement."""
        context = context or {}
        session_id = f"kg_session_{int(time.time())}"

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

            # Get KG-enhanced context
            enhanced_context = self.planner.get_enhanced_context(
                query, context
            )

            # Create plan using KG-enhanced LADDER
            plan = await self.planner.create_plan(
                goal=query,
                context=enhanced_context,
            )

            # Store plan
            self.active_plans[session_id] = plan

            # Execute plan
            result = await self.planner.execute_plan(plan)

            planning_time = time.time() - start_time
            self._update_metrics(result, planning_time, enhanced_context)

            # Emit planning completion event
            await self.event_bus.emit(
                "planning_completed",
                source_plugin=self.source_plugin,
                session_id=session_id,
                success=result.success,
                execution_time=result.execution_time,
                tasks_completed=len(plan.get_all_task_ids()),
                kg_enhanced=True,
                kg_patterns_used=enhanced_context.get("patterns_found", 0),
            )

            return {
                "status": "success" if result.success else "failed",
                "answer": str(result.result) if result.result else "No result",
                "tasks_created": len(plan.get_all_task_ids()),
                "execution_time_ms": int(planning_time * 1000),
                "task_graph": plan,
                "session_id": session_id,
                "kg_enhanced": True,
                "kg_context": {
                    "domain": enhanced_context.get("domain", "general"),
                    "patterns_found": enhanced_context.get(
                        "patterns_found", 0
                    ),
                    "similar_goals": enhanced_context.get(
                        "similar_goals_found", 0
                    ),
                    "historical_outcomes": enhanced_context.get(
                        "historical_outcomes", 0
                    ),
                },
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
                "kg_enhanced": False,
            }

    def _update_metrics(
        self,
        result: ExecutionResult,
        planning_time: float,
        enhanced_context: dict[str, Any],
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

        # KG-specific metrics
        if enhanced_context.get("kg_enhanced"):
            self.metrics.kg_queries_made += 1
            self.metrics.kg_patterns_used += enhanced_context.get(
                "patterns_found", 0
            )

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

    async def _handle_kg_learned(self, event: BaseEvent) -> None:
        """Handle KG learning events."""
        event_data = event.model_dump() if hasattr(event, "model_dump") else {}

        # Log KG learning for debugging
        event_data.get("session_id", "")
        event_data.get("success", False)
        event_data.get("domain", "general")

        # Could add additional learning logic here
        pass

    def get_metrics(self) -> dict[str, Any]:
        """Get current integration metrics with KG statistics."""
        success_rate = 0.0
        if self.metrics.total_plans > 0:
            success_rate = (
                self.metrics.successful_executions / self.metrics.total_plans
            )

        kg_usage_rate = 0.0
        if self.metrics.total_plans > 0:
            kg_usage_rate = (
                self.metrics.kg_queries_made / self.metrics.total_plans
            )

        base_metrics = {
            "total_plans": self.metrics.total_plans,
            "success_rate": success_rate,
            "avg_planning_time": self.metrics.average_planning_time,
            "avg_execution_time": self.metrics.average_execution_time,
            "active_plans": len(self.active_plans),
            "planning_mode": self.config.planning_mode.value,
        }

        kg_metrics = {
            "kg_enhanced": True,
            "kg_queries_made": self.metrics.kg_queries_made,
            "kg_patterns_used": self.metrics.kg_patterns_used,
            "kg_usage_rate": kg_usage_rate,
            "kg_statistics": self.planner.get_kg_statistics(),
        }

        return {**base_metrics, **kg_metrics}
