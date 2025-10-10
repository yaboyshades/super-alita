"""Energy-enhanced LADDER adapter with prioritization integration."""

import time
from dataclasses import dataclass
from typing import Any

from src.core.event_bus import EventBus
from src.knowledge_graph import KnowledgeGraphInterface
from src.ladder.integration.kg_enhanced_adapter import (
    IntegrationMetrics,
    KGEnhancedLadderAdapter,
    KGLadderIntegrationConfig,
)

from .energy_enhanced_planner import EnergyEnhancedLadderPlanner
from .priority_engine import PriorityConfig, PriorityStrategy


@dataclass
class EnergyIntegrationConfig(KGLadderIntegrationConfig):
    """Configuration for energy-enhanced LADDER integration."""

    # Energy prioritization settings
    priority_strategy: PriorityStrategy = PriorityStrategy.BALANCED
    energy_threshold: float = 0.8
    rebalance_interval: float = 300.0  # 5 minutes
    confidence_threshold: float = 0.3

    # Energy calculator weights
    effort_weight: float = 0.3
    success_weight: float = 0.4
    complexity_weight: float = 0.2
    context_weight: float = 0.1


@dataclass
class EnergyMetrics(IntegrationMetrics):
    """Extended metrics including energy prioritization."""

    total_prioritizations: int = 0
    average_prioritization_time: float = 0.0
    energy_improvements: int = 0
    high_priority_tasks_completed: int = 0
    low_energy_tasks_completed: int = 0
    prioritization_accuracy: float = 0.0


class EnergyEnhancedLadderAdapter(KGEnhancedLadderAdapter):
    """
    Energy-enhanced LADDER EventBus integration adapter.

    This adapter extends KG-enhanced capabilities with energy-based
    task prioritization for optimal execution ordering.
    """

    def __init__(
        self,
        kg_interface: KnowledgeGraphInterface,
        event_bus: EventBus,
        source_plugin: str = "energy_ladder_adapter",
        config: EnergyIntegrationConfig | None = None,
    ):
        """Initialize the energy-enhanced LADDER adapter.

        Args:
            kg_interface: Knowledge graph interface
            event_bus: Event bus for system communication
            source_plugin: Plugin identifier for events
            config: Energy integration configuration
        """
        # Initialize with energy config
        self.energy_config = config or EnergyIntegrationConfig()

        # Initialize parent with base config
        super().__init__(
            kg_interface=kg_interface,
            event_bus=event_bus,
            source_plugin=source_plugin,
            config=self.energy_config,
        )

        # Replace planner with energy-enhanced version
        priority_config = PriorityConfig(
            strategy=self.energy_config.priority_strategy,
            energy_threshold=self.energy_config.energy_threshold,
            max_parallel_tasks=self.energy_config.max_concurrent_tasks,
            dependency_strict=True,
            rebalance_interval=self.energy_config.rebalance_interval,
            confidence_threshold=self.energy_config.confidence_threshold,
        )

        self.planner = EnergyEnhancedLadderPlanner(
            kg_interface=kg_interface,
            kg_adapter=self.kg_adapter,
            priority_config=priority_config,
            config=self.planner.config,  # Keep existing planner config
            # Energy calculator weights
            effort_weight=self.energy_config.effort_weight,
            success_weight=self.energy_config.success_weight,
            complexity_weight=self.energy_config.complexity_weight,
            context_weight=self.energy_config.context_weight,
        )

        # Enhanced metrics
        self.metrics = EnergyMetrics()

    async def handle_request(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Handle planning request with energy-enhanced prioritization."""
        start_time = time.time()
        context = context or {}

        # Add energy prioritization to context
        context["energy_prioritization"] = {
            "enabled": True,
            "strategy": self.energy_config.priority_strategy.value,
            "confidence_threshold": self.energy_config.confidence_threshold,
        }

        # Call parent's request handling
        result = await super().handle_request(goal, context, session_id)

        # Add energy-specific metrics to result
        if "success" in result and result["success"]:
            energy_status = self.planner.get_prioritization_status()
            result["energy_prioritization"] = energy_status

            # Get task recommendations
            if hasattr(result, "plan") and result["plan"]:
                recommendations = self.planner.get_next_recommended_tasks(
                    result["plan"], count=3
                )
                result["recommended_tasks"] = recommendations

        # Update metrics
        processing_time = time.time() - start_time
        self._update_energy_metrics(result, processing_time)

        return result

    def _update_energy_metrics(
        self, result: dict[str, Any], processing_time: float
    ):
        """Update energy-specific metrics."""
        # Update base metrics
        self.metrics.total_plans += 1

        if result.get("success", False):
            self.metrics.successful_executions += 1
        else:
            self.metrics.failed_executions += 1

        # Update energy metrics
        if "energy_prioritization" in result:
            energy_data = result["energy_prioritization"]

            if "current_prioritization" in energy_data:
                current = energy_data["current_prioritization"]
                self.metrics.total_prioritizations += 1

                # Update average prioritization time
                new_time = current.get("calculation_time", 0.0)
                current_avg = self.metrics.average_prioritization_time
                total_count = self.metrics.total_prioritizations

                self.metrics.average_prioritization_time = (
                    current_avg * (total_count - 1) + new_time
                ) / total_count

        # Update planning time
        current_plan_avg = self.metrics.average_planning_time
        total_plans = self.metrics.total_plans

        self.metrics.average_planning_time = (
            current_plan_avg * (total_plans - 1) + processing_time
        ) / total_plans

    async def handle_task_completion(
        self,
        task_id: str,
        success: bool,
        context: dict[str, Any] | None = None,
    ):
        """Handle task completion with energy learning."""
        context = context or {}

        # Get task priority info before marking complete
        priority_info = self.planner.prioritizer.get_task_priority_info(
            task_id
        )

        # Mark task as completed in prioritizer
        completion_time = time.time()
        self.planner.prioritizer.mark_task_completed(task_id, completion_time)

        # Update energy-specific completion metrics
        if priority_info and success:
            # Track high-priority task completions
            if priority_info.priority_score > 0.7:
                self.metrics.high_priority_tasks_completed += 1

            # Track low-energy task completions
            if priority_info.energy.energy_score < 0.3:
                self.metrics.low_energy_tasks_completed += 1

            # Simple accuracy calculation
            predicted_success = (
                priority_info.energy.metrics.success_probability
            )
            actual_success = 1.0 if success else 0.0

            # Update prioritization accuracy (simple moving average)
            accuracy_error = abs(predicted_success - actual_success)
            current_accuracy = self.metrics.prioritization_accuracy
            total_completions = (
                self.metrics.high_priority_tasks_completed
                + self.metrics.low_energy_tasks_completed
            )

            if total_completions > 0:
                self.metrics.prioritization_accuracy = (
                    current_accuracy * (total_completions - 1)
                    + (1.0 - accuracy_error)
                ) / total_completions

        # Emit task completion event
        await self.event_bus.emit(
            "task_completed",
            {
                "task_id": task_id,
                "success": success,
                "completion_time": completion_time,
                "session_id": context.get("session_id"),
                "priority_info": (
                    {
                        "priority_score": priority_info.priority_score,
                        "energy_score": priority_info.energy.energy_score,
                        "confidence": priority_info.energy.confidence,
                    }
                    if priority_info
                    else None
                ),
            },
            source=self.source_plugin,
        )

    def get_energy_summary(self) -> dict[str, Any]:
        """Get comprehensive energy prioritization summary."""
        base_summary = self.get_status()
        energy_status = self.planner.get_prioritization_status()

        return {
            "adapter_status": base_summary,
            "energy_prioritization": energy_status,
            "energy_metrics": {
                "total_prioritizations": self.metrics.total_prioritizations,
                "average_prioritization_time": self.metrics.average_prioritization_time,
                "energy_improvements": self.metrics.energy_improvements,
                "high_priority_completed": self.metrics.high_priority_tasks_completed,
                "low_energy_completed": self.metrics.low_energy_tasks_completed,
                "prioritization_accuracy": self.metrics.prioritization_accuracy,
            },
            "configuration": {
                "strategy": self.energy_config.priority_strategy.value,
                "energy_threshold": self.energy_config.energy_threshold,
                "rebalance_interval": self.energy_config.rebalance_interval,
                "confidence_threshold": self.energy_config.confidence_threshold,
                "energy_weights": {
                    "effort": self.energy_config.effort_weight,
                    "success": self.energy_config.success_weight,
                    "complexity": self.energy_config.complexity_weight,
                    "context": self.energy_config.context_weight,
                },
            },
        }

    def explain_task_priority(self, task_id: str) -> dict[str, Any]:
        """Get detailed explanation of task prioritization."""
        return self.planner.explain_task_priority(task_id)

    async def rebalance_priorities(
        self, context: dict[str, Any] | None = None
    ) -> bool:
        """Manually trigger priority rebalancing."""
        if not hasattr(self, "active_plans") or not self.active_plans:
            return False

        rebalanced = False
        for plan_id, plan in self.active_plans.items():
            if self.planner.rebalance_priorities(plan, context):
                rebalanced = True

                # Emit rebalancing event
                await self.event_bus.emit(
                    "priorities_rebalanced",
                    {
                        "plan_id": plan_id,
                        "rebalance_time": time.time(),
                        "context": context,
                    },
                    source=self.source_plugin,
                )

        return rebalanced

    def get_next_recommended_tasks(
        self, plan_id: str | None = None, count: int = 3
    ) -> list[dict[str, Any]]:
        """Get recommended next tasks with priority explanations."""
        if plan_id and plan_id in self.active_plans:
            plan = self.active_plans[plan_id]
            return self.planner.get_next_recommended_tasks(plan, count)

        # If no specific plan, get from all active plans
        all_recommendations = []
        for plan in self.active_plans.values():
            recommendations = self.planner.get_next_recommended_tasks(
                plan, count
            )
            all_recommendations.extend(recommendations)

        # Sort by priority score and return top N
        all_recommendations.sort(
            key=lambda x: x["priority_score"], reverse=True
        )
        return all_recommendations[:count]

    def get_energy_performance_metrics(self) -> dict[str, Any]:
        """Get detailed performance metrics for energy prioritization."""
        planner_metrics = self.planner.prioritizer.get_performance_metrics()

        return {
            "prioritization_performance": planner_metrics,
            "adapter_metrics": {
                "total_requests": self.metrics.total_plans,
                "success_rate": (
                    self.metrics.successful_executions
                    / self.metrics.total_plans
                    if self.metrics.total_plans > 0
                    else 0.0
                ),
                "average_request_time": self.metrics.average_planning_time,
                "energy_accuracy": self.metrics.prioritization_accuracy,
            },
            "task_completion_analysis": {
                "high_priority_completed": self.metrics.high_priority_tasks_completed,
                "low_energy_completed": self.metrics.low_energy_tasks_completed,
                "total_completions": (
                    self.metrics.high_priority_tasks_completed
                    + self.metrics.low_energy_tasks_completed
                ),
            },
        }
