"""Energy-based prioritization system for LADDER tasks."""

import time
from dataclasses import dataclass
from typing import Any

from src.knowledge_graph import KnowledgeGraphInterface
from src.ladder.graph.task_graph import TaskGraph

from .energy_calculator import EnergyCalculator
from .priority_engine import PriorityConfig, PriorityEngine, TaskPriority


@dataclass
class PrioritizationResult:
    """Result of task prioritization."""

    priorities: list[TaskPriority]
    total_tasks: int
    executable_tasks: int
    blocked_tasks: int
    average_energy: float
    average_confidence: float
    strategy_used: str
    calculation_time: float


class EnergyBasedPrioritizer:
    """
    Complete energy-based prioritization system for LADDER.

    This system combines:
    - Energy calculation using KG historical data
    - Dependency-aware priority ordering
    - Resource-constrained scheduling
    - Adaptive strategy selection
    """

    def __init__(
        self,
        kg_interface: KnowledgeGraphInterface,
        priority_config: PriorityConfig | None = None,
        **energy_calculator_kwargs,
    ):
        """Initialize the energy-based prioritizer.

        Args:
            kg_interface: Knowledge graph for historical data
            priority_config: Configuration for priority engine
            **energy_calculator_kwargs: Arguments for energy calculator
        """
        self.kg_interface = kg_interface

        # Initialize energy calculator
        self.energy_calculator = EnergyCalculator(
            kg_interface=kg_interface, **energy_calculator_kwargs
        )

        # Initialize priority engine
        self.priority_engine = PriorityEngine(
            energy_calculator=self.energy_calculator,
            config=priority_config or PriorityConfig(),
        )

        # State tracking
        self.last_prioritization: PrioritizationResult | None = None
        self.prioritization_history: list[PrioritizationResult] = []

    def prioritize_tasks(
        self,
        task_graph: TaskGraph,
        context: dict[str, Any] | None = None,
        force_recalculate: bool = False,
    ) -> PrioritizationResult:
        """Prioritize all tasks in the task graph.

        Args:
            task_graph: Task graph to prioritize
            context: Planning context
            force_recalculate: Force recalculation even if not needed

        Returns:
            PrioritizationResult with prioritized tasks
        """
        start_time = time.time()
        context = context or {}

        # Check if we need to recalculate
        if not force_recalculate and not self._should_recalculate():
            if self.last_prioritization:
                return self.last_prioritization

        # Calculate priorities
        priorities = self.priority_engine.calculate_priorities(
            task_graph, context
        )

        # Calculate metrics
        executable_count = sum(1 for p in priorities if p.can_execute)
        blocked_count = len(priorities) - executable_count

        avg_energy = (
            sum(p.energy.energy_score for p in priorities) / len(priorities)
            if priorities
            else 0.0
        )

        avg_confidence = (
            sum(p.energy.confidence for p in priorities) / len(priorities)
            if priorities
            else 0.0
        )

        calculation_time = time.time() - start_time

        # Create result
        result = PrioritizationResult(
            priorities=priorities,
            total_tasks=len(priorities),
            executable_tasks=executable_count,
            blocked_tasks=blocked_count,
            average_energy=avg_energy,
            average_confidence=avg_confidence,
            strategy_used=self.priority_engine.config.strategy.value,
            calculation_time=calculation_time,
        )

        # Update state
        self.last_prioritization = result
        self.prioritization_history.append(result)

        # Keep only last 10 results for history
        if len(self.prioritization_history) > 10:
            self.prioritization_history.pop(0)

        return result

    def get_next_tasks(
        self,
        task_graph: TaskGraph,
        count: int = 1,
        context: dict[str, Any] | None = None,
    ) -> list[TaskPriority]:
        """Get the next N highest priority executable tasks.

        Args:
            task_graph: Task graph
            count: Number of tasks to return
            context: Planning context

        Returns:
            List of highest priority executable tasks
        """
        # Ensure we have current priorities
        if not self.last_prioritization or self._should_recalculate():
            self.prioritize_tasks(task_graph, context)

        return self.priority_engine.get_next_tasks(count)

    def mark_task_completed(
        self, task_id: str, completion_time: float | None = None
    ):
        """Mark a task as completed.

        This updates the priority engine's understanding of which
        tasks are done and may unblock dependent tasks.

        Args:
            task_id: ID of completed task
            completion_time: When task was completed (default: now)
        """
        self.priority_engine.mark_task_completed(task_id, completion_time)

    def get_task_energy(self, task_id: str) -> float | None:
        """Get the current energy score for a task.

        Args:
            task_id: Task ID

        Returns:
            Energy score (0-1, lower = higher priority) or None if not found
        """
        if not self.last_prioritization:
            return None

        for priority in self.last_prioritization.priorities:
            if priority.task_id == task_id:
                return priority.energy.energy_score

        return None

    def get_task_priority_info(self, task_id: str) -> TaskPriority | None:
        """Get complete priority information for a task.

        Args:
            task_id: Task ID

        Returns:
            TaskPriority object or None if not found
        """
        if not self.last_prioritization:
            return None

        for priority in self.last_prioritization.priorities:
            if priority.task_id == task_id:
                return priority

        return None

    def get_prioritization_summary(self) -> dict[str, Any]:
        """Get summary of current prioritization state.

        Returns:
            Dictionary with prioritization metrics and state
        """
        if not self.last_prioritization:
            return {"status": "no_prioritization_performed"}

        result = self.last_prioritization
        engine_summary = self.priority_engine.get_priority_summary()

        return {
            "current_prioritization": {
                "total_tasks": result.total_tasks,
                "executable_tasks": result.executable_tasks,
                "blocked_tasks": result.blocked_tasks,
                "average_energy": result.average_energy,
                "average_confidence": result.average_confidence,
                "strategy_used": result.strategy_used,
                "calculation_time": result.calculation_time,
            },
            "engine_state": engine_summary,
            "history": {
                "prioritizations_performed": len(self.prioritization_history),
                "last_recalculation": (
                    self.last_prioritization.calculation_time
                    if self.last_prioritization
                    else 0
                ),
            },
        }

    def explain_task_priority(self, task_id: str) -> dict[str, Any]:
        """Get detailed explanation of why a task has its current priority.

        Args:
            task_id: Task ID to explain

        Returns:
            Dictionary with detailed priority explanation
        """
        priority_info = self.get_task_priority_info(task_id)
        if not priority_info:
            return {"error": f"Task {task_id} not found in current priorities"}

        energy = priority_info.energy

        return {
            "task_id": task_id,
            "priority_rank": priority_info.rank,
            "priority_score": priority_info.priority_score,
            "can_execute": priority_info.can_execute,
            "blocked_by": priority_info.blocked_by,
            "energy_analysis": {
                "energy_score": energy.energy_score,
                "confidence": energy.confidence,
                "metrics": {
                    "effort_score": energy.metrics.effort_score,
                    "success_probability": energy.metrics.success_probability,
                    "pattern_confidence": energy.metrics.pattern_confidence,
                    "complexity_score": energy.metrics.complexity_score,
                    "dependency_score": energy.metrics.dependency_score,
                    "recency_bonus": energy.metrics.recency_bonus,
                    "context_relevance": energy.metrics.context_relevance,
                },
            },
            "reasoning": {
                "energy_reasoning": energy.reasoning,
                "priority_reasoning": priority_info.reasoning,
            },
            "calculated_at": energy.calculated_at,
        }

    def _should_recalculate(self) -> bool:
        """Check if prioritization should be recalculated."""
        return self.priority_engine.should_rebalance()

    def update_configuration(
        self, priority_config: PriorityConfig | None = None, **energy_kwargs
    ):
        """Update prioritizer configuration.

        Args:
            priority_config: New priority configuration
            **energy_kwargs: New energy calculator parameters
        """
        if priority_config:
            self.priority_engine.config = priority_config

        # Update energy calculator weights if provided
        if energy_kwargs:
            for key, value in energy_kwargs.items():
                if hasattr(self.energy_calculator, key):
                    setattr(self.energy_calculator, key, value)

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for the prioritization system.

        Returns:
            Dictionary with performance and effectiveness metrics
        """
        if not self.prioritization_history:
            return {"status": "no_history_available"}

        # Calculate trends
        recent_results = self.prioritization_history[-5:]  # Last 5 results

        avg_calculation_time = sum(
            r.calculation_time for r in recent_results
        ) / len(recent_results)

        avg_energy_trend = [r.average_energy for r in recent_results]
        avg_confidence_trend = [r.average_confidence for r in recent_results]

        return {
            "performance": {
                "average_calculation_time": avg_calculation_time,
                "total_prioritizations": len(self.prioritization_history),
                "energy_calculator_weights": {
                    "effort_weight": self.energy_calculator.effort_weight,
                    "success_weight": self.energy_calculator.success_weight,
                    "complexity_weight": self.energy_calculator.complexity_weight,
                    "context_weight": self.energy_calculator.context_weight,
                },
            },
            "trends": {
                "average_energy": avg_energy_trend,
                "average_confidence": avg_confidence_trend,
            },
            "current_config": {
                "strategy": self.priority_engine.config.strategy.value,
                "energy_threshold": self.priority_engine.config.energy_threshold,
                "max_parallel_tasks": self.priority_engine.config.max_parallel_tasks,
                "confidence_threshold": self.priority_engine.config.confidence_threshold,
            },
        }
