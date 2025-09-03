"""Priority engine for task ordering and optimization."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.ladder.graph.task_graph import Task, TaskGraph

from .energy_calculator import EnergyCalculator, TaskEnergy


class PriorityStrategy(Enum):
    """Strategies for task prioritization."""

    ENERGY_ONLY = "energy_only"  # Pure energy-based ordering
    ENERGY_DEPENDENCY = "energy_dependency"  # Energy + dependency constraints
    BALANCED = "balanced"  # Energy + dependencies + resource constraints
    ADAPTIVE = "adaptive"  # Dynamic strategy based on context


@dataclass
class PriorityConfig:
    """Configuration for priority engine."""

    strategy: PriorityStrategy = PriorityStrategy.BALANCED
    energy_threshold: float = 0.8  # Only consider tasks below this energy
    max_parallel_tasks: int = 3  # Maximum tasks to run in parallel
    dependency_strict: bool = True  # Strict dependency ordering
    rebalance_interval: float = 300.0  # Rebalance every 5 minutes
    confidence_threshold: float = 0.3  # Minimum confidence for prioritization


@dataclass
class TaskPriority:
    """Priority information for a task."""

    task_id: str
    priority_score: float  # Higher = higher priority (inverse of energy)
    rank: int  # Position in priority queue (1 = highest)
    energy: TaskEnergy
    can_execute: bool = True  # Whether task can execute now
    blocked_by: list[str] = field(default_factory=list)  # Blocking task IDs
    estimated_start_time: float = 0.0  # When task can start
    reasoning: list[str] = field(default_factory=list)


class PriorityEngine:
    """
    Engine for calculating and managing task priorities.

    Uses energy calculations and dependency analysis to create
    optimal task execution order.
    """

    def __init__(
        self,
        energy_calculator: EnergyCalculator,
        config: PriorityConfig | None = None,
    ):
        """Initialize the priority engine.

        Args:
            energy_calculator: Calculator for task energy
            config: Priority configuration
        """
        self.energy_calculator = energy_calculator
        self.config = config or PriorityConfig()

        # State tracking
        self.last_rebalance: float = 0.0
        self.execution_history: dict[str, float] = {}  # task_id -> completion_time
        self.current_priorities: dict[str, TaskPriority] = {}

    def calculate_priorities(
        self,
        task_graph: TaskGraph,
        context: dict[str, Any] | None = None,
    ) -> list[TaskPriority]:
        """Calculate priorities for all tasks in the graph.

        Args:
            task_graph: Task graph to prioritize
            context: Planning context

        Returns:
            List of TaskPriority sorted by priority (highest first)
        """
        context = context or {}
        all_priorities = []

        # 1. Calculate energy for all tasks
        task_energies = {}
        for task_id in task_graph.get_all_task_ids():
            task = task_graph.get_task(task_id)
            if task is None:
                continue
            if self._should_consider_task(task, context):
                energy = self.energy_calculator.calculate_task_energy(task, context)
                task_energies[task.id] = energy

        # 2. Apply priority strategy
        if self.config.strategy == PriorityStrategy.ENERGY_ONLY:
            priorities = self._prioritize_by_energy_only(task_energies, context)
        elif self.config.strategy == PriorityStrategy.ENERGY_DEPENDENCY:
            priorities = self._prioritize_energy_dependency(
                task_graph, task_energies, context
            )
        elif self.config.strategy == PriorityStrategy.BALANCED:
            priorities = self._prioritize_balanced(task_graph, task_energies, context)
        else:  # ADAPTIVE
            priorities = self._prioritize_adaptive(task_graph, task_energies, context)

        # 3. Assign ranks and update state
        for i, priority in enumerate(priorities):
            priority.rank = i + 1

        self.current_priorities = {p.task_id: p for p in priorities}
        self.last_rebalance = time.time()

        return priorities

    def _should_consider_task(self, task: Task, context: dict[str, Any]) -> bool:
        """Check if task should be considered for prioritization."""
        # Skip completed tasks
        if hasattr(task, "status") and task.status == "completed":
            return False

        # Skip tasks that don't meet confidence threshold
        # (will be checked later when calculating energy)
        return True

    def _prioritize_by_energy_only(
        self,
        task_energies: dict[str, TaskEnergy],
        context: dict[str, Any],
    ) -> list[TaskPriority]:
        """Simple energy-based prioritization."""
        priorities = []

        for task_id, energy in task_energies.items():
            # Skip low-confidence calculations
            if energy.confidence < self.config.confidence_threshold:
                continue

            # Convert energy to priority (inverse relationship)
            priority_score = 1.0 - energy.energy_score

            priority = TaskPriority(
                task_id=task_id,
                priority_score=priority_score,
                rank=0,  # Will be set later
                energy=energy,
                reasoning=[
                    f"Energy-only priority: {priority_score:.3f}",
                    f"Based on energy: {energy.energy_score:.3f}",
                ],
            )
            priorities.append(priority)

        # Sort by priority score (descending)
        priorities.sort(key=lambda p: p.priority_score, reverse=True)
        return priorities

    def _prioritize_energy_dependency(
        self,
        task_graph: TaskGraph,
        task_energies: dict[str, TaskEnergy],
        context: dict[str, Any],
    ) -> list[TaskPriority]:
        """Prioritization considering energy and dependencies."""
        priorities = []
        dependency_map = task_graph.dependencies

        for task_id, energy in task_energies.items():
            if energy.confidence < self.config.confidence_threshold:
                continue

            task = task_graph.get_task(task_id)
            if not task:
                continue

            # Base priority from energy
            base_priority = 1.0 - energy.energy_score

            # Check dependencies
            blocked_by = []
            can_execute = True

            dependencies = dependency_map.get(task_id, [])
            for dep_id in dependencies:
                if dep_id not in self.execution_history:
                    blocked_by.append(dep_id)
                    can_execute = False

            # Adjust priority based on dependencies
            if blocked_by:
                # Reduce priority for blocked tasks
                priority_score = base_priority * 0.5
            else:
                # Boost priority for ready tasks
                priority_score = base_priority * 1.2

            priority = TaskPriority(
                task_id=task_id,
                priority_score=min(1.0, priority_score),
                rank=0,
                energy=energy,
                can_execute=can_execute,
                blocked_by=blocked_by,
                reasoning=[
                    f"Energy+Dependency priority: {priority_score:.3f}",
                    f"Base energy priority: {base_priority:.3f}",
                    (
                        f"Blocked by {len(blocked_by)} dependencies"
                        if blocked_by
                        else "Ready to execute"
                    ),
                ],
            )
            priorities.append(priority)

        # Sort by executability first, then priority
        priorities.sort(key=lambda p: (not p.can_execute, -p.priority_score))
        return priorities

    def _prioritize_balanced(
        self,
        task_graph: TaskGraph,
        task_energies: dict[str, TaskEnergy],
        context: dict[str, Any],
    ) -> list[TaskPriority]:
        """Balanced prioritization with resource constraints."""
        priorities = self._prioritize_energy_dependency(
            task_graph, task_energies, context
        )

        # Apply resource balancing
        available_slots = self.config.max_parallel_tasks
        current_time = time.time()

        for priority in priorities:
            if not priority.can_execute:
                # Estimate when dependencies will complete
                max_dependency_time = 0.0
                for dep_id in priority.blocked_by:
                    dep_completion = self._estimate_task_completion_time(dep_id)
                    max_dependency_time = max(max_dependency_time, dep_completion)

                priority.estimated_start_time = max_dependency_time
            else:
                # Can start immediately if slots available
                if available_slots > 0:
                    priority.estimated_start_time = current_time
                    available_slots -= 1
                else:
                    # Estimate based on when current tasks finish
                    priority.estimated_start_time = (
                        current_time + 300.0
                    )  # 5 min estimate

            # Add resource reasoning
            priority.reasoning.append(
                f"Estimated start time: {priority.estimated_start_time:.0f}"
            )

        return priorities

    def _prioritize_adaptive(
        self,
        task_graph: TaskGraph,
        task_energies: dict[str, TaskEnergy],
        context: dict[str, Any],
    ) -> list[TaskPriority]:
        """Adaptive prioritization based on context."""
        # Choose strategy based on context
        task_count = len(task_energies)
        blocked_count = sum(
            1
            for energy in task_energies.values()
            if energy.metrics.dependency_score > 0.5
        )

        if blocked_count / task_count > 0.6:
            # Many dependencies - use dependency-aware strategy
            return self._prioritize_energy_dependency(
                task_graph, task_energies, context
            )
        elif task_count > 10:
            # Many tasks - use balanced approach
            return self._prioritize_balanced(task_graph, task_energies, context)
        else:
            # Simple case - use energy only
            return self._prioritize_by_energy_only(task_energies, context)

    def _estimate_task_completion_time(self, task_id: str) -> float:
        """Estimate when a task will complete."""
        # Simple estimation - could be enhanced with ML
        if task_id in self.execution_history:
            return self.execution_history[task_id]

        # Default estimate: current time + 15 minutes
        return time.time() + 900.0

    def should_rebalance(self) -> bool:
        """Check if priorities should be recalculated."""
        current_time = time.time()
        return (current_time - self.last_rebalance) > self.config.rebalance_interval

    def get_next_tasks(self, count: int = 1) -> list[TaskPriority]:
        """Get the next N highest priority executable tasks."""
        executable_priorities = [
            p for p in self.current_priorities.values() if p.can_execute
        ]

        # Sort by priority score
        executable_priorities.sort(key=lambda p: p.priority_score, reverse=True)

        return executable_priorities[:count]

    def mark_task_completed(self, task_id: str, completion_time: float | None = None):
        """Mark a task as completed for future priority calculations."""
        if completion_time is None:
            completion_time = time.time()

        self.execution_history[task_id] = completion_time

        # Update blocked tasks
        for priority in self.current_priorities.values():
            if task_id in priority.blocked_by:
                priority.blocked_by.remove(task_id)
                if not priority.blocked_by:
                    priority.can_execute = True

    def get_priority_summary(self) -> dict[str, Any]:
        """Get summary of current priority state."""
        if not self.current_priorities:
            return {"status": "no_priorities_calculated"}

        executable_count = sum(
            1 for p in self.current_priorities.values() if p.can_execute
        )
        blocked_count = len(self.current_priorities) - executable_count

        avg_energy = sum(
            p.energy.energy_score for p in self.current_priorities.values()
        ) / len(self.current_priorities)

        avg_confidence = sum(
            p.energy.confidence for p in self.current_priorities.values()
        ) / len(self.current_priorities)

        return {
            "total_tasks": len(self.current_priorities),
            "executable_tasks": executable_count,
            "blocked_tasks": blocked_count,
            "average_energy": avg_energy,
            "average_confidence": avg_confidence,
            "last_rebalance": self.last_rebalance,
            "strategy": self.config.strategy.value,
        }
