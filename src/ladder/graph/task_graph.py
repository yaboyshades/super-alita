"""TaskGraph system for managing task dependencies and execution in LADDER."""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..models.task import Task, TaskStatus


class GraphValidationError(Exception):
    """Raised when task graph validation fails."""

    pass


class ExecutionStrategy(Enum):
    """Task execution strategies."""

    SEQUENTIAL = "sequential"
    PARALLEL_SAFE = "parallel_safe"
    PARALLEL_AGGRESSIVE = "parallel_aggressive"


@dataclass
class TaskGraphMetrics:
    """Metrics for task graph execution."""

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    pending_tasks: int = 0
    running_tasks: int = 0
    total_energy: float = 0.0
    consumed_energy: float = 0.0
    start_time: float | None = None
    end_time: float | None = None

    @property
    def completion_rate(self) -> float:
        """Calculate completion percentage."""
        if self.total_tasks == 0:
            return 0.0
        return self.completed_tasks / self.total_tasks

    @property
    def energy_efficiency(self) -> float:
        """Calculate energy efficiency (completed/consumed)."""
        if self.consumed_energy == 0:
            return 0.0
        return self.completed_tasks / self.consumed_energy

    @property
    def execution_time(self) -> float:
        """Calculate total execution time."""
        if not self.start_time:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


class TaskGraph:
    """
    Manages a directed acyclic graph (DAG) of tasks with dependencies.

    Provides topological sorting, cycle detection, and parallel execution
    planning for hierarchical task structures.
    """

    def __init__(self, name: str = "TaskGraph"):
        """Initialize empty task graph."""
        self.name = name
        self.tasks: dict[str, Task] = {}
        self.dependencies: dict[str, set[str]] = defaultdict(set)
        self.dependents: dict[str, set[str]] = defaultdict(set)
        self.metrics = TaskGraphMetrics()
        self._execution_order: list[str] | None = None
        self._parallel_groups: list[list[str]] | None = None

    def add_task(self, task: Task) -> None:
        """Add a task to the graph."""
        if task.id in self.tasks:
            raise ValueError(f"Task {task.id} already exists in graph")

        self.tasks[task.id] = task
        self.metrics.total_tasks += 1
        self.metrics.total_energy += task.energy

        # Add dependencies from task
        for dep_id in task.dependencies:
            self.add_dependency(task.id, dep_id)

        # Invalidate cached execution orders
        self._execution_order = None
        self._parallel_groups = None

    def add_dependency(self, task_id: str, depends_on: str) -> None:
        """Add a dependency relationship between tasks."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found in graph")
        if depends_on not in self.tasks:
            raise ValueError(f"Dependency task {depends_on} not found in graph")
        if task_id == depends_on:
            raise ValueError("Task cannot depend on itself")

        # Add dependency
        self.dependencies[task_id].add(depends_on)
        self.dependents[depends_on].add(task_id)

        # Update task's dependency list
        task = self.tasks[task_id]
        if depends_on not in task.dependencies:
            task.dependencies.append(depends_on)

        # Check for cycles
        if self._has_cycle():
            # Rollback the dependency
            self.dependencies[task_id].discard(depends_on)
            self.dependents[depends_on].discard(task_id)
            task.dependencies.remove(depends_on)
            raise GraphValidationError(
                f"Adding dependency {task_id} -> {depends_on} creates a cycle"
            )

        # Invalidate cached execution orders
        self._execution_order = None
        self._parallel_groups = None

    def remove_task(self, task_id: str) -> None:
        """Remove a task and all its dependencies."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found in graph")

        task = self.tasks[task_id]

        # Remove from dependencies and dependents
        for dep_id in list(self.dependencies[task_id]):
            self.dependents[dep_id].discard(task_id)

        for dependent_id in list(self.dependents[task_id]):
            self.dependencies[dependent_id].discard(task_id)
            # Also remove from task's dependency list
            if dependent_id in self.tasks:
                dependent_task = self.tasks[dependent_id]
                if task_id in dependent_task.dependencies:
                    dependent_task.dependencies.remove(task_id)

        # Remove from graph
        del self.tasks[task_id]
        del self.dependencies[task_id]
        del self.dependents[task_id]

        # Update metrics
        self.metrics.total_tasks -= 1
        self.metrics.total_energy -= task.energy

        # Invalidate cached execution orders
        self._execution_order = None
        self._parallel_groups = None

    def get_ready_tasks(self) -> list[Task]:
        """Get tasks that are ready to execute (all dependencies completed)."""
        ready_tasks = []

        for task_id, task in self.tasks.items():
            if task.status != TaskStatus.PENDING:
                continue

            # Check if all dependencies are completed
            deps_completed = all(
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in self.dependencies[task_id]
                if dep_id in self.tasks
            )

            if deps_completed:
                ready_tasks.append(task)

        return ready_tasks

    def get_blocked_tasks(self) -> list[Task]:
        """Get tasks that are blocked by dependencies."""
        blocked_tasks = []

        for task_id, task in self.tasks.items():
            if task.status != TaskStatus.PENDING:
                continue

            # Check if any dependencies are not completed
            has_incomplete_deps = any(
                self.tasks[dep_id].status != TaskStatus.COMPLETED
                for dep_id in self.dependencies[task_id]
                if dep_id in self.tasks
            )

            if has_incomplete_deps:
                blocked_tasks.append(task)

        return blocked_tasks

    def topological_sort(self) -> list[str]:
        """Return topologically sorted task IDs."""
        if self._execution_order is not None:
            return self._execution_order.copy()

        # Kahn's algorithm for topological sorting
        in_degree = {task_id: len(self.dependencies[task_id]) for task_id in self.tasks}

        queue = deque([task_id for task_id, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            task_id = queue.popleft()
            result.append(task_id)

            # Reduce in-degree for dependent tasks
            for dependent_id in self.dependents[task_id]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        if len(result) != len(self.tasks):
            raise GraphValidationError("Graph contains cycles")

        self._execution_order = result
        return result.copy()

    def get_parallel_groups(self) -> list[list[str]]:
        """Group tasks that can be executed in parallel."""
        if self._parallel_groups is not None:
            return [group.copy() for group in self._parallel_groups]

        # Get topological order
        topo_order = self.topological_sort()

        # Build parallel execution groups
        groups = []
        remaining = set(topo_order)

        while remaining:
            # Find tasks with no dependencies in remaining set
            current_group = []
            for task_id in topo_order:
                if task_id not in remaining:
                    continue

                # Check if all dependencies are satisfied
                deps_satisfied = all(
                    dep_id not in remaining for dep_id in self.dependencies[task_id]
                )

                if deps_satisfied:
                    current_group.append(task_id)

            if not current_group:
                # This shouldn't happen with a valid DAG
                raise GraphValidationError("Unable to find next parallel group")

            groups.append(current_group)
            remaining -= set(current_group)

        self._parallel_groups = groups
        return [group.copy() for group in groups]

    def estimate_execution_time(
        self, strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL_SAFE
    ) -> float:
        """Estimate total execution time for different strategies."""
        if not self.tasks:
            return 0.0

        if strategy == ExecutionStrategy.SEQUENTIAL:
            # Sum all task energies (assuming energy correlates with time)
            return sum(task.energy for task in self.tasks.values())

        elif strategy == ExecutionStrategy.PARALLEL:
            # Maximum energy in any single task (if all could run in parallel)
            return max(task.energy for task in self.tasks.values())

        else:  # PARALLEL strategy
            # Sum of maximum energy in each parallel group
            parallel_groups = self.get_parallel_groups()
            total_time = 0.0

            for group in parallel_groups:
                group_max_energy = max(self.tasks[task_id].energy for task_id in group)
                total_time += group_max_energy

            return total_time

    def validate(self) -> bool:
        """Validate the task graph for consistency."""
        try:
            # Check for cycles
            if self._has_cycle():
                raise GraphValidationError("Graph contains cycles")

            # Check that all dependencies exist
            for task_id, deps in self.dependencies.items():
                for dep_id in deps:
                    if dep_id not in self.tasks:
                        raise GraphValidationError(
                            f"Task {task_id} depends on non-existent task {dep_id}"
                        )

            # Check consistency between dependencies and dependents
            for task_id, deps in self.dependencies.items():
                for dep_id in deps:
                    if task_id not in self.dependents[dep_id]:
                        raise GraphValidationError(
                            f"Inconsistent dependency: {task_id} -> {dep_id}"
                        )

            return True

        except GraphValidationError:
            return False

    def _has_cycle(self) -> bool:
        """Check if the graph has cycles using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = dict.fromkeys(self.tasks, WHITE)

        def dfs(task_id: str) -> bool:
            if colors[task_id] == GRAY:
                return True  # Back edge found - cycle detected
            if colors[task_id] == BLACK:
                return False  # Already processed

            colors[task_id] = GRAY

            for dep_id in self.dependencies[task_id]:
                if dep_id in colors and dfs(dep_id):
                    return True

            colors[task_id] = BLACK
            return False

        return any(dfs(task_id) for task_id in self.tasks if colors[task_id] == WHITE)

    def update_metrics(self) -> None:
        """Update execution metrics based on current task states."""
        self.metrics.completed_tasks = sum(
            1 for task in self.tasks.values() if task.status == TaskStatus.COMPLETED
        )

        self.metrics.failed_tasks = sum(
            1 for task in self.tasks.values() if task.status == TaskStatus.FAILED
        )

        self.metrics.pending_tasks = sum(
            1 for task in self.tasks.values() if task.status == TaskStatus.PENDING
        )

        self.metrics.running_tasks = sum(
            1 for task in self.tasks.values() if task.status == TaskStatus.IN_PROGRESS
        )

        self.metrics.consumed_energy = sum(
            task.energy
            for task in self.tasks.values()
            if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}
        )

    def get_critical_path(self) -> tuple[list[str], float]:
        """
        Find the critical path (longest path) through the graph.

        Returns:
            Tuple of (task_ids_on_critical_path, total_energy)
        """
        # Topologically sort tasks
        topo_order = self.topological_sort()

        # Calculate longest path to each task
        distances = dict.fromkeys(self.tasks, 0.0)
        predecessors = dict.fromkeys(self.tasks)

        for task_id in topo_order:
            task_energy = self.tasks[task_id].energy

            for dependent_id in self.dependents[task_id]:
                new_distance = distances[task_id] + task_energy
                if new_distance > distances[dependent_id]:
                    distances[dependent_id] = new_distance
                    predecessors[dependent_id] = task_id

        # Find the task with maximum distance (end of critical path)
        max_distance = max(distances.values())
        end_task = max(distances.items(), key=lambda x: x[1])[0]

        # Reconstruct critical path
        critical_path = []
        current = end_task

        while current is not None:
            critical_path.append(current)
            current = predecessors[current]

        critical_path.reverse()

        # Add the energy of the last task
        total_energy = max_distance + self.tasks[end_task].energy

        return critical_path, total_energy

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "name": self.name,
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "dependencies": {
                task_id: list(deps) for task_id, deps in self.dependencies.items()
            },
            "metrics": {
                "total_tasks": self.metrics.total_tasks,
                "completed_tasks": self.metrics.completed_tasks,
                "failed_tasks": self.metrics.failed_tasks,
                "pending_tasks": self.metrics.pending_tasks,
                "running_tasks": self.metrics.running_tasks,
                "total_energy": self.metrics.total_energy,
                "consumed_energy": self.metrics.consumed_energy,
                "completion_rate": self.metrics.completion_rate,
                "energy_efficiency": self.metrics.energy_efficiency,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskGraph":
        """Deserialize graph from dictionary."""
        graph = cls(name=data.get("name", "TaskGraph"))

        # Add tasks first
        for task_data in data.get("tasks", {}).values():
            task = Task.from_dict(task_data)
            # Temporarily clear dependencies to avoid validation issues
            original_deps = task.dependencies.copy()
            task.dependencies = []
            graph.add_task(task)
            task.dependencies = original_deps

        # Add dependencies
        dependencies = data.get("dependencies", {})
        for task_id, deps in dependencies.items():
            for dep_id in deps:
                if task_id in graph.tasks and dep_id in graph.tasks:
                    graph.add_dependency(task_id, dep_id)

        return graph

    def __len__(self) -> int:
        """Return number of tasks in graph."""
        return len(self.tasks)

    def __contains__(self, task_id: str) -> bool:
        """Check if task is in graph."""
        return task_id in self.tasks

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def get_all_task_ids(self) -> list[str]:
        """Get all task IDs in the graph."""
        return list(self.tasks.keys())

    def get_dependencies(self, task_id: str) -> list[str]:
        """Get dependencies for a task."""
        return list(self.dependencies.get(task_id, set()))

    def get_execution_order(self) -> list[list[str]]:
        """Get execution order as parallel groups."""
        return self.get_parallel_groups()

    def has_cycles(self) -> bool:
        """Check if graph has cycles."""
        return self._has_cycle()

    def is_connected(self) -> bool:
        """Check if graph is connected."""
        if not self.tasks:
            return True

        # Start from any task and see if we can reach all others
        start_task = next(iter(self.tasks.keys()))
        visited = set()

        def dfs(task_id: str) -> None:
            if task_id in visited:
                return
            visited.add(task_id)

            # Visit dependencies
            for dep in self.dependencies.get(task_id, set()):
                dfs(dep)

            # Visit dependents
            for tid, deps in self.dependencies.items():
                if task_id in deps:
                    dfs(tid)

        dfs(start_task)
        return len(visited) == len(self.tasks)

    def get_metrics(self) -> TaskGraphMetrics:
        """Get graph metrics."""
        self.update_metrics()
        return self.metrics

    def __repr__(self) -> str:
        """String representation of graph."""
        return (
            f"TaskGraph(name='{self.name}', tasks={len(self.tasks)}, "
            f"completion={self.metrics.completion_rate:.1%})"
        )


def merge_task_graphs(graphs: list[TaskGraph], name: str = "MergedGraph") -> TaskGraph:
    """
    Merge multiple task graphs into a single graph.

    Args:
        graphs: List of TaskGraph instances to merge
        name: Name for the merged graph

    Returns:
        New TaskGraph with all tasks and dependencies
    """
    merged = TaskGraph(name)

    # Add all tasks first
    for graph in graphs:
        for task in graph.tasks.values():
            if task.id not in merged:
                merged.add_task(task)

    # Add all dependencies
    for graph in graphs:
        for task_id, deps in graph.dependencies.items():
            for dep_id in deps:
                if task_id in merged and dep_id in merged:
                    try:
                        merged.add_dependency(task_id, dep_id)
                    except (ValueError, GraphValidationError):
                        # Skip if dependency already exists or would create cycle
                        pass

    return merged
