"""Core LADDER planner orchestrator for hierarchical task planning."""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .decomposers.base import DefaultLLMDecomposer, LadderDecomposer
from .graph.task_graph import ExecutionStrategy, TaskGraph, TaskGraphMetrics
from .models.task import Task, TaskStatus, TaskTemplate
from .policies.bandit import BanditPolicy, UCB1Policy


@dataclass
class PlannerConfig:
    """Configuration for LADDER planner behavior."""

    max_decomposition_depth: int = 5
    max_concurrent_tasks: int = 4
    execution_timeout: float = 300.0  # 5 minutes
    shadow_mode: bool = False
    enable_energy_optimization: bool = True
    bandit_exploration_rate: float = 0.1
    task_retry_limit: int = 3
    enable_knowledge_graph: bool = False
    debug_mode: bool = False


@dataclass
class ExecutionContext:
    """Context for task execution including shared state and resources."""

    variables: dict[str, Any] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    tools_used: set[str] = field(default_factory=set)
    execution_history: list[dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)

    def add_fact(self, key: str, value: Any) -> None:
        """Add a fact to the knowledge base."""
        self.facts[key] = value

    def get_fact(self, key: str, default: Any = None) -> Any:
        """Retrieve a fact from the knowledge base."""
        return self.facts.get(key, default)

    def set_variable(self, key: str, value: Any) -> None:
        """Set a variable in the execution context."""
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a variable from the execution context."""
        return self.variables.get(key, default)


@dataclass
class ExecutionResult:
    """Result of executing a task or plan."""

    task_id: str
    success: bool
    result: Any = None
    error: str | None = None
    execution_time: float = 0.0
    tools_used: list[str] = field(default_factory=list)
    subtask_results: list["ExecutionResult"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class LadderPlanner:
    """Main LADDER planner orchestrator for hierarchical task planning and execution."""

    def __init__(
        self,
        decomposer: LadderDecomposer | None = None,
        bandit_policy: BanditPolicy | None = None,
        config: PlannerConfig | None = None,
    ):
        """Initialize the LADDER planner.

        Args:
            decomposer: Task decomposition strategy
            bandit_policy: Multi-armed bandit for tool selection
            config: Planner configuration options
        """
        self.decomposer = decomposer or DefaultLLMDecomposer()
        self.bandit_policy = bandit_policy or UCB1Policy(tools=["default_tool"])
        self.config = config or PlannerConfig()

        # Execution state
        self.task_graph = TaskGraph()
        self.execution_context = ExecutionContext()
        self.active_tasks: set[str] = set()
        self.completed_tasks: set[str] = set()

        # Metrics
        self.total_tasks_created = 0
        self.total_tasks_completed = 0
        self.total_execution_time = 0.0
        self.decomposition_history: list[dict[str, Any]] = []

    async def create_plan(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
        template: TaskTemplate | None = None,
    ) -> TaskGraph:
        """Create a hierarchical plan for achieving the given goal.

        Args:
            goal: High-level goal description
            context: Additional context for planning
            template: Optional task template to use

        Returns:
            TaskGraph representing the execution plan
        """
        # Create root task
        if template:
            root_task = Task.create_from_template(
                template,
                description=goal,
                context=context or {},
            )
        else:
            root_task = Task(
                description=goal,
                metadata=context or {},
                energy=2.0,  # Default planning energy
            )

        # Initialize task graph
        self.task_graph = TaskGraph()
        self.task_graph.add_task(root_task)

        # Decompose the goal hierarchically
        await self._decompose_task_hierarchically(root_task.id, depth=0)

        # Optimize task order and dependencies
        if self.config.enable_energy_optimization:
            self._optimize_task_ordering()

        # Validate the plan
        self._validate_plan()

        return self.task_graph

    async def _decompose_task_hierarchically(self, task_id: str, depth: int) -> None:
        """Recursively decompose a task into subtasks.

        Args:
            task_id: ID of task to decompose
            depth: Current decomposition depth
        """
        if depth >= self.config.max_decomposition_depth:
            return

        task = self.task_graph.get_task(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return

        # Check if task needs decomposition
        if not self._should_decompose_task(task):
            return

        # Decompose the task
        decomposition_result = await self.decomposer.decompose(
            task, self.execution_context.facts
        )

        subtasks = decomposition_result.subtasks

        if not subtasks:
            return

        # Add subtasks to graph
        for subtask in subtasks:
            self.task_graph.add_task(subtask)
            self.task_graph.add_dependency(subtask.id, task_id)
            self.total_tasks_created += 1

        # Record decomposition
        self.decomposition_history.append(
            {
                "parent_task_id": task_id,
                "subtask_ids": [st.id for st in subtasks],
                "depth": depth,
                "timestamp": datetime.now(),
            }
        )

        # Recursively decompose subtasks
        for subtask in subtasks:
            await self._decompose_task_hierarchically(subtask.id, depth + 1)

    def _should_decompose_task(self, task: Task) -> bool:
        """Determine if a task should be decomposed further.

        Args:
            task: Task to evaluate

        Returns:
            True if task should be decomposed
        """
        # Don't decompose if already has subtasks
        dependencies = self.task_graph.get_dependencies(task.id)
        if dependencies:
            return False

        # Check if task is atomic (simple enough to execute directly)
        return not self._is_atomic_task(task)

    def _is_atomic_task(self, task: Task) -> bool:
        """Check if a task is atomic (cannot be decomposed further).

        Args:
            task: Task to check

        Returns:
            True if task is atomic
        """
        # Simple heuristics for atomic tasks
        description = task.description.lower()

        # Single action words
        atomic_patterns = [
            "read",
            "write",
            "delete",
            "create",
            "run",
            "execute",
            "call",
            "send",
            "get",
            "set",
            "update",
            "check",
        ]

        # Check if task starts with atomic action
        for pattern in atomic_patterns:
            if description.startswith(pattern):
                return True

        # Check task length (short tasks are often atomic)
        if len(task.description.split()) <= 5:
            return True

        return False

    def _optimize_task_ordering(self) -> None:
        """Optimize task ordering for efficient execution."""
        # Get topological ordering
        execution_order = self.task_graph.get_execution_order()

        # Apply energy-based optimization
        self._apply_energy_optimization(execution_order)

    def _apply_energy_optimization(self, execution_order: list[list[str]]) -> None:
        """Apply energy-based optimization to task ordering.

        Args:
            execution_order: Current execution order
        """
        # For now, just prioritize shorter tasks first within each parallel group
        for group in execution_order:
            group.sort(
                key=lambda task_id: len(self.task_graph.get_task(task_id).description)
            )

    def _validate_plan(self) -> None:
        """Validate the generated plan for consistency and feasibility."""
        # Check for cycles
        if self.task_graph.has_cycles():
            raise ValueError("Plan contains circular dependencies")

        # Check for disconnected components
        if not self.task_graph.is_connected():
            print("Warning: Plan has disconnected components")

        # Validate task dependencies
        for task_id in self.task_graph.get_all_task_ids():
            task = self.task_graph.get_task(task_id)
            dependencies = self.task_graph.get_dependencies(task_id)

            # Check dependency validity
            for dep_id in dependencies:
                if not self.task_graph.get_task(dep_id):
                    raise ValueError(
                        f"Task {task_id} depends on non-existent task {dep_id}"
                    )

    async def execute_plan(
        self,
        task_graph: TaskGraph | None = None,
        strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL_SAFE,
    ) -> ExecutionResult:
        """Execute the task plan.

        Args:
            task_graph: Task graph to execute (uses current if None)
            strategy: Execution strategy

        Returns:
            ExecutionResult with overall execution outcome
        """
        if task_graph:
            self.task_graph = task_graph

        start_time = time.time()
        root_task_id = self._find_root_task()

        try:
            # Execute tasks according to strategy
            if strategy == ExecutionStrategy.SEQUENTIAL:
                result = await self._execute_sequential()
            elif strategy == ExecutionStrategy.PARALLEL_AGGRESSIVE:
                result = await self._execute_parallel_aggressive()
            else:  # PARALLEL_SAFE
                result = await self._execute_parallel_safe()

            execution_time = time.time() - start_time
            self.total_execution_time += execution_time

            return ExecutionResult(
                task_id=root_task_id,
                success=result,
                execution_time=execution_time,
                tools_used=list(self.execution_context.tools_used),
                metadata={
                    "total_tasks": len(self.task_graph.get_all_task_ids()),
                    "completed_tasks": len(self.completed_tasks),
                    "strategy": strategy.name,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                task_id=root_task_id,
                success=False,
                error=str(e),
                execution_time=execution_time,
            )

    def _find_root_task(self) -> str:
        """Find the root task (task with no dependencies)."""
        all_tasks = self.task_graph.get_all_task_ids()
        for task_id in all_tasks:
            dependencies = self.task_graph.get_dependencies(task_id)
            if not dependencies:
                return task_id
        raise ValueError("No root task found")

    async def _execute_sequential(self) -> bool:
        """Execute tasks sequentially."""
        execution_order = self.task_graph.get_execution_order()

        for group in execution_order:
            for task_id in group:
                success = await self._execute_single_task(task_id)
                if not success:
                    return False

        return True

    async def _execute_parallel_safe(self) -> bool:
        """Execute tasks in parallel with conservative concurrency."""
        execution_order = self.task_graph.get_execution_order()

        for group in execution_order:
            # Limit concurrent tasks
            semaphore = asyncio.Semaphore(
                min(self.config.max_concurrent_tasks, len(group))
            )

            async def execute_with_semaphore(task_id: str) -> bool:
                async with semaphore:
                    return await self._execute_single_task(task_id)

            # Execute group in parallel
            tasks = [execute_with_semaphore(task_id) for task_id in group]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check if all succeeded
            for result in results:
                if isinstance(result, Exception) or not result:
                    return False

        return True

    async def _execute_parallel_aggressive(self) -> bool:
        """Execute tasks with maximum parallelism."""
        execution_order = self.task_graph.get_execution_order()

        for group in execution_order:
            # Execute all tasks in group simultaneously
            tasks = [self._execute_single_task(task_id) for task_id in group]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check results
            for result in results:
                if isinstance(result, Exception) or not result:
                    return False

        return True

    async def _execute_single_task(self, task_id: str) -> bool:
        """Execute a single task.

        Args:
            task_id: ID of task to execute

        Returns:
            True if task executed successfully
        """
        task = self.task_graph.get_task(task_id)
        if not task:
            return False

        # Mark task as running
        task.status = TaskStatus.RUNNING
        self.active_tasks.add(task_id)

        try:
            # In shadow mode, simulate execution
            if self.config.shadow_mode:
                await asyncio.sleep(0.1)  # Simulate work
                result = f"Shadow execution of: {task.description}"
            else:
                # Real execution would go here
                result = await self._execute_task_real(task)

            # Update task state
            task.status = TaskStatus.COMPLETED
            task.result = result
            self.completed_tasks.add(task_id)
            self.total_tasks_completed += 1

            # Update bandit policy with reward
            reward = 1.0  # Success reward
            self.bandit_policy.update_reward("default_tool", reward)

            return True

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.result = f"Error: {str(e)}"

            # Update bandit policy with penalty
            reward = 0.0  # Failure penalty
            self.bandit_policy.update_reward("default_tool", reward)

            return False

        finally:
            self.active_tasks.discard(task_id)

    async def _execute_task_real(self, task: Task) -> Any:
        """Execute a task for real (placeholder for actual implementation).

        Args:
            task: Task to execute

        Returns:
            Task execution result
        """
        # This is where actual task execution would happen
        # For now, return a placeholder result
        await asyncio.sleep(0.1)  # Simulate work
        return f"Executed: {task.description}"

    def get_metrics(self) -> TaskGraphMetrics:
        """Get current execution metrics."""
        return self.task_graph.get_metrics()

    def get_status(self) -> dict[str, Any]:
        """Get current planner status."""
        return {
            "total_tasks": self.total_tasks_created,
            "completed_tasks": self.total_tasks_completed,
            "active_tasks": len(self.active_tasks),
            "execution_time": self.total_execution_time,
            "decomposition_depth": len(self.decomposition_history),
            "shadow_mode": self.config.shadow_mode,
        }

    def reset(self) -> None:
        """Reset the planner state."""
        self.task_graph = TaskGraph()
        self.execution_context = ExecutionContext()
        self.active_tasks.clear()
        self.completed_tasks.clear()
        self.total_tasks_created = 0
        self.total_tasks_completed = 0
        self.total_execution_time = 0.0
        self.decomposition_history.clear()
