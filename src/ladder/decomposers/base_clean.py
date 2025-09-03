"""Task decomposition framework for LADDER hierarchical planning."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..models.task import Task, TaskType


class DecomposerType(Enum):
    """Types of decomposition strategies."""

    DEFAULT = "default"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


@dataclass
class DecompositionResult:
    """Result of task decomposition."""

    subtasks: list[Task] = field(default_factory=list)
    execution_strategy: str = "sequential"
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: str | None = None

    def add_dependency(self, task_id: str, depends_on: list[str]) -> None:
        """Add dependency relationship between tasks."""
        self.dependencies[task_id] = depends_on

        # Also update the task's dependencies field
        for task in self.subtasks:
            if task.id == task_id:
                task.dependencies.extend(depends_on)
                # Remove duplicates
                task.dependencies = list(set(task.dependencies))


class LadderDecomposer(ABC):
    """Abstract base class for task decomposition strategies."""

    def __init__(self, name: str, decomposer_type: DecomposerType):
        """Initialize decomposer with name and type."""
        self.name = name
        self.decomposer_type = decomposer_type
        self.decomposition_count = 0

    @abstractmethod
    async def decompose(
        self, task: Task, context: dict[str, Any] | None = None
    ) -> DecompositionResult:
        """
        Decompose a task into subtasks.

        Args:
            task: The task to decompose
            context: Optional context information

        Returns:
            DecompositionResult with subtasks and execution strategy
        """
        pass

    def can_decompose(self, task: Task) -> bool:
        """Check if this decomposer can handle the given task."""
        # Default implementation - can decompose any non-atomic task
        return not task.is_atomic


class DefaultLLMDecomposer(LadderDecomposer):
    """Default decomposer using LLM for task breakdown."""

    def __init__(self, llm_provider=None):
        """Initialize with optional LLM provider."""
        super().__init__("default_llm", DecomposerType.DEFAULT)
        self.llm_provider = llm_provider

    async def decompose(
        self, task: Task, context: dict[str, Any] | None = None
    ) -> DecompositionResult:
        """Decompose task using LLM-based analysis."""
        self.decomposition_count += 1

        try:
            # Determine task type from metadata or context
            try:
                task_type = TaskType(task.metadata.get("task_type", "general"))
            except ValueError:
                task_type = TaskType.GENERAL

            # Create heuristic decomposition
            subtasks_data = self._heuristic_decomposition(task, task_type)

            # Create Task objects from the data
            subtasks = []
            for i, task_data in enumerate(subtasks_data):
                subtask = Task(
                    description=task_data.get("description", f"Subtask {i+1}"),
                    energy=task_data.get("energy", 2.0),
                    tool_options=task_data.get("tool_options", []),
                    metadata={
                        "parent_id": task.id,
                        "task_type": task_type.value,
                        "decomposer": self.name,
                    },
                )
                subtasks.append(subtask)

            # Build dependency graph
            result = DecompositionResult(
                subtasks=subtasks,
                execution_strategy="sequential",
                metadata={
                    "decomposer_type": self.decomposer_type.value,
                    "original_task_id": task.id,
                    "task_type": task_type.value,
                },
            )

            # Add dependencies from decomposition
            for i, task_data in enumerate(subtasks_data):
                if "dependencies" in task_data and task_data["dependencies"]:
                    # Map dependency indices to actual task IDs
                    deps = []
                    for dep_idx in task_data["dependencies"]:
                        if isinstance(dep_idx, int) and 0 <= dep_idx < len(subtasks):
                            deps.append(subtasks[dep_idx].id)
                    result.add_dependency(subtasks[i].id, deps)

            return result

        except Exception as e:
            return DecompositionResult(
                success=False, error_message=f"Decomposition failed: {str(e)}"
            )

    def _heuristic_decomposition(
        self, task: Task, task_type: TaskType
    ) -> list[dict[str, Any]]:
        """Fallback heuristic decomposition when LLM is unavailable."""

        if task_type == TaskType.CODING:
            return [
                {
                    "description": (f"Plan implementation for: " f"{task.description}"),
                    "tool_options": ["llm", "knowledge_graph"],
                    "energy": 2.0,
                    "dependencies": [],
                },
                {
                    "description": f"Implement: {task.description}",
                    "tool_options": ["code_executor", "file_writer"],
                    "energy": 4.0,
                    "dependencies": [0],
                },
                {
                    "description": f"Test and validate: {task.description}",
                    "tool_options": ["test_runner", "code_executor"],
                    "energy": 2.0,
                    "dependencies": [1],
                },
            ]

        elif task_type == TaskType.RESEARCH:
            return [
                {
                    "description": f"Gather information: {task.description}",
                    "tool_options": ["web_search", "knowledge_graph"],
                    "energy": 3.0,
                    "dependencies": [],
                },
                {
                    "description": f"Analyze findings: {task.description}",
                    "tool_options": ["llm", "knowledge_graph"],
                    "energy": 3.0,
                    "dependencies": [0],
                },
                {
                    "description": f"Document results: {task.description}",
                    "tool_options": ["file_writer", "llm"],
                    "energy": 2.0,
                    "dependencies": [1],
                },
            ]

        else:
            # Generic decomposition
            return [
                {
                    "description": f"Analyze: {task.description}",
                    "tool_options": ["llm"],
                    "energy": 1.5,
                    "dependencies": [],
                },
                {
                    "description": f"Execute: {task.description}",
                    "tool_options": ["llm", "file_writer"],
                    "energy": 3.0,
                    "dependencies": [0],
                },
                {
                    "description": f"Finalize: {task.description}",
                    "tool_options": ["llm"],
                    "energy": 1.5,
                    "dependencies": [1],
                },
            ]


class SequentialDecomposer(LadderDecomposer):
    """Decomposer that creates sequential subtasks."""

    def __init__(self):
        """Initialize sequential decomposer."""
        super().__init__("sequential", DecomposerType.SEQUENTIAL)

    async def decompose(
        self, task: Task, context: dict[str, Any] | None = None
    ) -> DecompositionResult:
        """Create sequential decomposition."""
        self.decomposition_count += 1

        # Simple sequential breakdown
        subtasks = [
            Task(
                description=f"Prepare: {task.description}",
                energy=1.0,
                tool_options=["llm"],
                metadata={"parent_id": task.id, "step": 1},
            ),
            Task(
                description=f"Execute: {task.description}",
                energy=task.energy * 0.7,
                tool_options=task.tool_options or ["llm"],
                metadata={"parent_id": task.id, "step": 2},
            ),
            Task(
                description=f"Complete: {task.description}",
                energy=task.energy * 0.3,
                tool_options=["llm"],
                metadata={"parent_id": task.id, "step": 3},
            ),
        ]

        result = DecompositionResult(
            subtasks=subtasks,
            execution_strategy="sequential",
            metadata={"decomposer_type": self.decomposer_type.value},
        )

        # Add sequential dependencies
        for i in range(1, len(subtasks)):
            result.add_dependency(subtasks[i].id, [subtasks[i - 1].id])

        return result


class ParallelDecomposer(LadderDecomposer):
    """Decomposer that creates parallel subtasks."""

    def __init__(self):
        """Initialize parallel decomposer."""
        super().__init__("parallel", DecomposerType.PARALLEL)

    async def decompose(
        self, task: Task, context: dict[str, Any] | None = None
    ) -> DecompositionResult:
        """Create parallel decomposition."""
        self.decomposition_count += 1

        # Look for parallelizable elements
        desc_lower = task.description.lower()

        if "and" in desc_lower or "," in task.description:
            # Try to split on conjunctions
            parts = re.split(r"\s+and\s+|,\s*", task.description)
            parts = [p.strip() for p in parts if p.strip()]

            if len(parts) > 1:
                subtasks = []
                for i, part in enumerate(parts):
                    subtask = Task(
                        description=part,
                        energy=task.energy / len(parts),
                        tool_options=task.tool_options,
                        metadata={"parent_id": task.id, "parallel_part": i + 1},
                    )
                    subtasks.append(subtask)

                return DecompositionResult(
                    subtasks=subtasks,
                    execution_strategy="parallel",
                    metadata={"decomposer_type": self.decomposer_type.value},
                )

        # Fallback to two parallel components
        subtasks = [
            Task(
                description=f"Component A: {task.description}",
                energy=task.energy * 0.5,
                tool_options=task.tool_options,
                metadata={"parent_id": task.id, "component": "A"},
            ),
            Task(
                description=f"Component B: {task.description}",
                energy=task.energy * 0.5,
                tool_options=task.tool_options,
                metadata={"parent_id": task.id, "component": "B"},
            ),
        ]

        return DecompositionResult(
            subtasks=subtasks,
            execution_strategy="parallel",
            metadata={"decomposer_type": self.decomposer_type.value},
        )


def create_decomposer(decomposer_type: DecomposerType, **kwargs) -> LadderDecomposer:
    """Factory function to create decomposers."""
    if decomposer_type == DecomposerType.DEFAULT:
        return DefaultLLMDecomposer(**kwargs)
    elif decomposer_type == DecomposerType.SEQUENTIAL:
        return SequentialDecomposer()
    elif decomposer_type == DecomposerType.PARALLEL:
        return ParallelDecomposer()
    else:
        raise ValueError(f"Unknown decomposer type: {decomposer_type}")


def select_decomposer(
    task: Task,
    available_decomposers: list[LadderDecomposer],
    context: dict[str, Any] | None = None,
) -> LadderDecomposer | None:
    """Select the best decomposer for a given task."""
    if not available_decomposers:
        return None

    # Filter decomposers that can handle this task
    capable_decomposers = [d for d in available_decomposers if d.can_decompose(task)]

    if not capable_decomposers:
        return None

    # Simple selection heuristics
    desc_lower = task.description.lower()

    # Prefer parallel decomposer for tasks with conjunctions
    if ("and" in desc_lower or "," in task.description) and len(
        task.description.split()
    ) > 10:
        parallel_decomposers = [
            d
            for d in capable_decomposers
            if d.decomposer_type == DecomposerType.PARALLEL
        ]
        if parallel_decomposers:
            return parallel_decomposers[0]

    # Prefer sequential for step-by-step tasks
    step_keywords = ["first", "then", "next", "finally", "step"]
    if any(word in desc_lower for word in step_keywords):
        sequential_decomposers = [
            d
            for d in capable_decomposers
            if d.decomposer_type == DecomposerType.SEQUENTIAL
        ]
        if sequential_decomposers:
            return sequential_decomposers[0]

    # Default to first capable decomposer
    return capable_decomposers[0]
