"""Utilities for coordinating a refactoring task force."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum


class RefactoringFocus(Enum):
    """Focus areas the refactoring task force can specialize in."""

    RELIABILITY = "reliability"
    PERFORMANCE = "performance"
    API_CONTRACT = "api_contract"
    TEST_COVERAGE = "test_coverage"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"


class TaskStage(Enum):
    """Lifecycle stages for objective-aligned tasks."""

    ANALYSIS = "analysis"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    INTEGRATION = "integration"


class RefactoringTaskState(Enum):
    """Execution state for a refactoring task."""

    READY = "ready"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETE = "complete"


@dataclass
class RefactoringObjective:
    """Objective that the refactoring task force is responsible for."""

    objective_id: str
    description: str
    focus_area: RefactoringFocus
    priority: int = 3
    success_metrics: dict[str, float] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)


@dataclass
class TaskForceMember:
    """Represents a member in the refactoring task force."""

    name: str
    specialty: RefactoringFocus
    capacity: float = 1.0
    allocation: float = 0.0

    def available_capacity(self) -> float:
        return max(self.capacity - self.allocation, 0.0)

    def can_accept(self, load: float) -> bool:
        return load <= self.available_capacity() + 1e-6

    def assign(self, load: float) -> None:
        if not self.can_accept(load):
            raise ValueError(
                f"Member {self.name} cannot accept load {load} (capacity exceeded)."
            )
        self.allocation += load

    def release(self, load: float) -> None:
        self.allocation = max(self.allocation - load, 0.0)


@dataclass
class RefactoringTask:
    """Concrete task owned by the refactoring task force."""

    task_id: str
    objective_id: str
    stage: TaskStage
    focus_area: RefactoringFocus
    estimated_effort: float
    dependencies: list[str] = field(default_factory=list)
    assigned_to: str | None = None
    state: RefactoringTaskState = RefactoringTaskState.READY
    progress: float = 0.0
    blockers: list[str] = field(default_factory=list)

    def copy(self) -> RefactoringTask:
        """Return a shallow copy for safe external consumption."""

        return RefactoringTask(
            task_id=self.task_id,
            objective_id=self.objective_id,
            stage=self.stage,
            focus_area=self.focus_area,
            estimated_effort=self.estimated_effort,
            dependencies=list(self.dependencies),
            assigned_to=self.assigned_to,
            state=self.state,
            progress=self.progress,
            blockers=list(self.blockers),
        )


class RefactoringTaskForce:
    """Coordinates multiple refactoring objectives in parallel squads."""

    def __init__(
        self,
        objectives: Sequence[RefactoringObjective],
        members: Sequence[TaskForceMember],
        *,
        integration_focus: RefactoringFocus = RefactoringFocus.RELIABILITY,
        validation_focus: RefactoringFocus = RefactoringFocus.TEST_COVERAGE,
    ) -> None:
        if not objectives:
            raise ValueError("At least one objective is required to form a task force.")
        if not members:
            raise ValueError("At least one member is required to form a task force.")

        self._objectives: dict[str, RefactoringObjective] = {
            obj.objective_id: obj for obj in objectives
        }
        self._objective_order: list[RefactoringObjective] = sorted(
            objectives, key=lambda obj: (obj.priority, obj.objective_id)
        )
        self._members: dict[str, TaskForceMember] = {
            member.name: member for member in members
        }
        self._tasks: dict[str, RefactoringTask] = {}
        self._integration_focus = integration_focus
        self._validation_focus = validation_focus

        self._initialize_tasks()
        self._assign_initial_work()

    @property
    def members(self) -> dict[str, TaskForceMember]:
        return self._members

    def _initialize_tasks(self) -> None:
        for objective in self._objective_order:
            created_tasks = self._decompose_objective(objective)
            for task in created_tasks:
                self._tasks[task.task_id] = task

    def _decompose_objective(
        self, objective: RefactoringObjective
    ) -> list[RefactoringTask]:
        tasks: list[RefactoringTask] = []
        stage_sequence = [
            (TaskStage.ANALYSIS, objective.focus_area, 0.25),
            (TaskStage.IMPLEMENTATION, objective.focus_area, 0.4),
            (TaskStage.VALIDATION, self._validation_focus, 0.2),
        ]

        dependencies: list[str] = []
        for stage, focus, effort in stage_sequence:
            task_id = self._make_task_id(objective.objective_id, stage)
            tasks.append(
                RefactoringTask(
                    task_id=task_id,
                    objective_id=objective.objective_id,
                    stage=stage,
                    focus_area=focus,
                    estimated_effort=effort,
                )
            )
            dependencies.append(task_id)

        integration_task = RefactoringTask(
            task_id=self._make_task_id(objective.objective_id, TaskStage.INTEGRATION),
            objective_id=objective.objective_id,
            stage=TaskStage.INTEGRATION,
            focus_area=self._integration_focus,
            estimated_effort=0.15,
            dependencies=list(dependencies),
            state=RefactoringTaskState.BLOCKED,
        )
        tasks.append(integration_task)
        return tasks

    @staticmethod
    def _make_task_id(objective_id: str, stage: TaskStage) -> str:
        return f"{objective_id}:{stage.value}"

    def _assign_initial_work(self) -> None:
        for task in self._tasks.values():
            member = self._select_member(task.focus_area, task.estimated_effort)
            if member is not None:
                member.assign(task.estimated_effort)
                task.assigned_to = member.name

    def _select_member(
        self, focus: RefactoringFocus, estimated_effort: float
    ) -> TaskForceMember | None:
        candidates = [
            member
            for member in self._members.values()
            if member.specialty == focus and member.can_accept(estimated_effort)
        ]
        if not candidates:
            candidates = [
                member
                for member in self._members.values()
                if member.can_accept(estimated_effort)
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda member: member.available_capacity())

    def get_tasks_for_objective(self, objective_id: str) -> list[RefactoringTask]:
        return [
            task.copy()
            for task in self._tasks.values()
            if task.objective_id == objective_id
        ]

    def update_task_state(
        self,
        task_id: str,
        *,
        progress: float | None = None,
        state: RefactoringTaskState | None = None,
        blockers: Iterable[str] | None = None,
    ) -> RefactoringTask:
        if task_id not in self._tasks:
            raise KeyError(f"Unknown task_id: {task_id}")
        task = self._tasks[task_id]

        if progress is not None:
            if not 0.0 <= progress <= 1.0:
                raise ValueError("Progress must be within [0.0, 1.0].")
            if progress < task.progress:
                # Prevent regressions in progress reporting
                progress = task.progress
            task.progress = progress
            if progress >= 1.0:
                task.state = RefactoringTaskState.COMPLETE
            elif task.state == RefactoringTaskState.READY:
                task.state = RefactoringTaskState.IN_PROGRESS

        if state is not None:
            task.state = state
            if state == RefactoringTaskState.COMPLETE:
                task.progress = 1.0

        if blockers is not None:
            task.blockers = list(blockers)
            if task.blockers and task.state != RefactoringTaskState.COMPLETE:
                task.state = RefactoringTaskState.BLOCKED
            elif not task.blockers and task.state == RefactoringTaskState.BLOCKED:
                task.state = RefactoringTaskState.READY

        self._refresh_integration_state(task.objective_id)
        return task.copy()

    def _refresh_integration_state(self, objective_id: str) -> None:
        integration_id = self._make_task_id(objective_id, TaskStage.INTEGRATION)
        integration_task = self._tasks[integration_id]
        prerequisite_ids = integration_task.dependencies
        prerequisites_complete = all(
            self._tasks[dep_id].state == RefactoringTaskState.COMPLETE
            for dep_id in prerequisite_ids
        )
        if (
            prerequisites_complete
            and integration_task.state == RefactoringTaskState.BLOCKED
        ):
            integration_task.state = RefactoringTaskState.READY
        elif (
            not prerequisites_complete
            and integration_task.state == RefactoringTaskState.READY
        ):
            integration_task.state = RefactoringTaskState.BLOCKED

    def generate_convergence_plan(self) -> list[dict[str, object]]:
        plan: list[dict[str, object]] = []
        for objective in self._objective_order:
            tasks = [
                self._tasks[task_id]
                for task_id in sorted(self._tasks)
                if self._tasks[task_id].objective_id == objective.objective_id
            ]
            integration_task = next(
                task for task in tasks if task.stage == TaskStage.INTEGRATION
            )
            dependencies_complete = all(
                self._tasks[dep_id].state == RefactoringTaskState.COMPLETE
                for dep_id in integration_task.dependencies
            )
            integration_ready = integration_task.state in (
                RefactoringTaskState.READY,
                RefactoringTaskState.IN_PROGRESS,
                RefactoringTaskState.COMPLETE,
            )
            ready_for_integration = dependencies_complete and integration_ready
            plan.append(
                {
                    "objective_id": objective.objective_id,
                    "description": objective.description,
                    "ready_for_integration": ready_for_integration,
                    "converged": integration_task.state
                    == RefactoringTaskState.COMPLETE,
                    "integration_task": integration_task.task_id,
                    "pending_tasks": [
                        task.task_id
                        for task in tasks
                        if task.state != RefactoringTaskState.COMPLETE
                    ],
                    "loop_alignment": self._build_loop_alignment(
                        objective, tasks, integration_task
                    ),
                }
            )
        return plan

    def _build_loop_alignment(
        self,
        objective: RefactoringObjective,
        tasks: list[RefactoringTask],
        integration_task: RefactoringTask,
    ) -> dict[str, object]:
        energy = sum(task.estimated_effort * task.progress for task in tasks)
        todo_count = sum(
            1
            for task in tasks
            if task.state
            in {RefactoringTaskState.READY, RefactoringTaskState.BLOCKED}
        )
        bandit_ready = sum(
            1 for task in tasks if task.state == RefactoringTaskState.READY
        )
        reward = (
            objective.success_metrics
            if integration_task.state == RefactoringTaskState.COMPLETE
            else {}
        )
        return {
            "event": f"refactor::{objective.objective_id}",
            "atoms": [task.task_id for task in tasks],
            "bonds": list(integration_task.dependencies),
            "energy": round(energy, 4),
            "todo": todo_count,
            "bandit": bandit_ready,
            "reward": reward,
        }


__all__ = [
    "RefactoringFocus",
    "TaskStage",
    "RefactoringTaskState",
    "RefactoringObjective",
    "TaskForceMember",
    "RefactoringTask",
    "RefactoringTaskForce",
]
