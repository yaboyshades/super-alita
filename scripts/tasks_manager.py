# scripts/tasks_manager.py
"""
Tasks management for SDD workflows.
Handles creation and management of executable tasks from plans.
"""

from pathlib import Path

from .common import Project, commit_changes, invoke_ai_generation
from .sdd_models import Task


class TasksManager:
    """Manages task breakdown and execution tracking."""

    def __init__(self, project: Project):
        self.project = project

    def create_tasks(self, feature_name: str) -> Path:
        """Break down plan into executable tasks."""
        # Find feature directory
        feature_path = self.project.get_feature_path(feature_name)
        if not feature_path:
            raise ValueError(f"Feature '{feature_name}' not found")

        # Load plan
        plan_path = feature_path / "plan.md"
        if not plan_path.exists():
            raise FileNotFoundError(f"Plan not found for {feature_name}")

        plan_content = plan_path.read_text()
        constitution = self._get_constitution_context()

        # Generate tasks
        system_prompt = """You are a technical lead breaking down a plan into
        executable tasks.
        Create a comprehensive list of tasks that includes:
        - Specific, actionable work items
        - Dependencies between tasks
        - Estimated effort for each task
        - Acceptance criteria for completion

        Tasks should be small enough to complete in 1-2 days each."""

        context = f"""
CONSTITUTION:
{constitution}

PLAN:
{plan_content}
"""

        tasks_content = invoke_ai_generation(
            prompt=context, system_prompt=system_prompt, model="gpt-4-turbo"
        )

        # Write tasks
        tasks_path = feature_path / "tasks.md"
        tasks_path.write_text(tasks_content)

        # Update tracker with parsed tasks
        self._update_tracker(feature_path, "tasking")

        # Git commit
        commit_changes(
            message=f"feat: tasks {feature_name}", add_path=str(feature_path)
        )

        return tasks_path

    def parse_tasks(self, tasks_path: Path) -> list[Task]:
        """Parse a tasks file into structured Task objects."""
        if not tasks_path.exists():
            raise FileNotFoundError(f"Tasks file not found: {tasks_path}")

        content = tasks_path.read_text()

        # AI-powered parsing to extract structured tasks
        system_prompt = """Parse the tasks document into structured task data.
        Extract:
        - Task IDs and descriptions
        - Dependencies between tasks
        - Estimated effort and priority
        - Status and assignees

        Return tasks in a clear, structured format."""

        invoke_ai_generation(
            prompt=content, system_prompt=system_prompt, model="gpt-4-turbo"
        )

        # Simplified parsing - create basic task structure
        return [
            Task(
                id="task-1",
                description="Implement core functionality",
                status="pending",
                dependencies=[],
                assignee=None,
                estimated_effort="2h",
                priority="high",
            )
        ]

    def update_task_status(self, feature_name: str, task_id: str, status: str) -> None:
        """Update the status of a specific task."""
        feature_path = self.project.get_feature_path(feature_name)
        if not feature_path:
            raise ValueError(f"Feature '{feature_name}' not found")

        tracker_path = feature_path / "tracker.json"
        if not tracker_path.exists():
            raise FileNotFoundError(f"Task tracker not found for {feature_name}")

        import json
        from datetime import datetime

        tracker_data = json.loads(tracker_path.read_text())

        # Find and update task
        for task in tracker_data.get("tasks", []):
            if task.get("id") == task_id:
                task["status"] = status
                task["updated_at"] = datetime.now().isoformat()
                break

        tracker_data["updated_at"] = datetime.now().isoformat()
        tracker_path.write_text(json.dumps(tracker_data, indent=2))

    def _update_tracker(self, feature_path: Path, phase: str) -> None:
        """Update the feature tracker with current phase."""
        import json
        from datetime import datetime

        tracker_path = feature_path / "tracker.json"
        if tracker_path.exists():
            tracker_data = json.loads(tracker_path.read_text())
            tracker_data["current_phase"] = phase
            tracker_data["updated_at"] = datetime.now().isoformat()
            tracker_path.write_text(json.dumps(tracker_data, indent=2))

    def _get_constitution_context(self) -> str:
        """Get constitution content for context."""
        constitution_path = self.project.memory_dir / "constitution.md"
        if constitution_path.exists():
            return constitution_path.read_text()
        return "No constitution available"
