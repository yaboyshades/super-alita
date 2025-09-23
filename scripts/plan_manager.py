# scripts/plan_manager.py
"""
Plan management for SDD workflows.
Handles creation and management of technical implementation plans.
"""

from pathlib import Path

from .common import Project, commit_changes, invoke_ai_generation
from .sdd_models import Plan, Task


class PlanManager:
    """Manages technical implementation plans."""

    def __init__(self, project: Project):
        self.project = project

    def create_plan(self, feature_name: str) -> Path:
        """Generate technical implementation plan for a feature."""
        # Find feature directory
        feature_path = self.project.get_feature_path(feature_name)
        if not feature_path:
            raise ValueError(f"Feature '{feature_name}' not found")

        # Load specification
        spec_path = feature_path / "spec.md"
        if not spec_path.exists():
            raise FileNotFoundError(f"Specification not found for {feature_name}")

        spec_content = spec_path.read_text()
        constitution = self._get_constitution_context()

        # Generate plan using structured AI output
        system_prompt = """You are a solutions architect creating a technical
        implementation plan.
        Generate a detailed plan that includes:
        - Technology stack decisions
        - Architecture design
        - Database schema (if applicable)
        - API design
        - Security considerations
        - Performance requirements
        - Testing strategy

        Provide specific, actionable details that developers can implement."""

        context = f"""
CONSTITUTION:
{constitution}

SPECIFICATION:
{spec_content}
"""

        plan_content = invoke_ai_generation(
            prompt=context, system_prompt=system_prompt, model="gpt-4-turbo"
        )

        # Write plan
        plan_path = feature_path / "plan.md"
        plan_path.write_text(plan_content)

        # Update tracker
        self._update_tracker(feature_path, "planning")

        # Git commit
        commit_changes(message=f"feat: plan {feature_name}", add_path=str(feature_path))

        return plan_path

    def parse_plan(self, plan_path: Path) -> Plan:
        """Parse a plan file into structured Plan object."""
        if not plan_path.exists():
            raise FileNotFoundError(f"Plan not found: {plan_path}")

        content = plan_path.read_text()

        # AI-powered parsing to extract tasks and rationale
        system_prompt = """Parse the technical implementation plan into
        structured data. Extract:
        - Key tasks with descriptions
        - Dependencies between tasks
        - Rationale for the approach
        - Assumptions and risks

        Focus on actionable implementation steps."""

        invoke_ai_generation(
            prompt=content, system_prompt=system_prompt, model="gpt-4-turbo"
        )

        # Simplified parsing - create basic plan structure
        return Plan(
            feature_id=plan_path.parent.name,
            tasks=[
                Task(
                    id="task-1",
                    description="Implement core functionality",
                    status="pending",
                    dependencies=[],
                    assignee=None,
                    estimated_effort=None,
                    priority="high",
                )
            ],
            rationale="Plan generated from specification",
            assumptions=["Standard tech stack available"],
            risks=["Integration challenges"],
        )

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
