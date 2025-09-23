# scripts/spec_manager.py
"""
Specification management for SDD workflows.
Handles creation and management of feature specifications.
"""

from datetime import datetime
from pathlib import Path

from .common import Project, commit_changes, invoke_ai_generation
from .sdd_models import FeatureSpec, UserScenario


class SpecManager:
    """Manages feature specification creation and management."""

    def __init__(self, project: Project):
        self.project = project

    def create_specification(
        self, feature_name: str, requirements: str, context: str | None = None
    ) -> Path:
        """Create a feature specification from requirements."""
        # Create feature directory
        feature_path = self.project.create_feature_directory(feature_name)

        # Get constitution context
        constitution = self._get_constitution_context()

        # Generate specification
        system_prompt = """You are a senior product manager creating a feature
        specification.
        Generate a comprehensive, testable specification that includes:
        - Clear requirements
        - Acceptance criteria
        - Constraints and assumptions
        - Success metrics

        Follow the provided template structure and ensure all sections are
        filled."""

        full_context = f"""
CONSTITUTION:
{constitution}

FEATURE: {feature_name}
REQUIREMENTS: {requirements}
"""

        if context:
            full_context += f"\nADDITIONAL CONTEXT: {context}"

        spec_content = invoke_ai_generation(
            prompt=full_context, system_prompt=system_prompt, model="gpt-4-turbo"
        )

        # Write specification
        spec_path = feature_path / "spec.md"
        spec_path.write_text(spec_content)

        # Create initial task tracker
        tracker_data = {
            "feature_name": feature_name,
            "feature_path": str(feature_path),
            "current_phase": "spec",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "tasks": [],
        }

        tracker_path = feature_path / "tracker.json"
        import json

        tracker_path.write_text(json.dumps(tracker_data, indent=2))

        # Git commit
        commit_changes(
            message=f"feat: specify {feature_name}", add_path=str(feature_path)
        )

        return spec_path

    def parse_specification(self, spec_path: Path) -> FeatureSpec:
        """Parse a specification file into structured data."""
        if not spec_path.exists():
            raise FileNotFoundError(f"Specification not found: {spec_path}")

        content = spec_path.read_text()

        # AI-powered parsing (simplified - in practice you'd want
        # more robust parsing)
        system_prompt = """Parse the feature specification into structured
        data.
        Extract:
        - Feature name and description
        - User scenarios with acceptance criteria
        - Any additional context

        Return the information in a clear, structured format."""

        parsed_response = invoke_ai_generation(
            prompt=content, system_prompt=system_prompt, model="gpt-4-turbo"
        )

        # Simplified parsing - extract basic info from first few lines
        lines = content.split("\n")
        feature_name = ""
        description = ""

        for line in lines[:10]:  # Check first few lines
            line = line.strip()
            if line.startswith("#"):
                feature_name = line.lstrip("#").strip()
                break

        # Create basic feature spec
        return FeatureSpec(
            name=feature_name or "Unknown Feature",
            description=description or parsed_response[:200],
            scenarios=[
                UserScenario(
                    description="Primary user scenario",
                    acceptance_criteria=["To be defined from spec"],
                )
            ],
        )

    def _get_constitution_context(self) -> str:
        """Get constitution content for context."""
        constitution_path = self.project.memory_dir / "constitution.md"
        if constitution_path.exists():
            return constitution_path.read_text()
        return "No constitution available"
