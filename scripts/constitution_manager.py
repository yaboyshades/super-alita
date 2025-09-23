# scripts/constitution_manager.py
"""
Constitution management for SDD workflows.
Handles creation and validation of project constitutions.
"""

from datetime import datetime
from pathlib import Path

from jinja2 import Template

from .common import (
    Project,
    commit_changes,
    invoke_ai_generation,
    load_template,
)
from .sdd_models import ConstitutionCheck, ConstitutionResult


class ConstitutionManager:
    """Manages project constitution creation and validation."""

    def __init__(self, project: Project):
        self.project = project

    def create_constitution(self, principles: str, force: bool = False) -> Path:
        """Create a new project constitution."""
        constitution_path = self.project.memory_dir / "constitution.md"

        if constitution_path.exists() and not force:
            raise FileExistsError(
                f"Constitution already exists at {constitution_path}. "
                "Use force=True to overwrite."
            )

        self.project.memory_dir.mkdir(parents=True, exist_ok=True)

        # Load template
        template_content = load_template(self.project, "constitution-template.md")
        template = Template(template_content)

        # Generate constitution content
        system_prompt = """You are a principal engineer establishing a
        project constitution.
        Create a comprehensive constitution based on the provided principles.
        Include governance rules, development practices, and quality standards.
        Be specific, actionable, and focused on long-term maintainability."""

        constitution_content = invoke_ai_generation(
            prompt=principles, system_prompt=system_prompt, model="gpt-4-turbo"
        )

        # Render template
        rendered = template.render(
            ai_generated_content=constitution_content,
            project_name=self.project.root.name,
            date=datetime.now().strftime("%Y-%m-%d"),
        )

        # Write constitution
        constitution_path.write_text(rendered)

        # Git commit
        commit_changes(
            message="feat: establish project constitution",
            add_path=str(self.project.memory_dir),
        )

        return constitution_path

    def validate_constitution(self, feature_context: str) -> ConstitutionResult:
        """Validate a feature against the project constitution."""
        constitution_path = self.project.memory_dir / "constitution.md"

        if not constitution_path.exists():
            return ConstitutionResult(
                feature_id="unknown",
                checks=[
                    ConstitutionCheck(
                        rule_name="constitution_exists",
                        passed=False,
                        severity="critical",
                        message="Project constitution not found",
                        details={"constitution_path": str(constitution_path)},
                    )
                ],
                overall_score=0.0,
                passed=False,
                recommendations=["Create project constitution first"],
            )

        constitution_content = constitution_path.read_text()

        # AI-powered validation
        system_prompt = """You are a senior engineer validating compliance
        with project constitution.
        Analyze the feature context against the constitution and identify:
        - Compliance issues
        - Risk areas
        - Required changes
        - Quality concerns

        Provide specific, actionable feedback."""

        context = f"""
CONSTITUTION:
{constitution_content}

FEATURE CONTEXT:
{feature_context}
"""

        validation_response = invoke_ai_generation(
            prompt=context, system_prompt=system_prompt, model="gpt-4-turbo"
        )

        # Parse AI response into structured checks
        # This is a simplified implementation - in practice you'd want
        # more structured parsing of the AI response
        checks = [
            ConstitutionCheck(
                rule_name="ai_validation",
                passed="COMPLIANT" in validation_response.upper(),
                severity="medium",
                message="AI-powered constitution validation",
                details={"ai_response": validation_response},
            )
        ]

        overall_score = 0.8 if checks[0].passed else 0.3
        passed = overall_score >= 0.75

        return ConstitutionResult(
            feature_id="validation",
            checks=checks,
            overall_score=overall_score,
            passed=passed,
            recommendations=([] if passed else ["Review AI validation feedback"]),
        )

    def get_constitution_context(self) -> str:
        """Get constitution content for context in other operations."""
        constitution_path = self.project.memory_dir / "constitution.md"
        if constitution_path.exists():
            return constitution_path.read_text()
        return "No constitution available"
