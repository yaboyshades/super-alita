# scripts/spec_kit_cli.py
"""
Python-native Spec Kit CLI for SDD workflows using Typer, Pydantic, and Jinja2.
Provides a robust, type-safe alternative to shell scripts for SDD workflows.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import openai
import typer
from git import Repo
from jinja2 import Environment, FileSystemLoader, Template
from pydantic import BaseModel, Field

# Configuration
DEFAULT_MODEL = "gpt-4-turbo"
MAX_TOKENS = 4000


class ProjectConfig(BaseModel):
    """Project configuration and paths."""

    root: Path
    memory_dir: Path
    specs_dir: Path
    templates_dir: Path
    scripts_dir: Path

    @classmethod
    def from_path(cls, path: str = ".") -> "ProjectConfig":
        root = Path(path).resolve()
        return cls(
            root=root,
            memory_dir=root / "memory",
            specs_dir=root / "specs",
            templates_dir=root / "templates",
            scripts_dir=root / "scripts",
        )


class AIClient:
    """AI client for generating content."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model

    def generate(self, system_prompt: str, user_content: str) -> str:
        """Generate content using AI."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.7,
            )
            content = response.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:
            typer.echo(f"❌ AI generation failed: {e}", err=True)
            raise typer.Exit(1) from e


class ConstitutionModel(BaseModel):
    """Constitution data model."""

    project_name: str
    version: str = "1.0.0"
    principles: list[dict[str, str]]
    governance: dict[str, Any]
    ratification_date: str
    last_amended: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d")
    )


class SpecificationModel(BaseModel):
    """Feature specification data model."""

    feature_name: str
    description: str
    requirements: list[str]
    acceptance_criteria: list[str]
    constraints: list[str]
    priority: str = "medium"


class PlanModel(BaseModel):
    """Technical plan data model."""

    technology_stack: list[str]
    architecture: dict[str, Any]
    database_schema: str | None = None
    api_endpoints: list[dict[str, Any]] = []
    security_plan: str
    performance_plan: str
    testing_strategy: str


class TaskModel(BaseModel):
    """Task data model."""

    id: str
    title: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, blocked
    assignee: str | None = None
    dependencies: list[str] = []
    estimated_hours: float | None = None
    actual_hours: float | None = None


class FeatureTracker(BaseModel):
    """Feature implementation tracker."""

    feature_name: str
    feature_path: Path
    tasks: list[TaskModel]
    current_phase: str = "planning"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# Global instances
project = ProjectConfig.from_path()
ai_client = AIClient()
jinja_env = Environment(loader=FileSystemLoader(project.templates_dir))

app = typer.Typer(help="Python-native Spec Kit CLI for SDD workflows")


def load_template(template_name: str) -> Template:
    """Load a Jinja2 template."""
    try:
        template = jinja_env.get_template(template_name)
        return template
    except Exception as e:
        typer.echo(f"❌ Template {template_name} not found: {e}", err=True)
        raise typer.Exit(1) from e


def get_constitution_context() -> str:
    """Get constitution content for context."""
    constitution_path = project.memory_dir / "constitution.md"
    if constitution_path.exists():
        return constitution_path.read_text()
    return ""


def create_feature_directory(feature_name: str) -> Path:
    """Create a numbered feature directory."""
    # Find next available number
    existing_features = list(project.specs_dir.glob("???-*"))
    next_num = len(existing_features) + 1

    # Create slug from feature name
    slug = feature_name.lower().replace(" ", "-").replace("_", "-")
    slug = "".join(c for c in slug if c.isalnum() or c == "-")

    feature_path = project.specs_dir / f"{next_num:03d}-{slug}"
    feature_path.mkdir(parents=True, exist_ok=True)
    return feature_path


def git_commit_feature(feature_path: Path, message: str) -> None:
    """Commit feature changes to git."""
    try:
        repo = Repo(project.root)
        # Add feature files
        repo.index.add([str(f) for f in feature_path.rglob("*") if f.is_file()])
        repo.index.commit(message)
        typer.echo(f"✅ Committed: {message}")
    except Exception as e:
        typer.echo(f"⚠️ Git commit failed: {e}")


@app.command()
def constitution(
    principles: str = typer.Option(
        ..., "--principles", "-p", help="High-level project principles"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing constitution"
    ),
):
    """Establish project constitution using AI."""
    typer.echo("🏛️ Establishing project constitution...")

    # Check if constitution exists
    constitution_path = project.memory_dir / "constitution.md"
    if constitution_path.exists() and not force:
        typer.echo("❌ Constitution already exists. Use --force to overwrite.")
        raise typer.Exit(1)

    project.memory_dir.mkdir(exist_ok=True)

    # Load template
    template = load_template("constitution-template.md")

    # Generate constitution content
    system_prompt = """You are a principal engineer establishing a
    project constitution.
    Create a comprehensive constitution based on the provided principles.
    Include governance rules, development practices, and quality standards.
    Be specific, actionable, and focused on long-term maintainability."""

    constitution_content = ai_client.generate(system_prompt, principles)

    # Render template
    rendered = template.render(
        ai_generated_content=constitution_content,
        project_name=project.root.name,
        date=datetime.now().strftime("%Y-%m-%d"),
    )

    # Write constitution
    constitution_path.write_text(rendered)

    # Git commit
    git_commit_feature(project.memory_dir, "feat: establish project constitution")

    typer.echo(f"✅ Constitution created at {constitution_path}")


@app.command()
def specify(
    requirements: str = typer.Option(
        ..., "--requirements", "-r", help="Feature requirements"
    ),
    feature_name: str = typer.Option(..., "--name", "-n", help="Feature name"),
):
    """Generate feature specification from requirements."""
    typer.echo(f"📋 Creating specification for: {feature_name}")

    # Create feature directory
    feature_path = create_feature_directory(feature_name)

    # Get context
    constitution = get_constitution_context()

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

    context = f"""
CONSTITUTION:
{constitution}

FEATURE: {feature_name}
REQUIREMENTS: {requirements}

TEMPLATE STRUCTURE: Use standard specification format with requirements,
acceptance criteria, constraints, and success metrics.
"""

    spec_content = ai_client.generate(system_prompt, context)

    # Write specification
    spec_path = feature_path / "spec.md"
    spec_path.write_text(spec_content)

    # Create initial task tracker
    tracker = FeatureTracker(
        feature_name=feature_name, feature_path=feature_path, tasks=[]
    )
    tracker_path = feature_path / "tracker.json"
    tracker_path.write_text(tracker.model_dump_json(indent=2))

    # Git commit
    git_commit_feature(feature_path, f"feat: specify {feature_name}")

    typer.echo(f"✅ Specification created at {spec_path}")


@app.command()
def plan(feature_name: str):
    """Generate technical implementation plan."""
    typer.echo(f"📐 Planning implementation for: {feature_name}")

    # Find feature directory
    feature_path = None
    for path in project.specs_dir.glob("???-*"):
        if path.name.endswith(feature_name.lower().replace(" ", "-")):
            feature_path = path
            break

    if not feature_path:
        typer.echo(f"❌ Feature '{feature_name}' not found")
        raise typer.Exit(1)

    # Load specification
    spec_path = feature_path / "spec.md"
    if not spec_path.exists():
        typer.echo("❌ Specification not found. Run 'specify' first.")
        raise typer.Exit(1)

    spec_content = spec_path.read_text()
    constitution = get_constitution_context()

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

    plan_content = ai_client.generate(system_prompt, context)

    # Write plan
    plan_path = feature_path / "plan.md"
    plan_path.write_text(plan_content)

    # Update tracker
    tracker_path = feature_path / "tracker.json"
    if tracker_path.exists():
        tracker_data = json.loads(tracker_path.read_text())
        tracker_data["current_phase"] = "planning"
        tracker_data["updated_at"] = datetime.now().isoformat()
        tracker_path.write_text(json.dumps(tracker_data, indent=2))

    # Git commit
    git_commit_feature(feature_path, f"feat: plan {feature_name}")

    typer.echo(f"✅ Plan created at {plan_path}")


@app.command()
def tasks(feature_name: str):
    """Break down plan into executable tasks."""
    typer.echo(f"📝 Breaking down tasks for: {feature_name}")

    # Find feature directory
    feature_path = None
    for path in project.specs_dir.glob("???-*"):
        if path.name.endswith(feature_name.lower().replace(" ", "-")):
            feature_path = path
            break

    if not feature_path:
        typer.echo(f"❌ Feature '{feature_name}' not found")
        raise typer.Exit(1)

    # Load plan
    plan_path = feature_path / "plan.md"
    if not plan_path.exists():
        typer.echo("❌ Plan not found. Run 'plan' first.")
        raise typer.Exit(1)

    plan_content = plan_path.read_text()
    constitution = get_constitution_context()

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

    tasks_content = ai_client.generate(system_prompt, context)

    # Write tasks
    tasks_path = feature_path / "tasks.md"
    tasks_path.write_text(tasks_content)

    # Update tracker with parsed tasks
    tracker_path = feature_path / "tracker.json"
    if tracker_path.exists():
        tracker_data = json.loads(tracker_path.read_text())
        tracker_data["current_phase"] = "tasking"
        tracker_data["updated_at"] = datetime.now().isoformat()
        tracker_path.write_text(json.dumps(tracker_data, indent=2))

    # Git commit
    git_commit_feature(feature_path, f"feat: tasks {feature_name}")

    typer.echo(f"✅ Tasks created at {tasks_path}")


@app.command()
def implement(feature_name: str):
    """Execute implementation tasks."""
    typer.echo(f"🚀 Implementing: {feature_name}")

    # Find feature directory
    feature_path = None
    for path in project.specs_dir.glob("???-*"):
        if path.name.endswith(feature_name.lower().replace(" ", "-")):
            feature_path = path
            break

    if not feature_path:
        typer.echo(f"❌ Feature '{feature_name}' not found")
        raise typer.Exit(1)

    # Load tracker
    tracker_path = feature_path / "tracker.json"
    if not tracker_path.exists():
        typer.echo("❌ Task tracker not found. Run 'tasks' first.")
        raise typer.Exit(1)

    tracker_data = json.loads(tracker_path.read_text())

    # For now, this is a placeholder - actual implementation would
    # parse tasks and execute them based on the plan
    typer.echo("⚠️ Implementation execution is a complex process that")
    typer.echo("   requires integration with your specific development tools")
    typer.echo("   and workflows.")
    typer.echo("   This CLI provides the foundation - you'll need to extend")
    typer.echo("   it with your project's specific implementation logic.")

    # Update tracker
    tracker_data["current_phase"] = "implementing"
    tracker_data["updated_at"] = datetime.now().isoformat()
    tracker_path.write_text(json.dumps(tracker_data, indent=2))

    typer.echo("✅ Implementation phase initialized")


if __name__ == "__main__":
    app()
