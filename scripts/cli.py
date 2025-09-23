# scripts/cli.py
"""
Python-native Spec Kit CLI using Typer and modular managers.
Provides a clean, type-safe interface for SDD workflows.
"""

import typer

from .common import Project
from .constitution_manager import ConstitutionManager
from .plan_manager import PlanManager
from .spec_manager import SpecManager
from .tasks_manager import TasksManager

# Initialize project and managers
project = Project()
constitution_mgr = ConstitutionManager(project)
spec_mgr = SpecManager(project)
plan_mgr = PlanManager(project)
tasks_mgr = TasksManager(project)

app = typer.Typer(help="Spec Kit CLI - Python-native SDD workflow management")


@app.command()
def constitution(
    principles: str = typer.Option(
        ..., "--principles", "-p", help="Project principles"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing constitution"
    ),
):
    """Create a new project constitution."""
    typer.echo("📋 Creating constitution...")

    try:
        path = constitution_mgr.create_constitution(principles, force)
        typer.echo(f"✅ Constitution created at {path}")
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def specify(
    requirements: str = typer.Option(
        ..., "--requirements", "-r", help="Feature requirements"
    ),
    feature_name: str = typer.Option(..., "--name", "-n", help="Feature name"),
    context: str = typer.Option(None, "--context", "-c", help="Additional context"),
):
    """Generate feature specification from requirements."""
    typer.echo(f"📋 Creating specification for: {feature_name}")

    try:
        path = spec_mgr.create_specification(feature_name, requirements, context)
        typer.echo(f"✅ Specification created at {path}")
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def plan(feature_name: str):
    """Generate technical implementation plan."""
    typer.echo(f"📐 Planning implementation for: {feature_name}")

    try:
        path = plan_mgr.create_plan(feature_name)
        typer.echo(f"✅ Plan created at {path}")
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def tasks(feature_name: str):
    """Break down plan into executable tasks."""
    typer.echo(f"📝 Breaking down tasks for: {feature_name}")

    try:
        path = tasks_mgr.create_tasks(feature_name)
        typer.echo(f"✅ Tasks created at {path}")
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def implement(feature_name: str):
    """Execute implementation tasks."""
    typer.echo(f"🚀 Implementing: {feature_name}")

    # Find feature directory
    feature_path = project.get_feature_path(feature_name)
    if not feature_path:
        typer.echo(f"❌ Feature '{feature_name}' not found")
        raise typer.Exit(1)

    # Load tracker
    tracker_path = feature_path / "tracker.json"
    if not tracker_path.exists():
        typer.echo("❌ Task tracker not found. Run 'tasks' first.")
        raise typer.Exit(1)

    import json

    tracker_data = json.loads(tracker_path.read_text())

    # For now, this is a placeholder - actual implementation would
    # parse tasks and execute them based on the plan
    typer.echo("⚠️ Implementation execution is a complex process that")
    typer.echo("   requires integration with your specific development tools")
    typer.echo("   and workflows.")
    typer.echo("   This CLI provides the foundation - you'll need to extend")
    typer.echo("   it with your project's specific implementation logic.")

    # Update tracker
    from datetime import datetime

    tracker_data["current_phase"] = "implementing"
    tracker_data["updated_at"] = datetime.now().isoformat()
    tracker_path.write_text(json.dumps(tracker_data, indent=2))

    typer.echo("✅ Implementation phase initialized")


@app.command()
def index(
    project_path: str = typer.Option(
        ".", "--path", "-p", help="Path to project to index"
    ),
    force_reindex: bool = typer.Option(
        False, "--force", "-f", help="Force reindexing of all files"
    ),
):
    """Index project codebase into knowledge graph."""
    typer.echo(f"🧠 Indexing project: {project_path}")

    try:
        # Import here to avoid circular imports
        from pathlib import Path

        from .knowledge_base.graph_db_client import GraphDatabaseClient
        from .knowledge_base.neural_indexer import NeuralCodeIndexer

        # Initialize components
        # Note: In production, get these from environment/config
        db_client = GraphDatabaseClient(
            uri="bolt://localhost:7687", user="neo4j", password="password"
        )

        indexer = NeuralCodeIndexer(db_client)

        # Connect to database
        typer.echo("🔌 Connecting to knowledge graph...")
        await db_client.connect()

        # Index the project
        typer.echo("📊 Analyzing codebase...")
        results = await indexer.index_project_with_evolution(
            Path(project_path), force_reindex
        )

        # Display results
        typer.echo("✅ Indexing complete!")
        typer.echo(f"📁 Files processed: {results['metrics']['files_processed']}")
        typer.echo(f"🏗️ Entities created: {results['metrics']['entities_created']}")
        typer.echo(f"🧠 Neural atoms: {results['metrics']['neural_atoms_generated']}")
        typer.echo(f"🔧 Generated MCPs: {len(results['generated_mcps'])}")

        if results["generated_mcps"]:
            typer.echo("💡 Auto-generated MCP suggestions:")
            for mcp in results["generated_mcps"]:
                typer.echo(f"   - {mcp['name']}: {mcp['description']}")

    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1) from e
    finally:
        # Cleanup
        if "db_client" in locals():
            await db_client.disconnect()


if __name__ == "__main__":
    app()
