"""
Enhanced CLI with Copilot-like commands for SDD framework.

This module provides a command-line interface that integrates:
- Natural language queries about the codebase
- Constitutional compliance validation
- Code-to-specification traceability
- Quality analysis and recommendations
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import click

from .enhanced_sdd_framework import EnhancedSDDFramework
from .models import PlanRequest, SpecifyRequest, TasksRequest
from .session.factory import FeatureSessionFactory

logger = logging.getLogger(__name__)


def _derive_feature_id_from_path(path: str) -> str:
    """Derive feature ID from a specification or plan path."""
    from pathlib import Path

    path_obj = Path(path)
    # Look for pattern like specs/020-feature-name/feature-spec.md
    if "specs" in path_obj.parts:
        specs_index = path_obj.parts.index("specs")
        if specs_index + 1 < len(path_obj.parts):
            feature_dir = path_obj.parts[specs_index + 1]
            # Extract feature ID (first 3 digits)
            if len(feature_dir) >= 3 and feature_dir[:3].isdigit():
                return feature_dir[:3]

    # Fallback to default
    return "001"


class CLIFormatter:
    """Utility class for formatting CLI output."""

    @staticmethod
    def format_table(data: list[dict], headers: list[str]) -> str:
        """Format data as a simple table."""
        if not data:
            return "No data to display."

        # Calculate column widths
        widths = {header: len(header) for header in headers}
        for row in data:
            for header in headers:
                widths[header] = max(
                    widths[header], len(str(row.get(header, "")))
                )

        # Create table
        lines = []

        # Header
        header_line = " | ".join(h.ljust(widths[h]) for h in headers)
        lines.append(header_line)
        lines.append("-" * len(header_line))

        # Rows
        for row in data:
            row_line = " | ".join(
                str(row.get(h, "")).ljust(widths[h]) for h in headers
            )
            lines.append(row_line)

        return "\n".join(lines)

    @staticmethod
    def format_list(items: list[str], title: str | None = None) -> str:
        """Format a list of items."""
        if not items:
            return f"{title}: None" if title else "No items."

        lines = []
        if title:
            lines.append(f"{title}:")

        for i, item in enumerate(items, 1):
            lines.append(f"  {i}. {item}")

        return "\n".join(lines)

    @staticmethod
    def format_dict(
        data: dict, title: str | None = None, indent: int = 0
    ) -> str:
        """Format a dictionary for display."""
        lines = []
        prefix = "  " * indent

        if title:
            lines.append(f"{prefix}{title}:")
            prefix += "  "

        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(
                    CLIFormatter.format_dict(value, indent=indent + 1)
                )
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}: {len(value)} items")
                for item in value[:3]:  # Show first 3 items
                    lines.append(f"{prefix}  - {item}")
                if len(value) > 3:
                    lines.append(f"{prefix}  ... and {len(value) - 3} more")
            else:
                lines.append(f"{prefix}{key}: {value}")

        return "\n".join(lines)


@click.group()
@click.option(
    "--workspace", "-w", default=".", help="Workspace root directory"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, workspace, verbose):
    """Enhanced SDD Framework CLI with Mangle reasoning."""
    # Setup logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Initialize framework
    try:
        framework = EnhancedSDDFramework(workspace_root=Path(workspace))
        session_factory = FeatureSessionFactory(workspace_root=Path(workspace))
        ctx.ensure_object(dict)
        ctx.obj["framework"] = framework
        ctx.obj["session_factory"] = session_factory
        ctx.obj["workspace"] = workspace
    except Exception as e:
        click.echo(f"Error initializing framework: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("question")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.pass_context
def ask(ctx, question, output_format):
    """Ask a natural language question about the codebase."""
    framework = ctx.obj["framework"]

    try:
        result = framework.ask_question(question)

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"Question: {result['question']}")
            click.echo(f"Success: {result['success']}")

            if result["success"]:
                click.echo(f"Results: {len(result['results'])} found")
                if result["results"]:
                    for i, item in enumerate(result["results"][:10], 1):
                        if isinstance(item, list):
                            click.echo(
                                f"  {i}. {', '.join(str(x) for x in item)}"
                            )
                        else:
                            click.echo(f"  {i}. {item}")

                    if len(result["results"]) > 10:
                        click.echo(
                            f"  ... and {len(result['results']) - 10} more"
                        )
                else:
                    click.echo("  No results found.")
            else:
                click.echo(
                    f"Error: {result.get('query_used', 'Unknown error')}"
                )

            click.echo(f"Execution time: {result['execution_time']:.3f}s")

    except Exception as e:
        click.echo(f"Error processing question: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.option("--article", help="Check specific constitutional article only")
@click.pass_context
def validate(ctx, output_format, article):
    """Validate code against constitutional rules."""
    framework = ctx.obj["framework"]

    try:
        click.echo("Running constitutional compliance validation...")
        results = framework.validate_constitutional_compliance()

        if output_format == "json":
            click.echo(json.dumps(results, indent=2))
        else:
            # Display summary
            summary = results.get("summary", {})
            click.echo("\nConstitutional Compliance Report")
            click.echo("=" * 40)
            click.echo(
                f"Overall Status: {summary.get('overall_compliance', 'UNKNOWN')}"
            )
            click.echo(
                f"Total Violations: {summary.get('total_violations', 0)}"
            )
            click.echo(
                f"Articles with Violations: {summary.get('articles_with_violations', 0)}"
            )

            if summary.get("traditional_score"):
                click.echo(
                    f"Traditional Score: {summary['traditional_score']:.2f}"
                )

            # Display violations by article
            mangle_analysis = results.get("mangle_analysis", {})
            for article_name, article_data in mangle_analysis.items():
                if article and article.lower() not in article_name.lower():
                    continue

                click.echo(f"\n{article_name}:")
                if article_data["success"]:
                    violations = article_data.get("violations", [])
                    if violations:
                        click.echo(f"  Violations ({len(violations)}):")
                        for violation in violations[:5]:  # Show first 5
                            if isinstance(violation, list):
                                click.echo(
                                    f"    - {', '.join(str(v) for v in violation)}"
                                )
                            else:
                                click.echo(f"    - {violation}")
                        if len(violations) > 5:
                            click.echo(
                                f"    ... and {len(violations) - 5} more"
                            )
                    else:
                        click.echo("  ✓ No violations found")
                else:
                    click.echo(
                        f"  ✗ Analysis failed: {article_data.get('error', 'Unknown error')}"
                    )

            # Display recommendations
            recommendations = results.get("recommendations", [])
            if recommendations:
                click.echo("\nRecommendations:")
                for i, rec in enumerate(recommendations, 1):
                    click.echo(f"  {i}. {rec}")

    except Exception as e:
        click.echo(f"Error running validation: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("code_element")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.pass_context
def trace(ctx, code_element, output_format):
    """Trace code element back to specification."""
    framework = ctx.obj["framework"]

    try:
        result = framework.trace_code_to_spec(code_element)

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"Tracing: {code_element}")

            if result["success"]:
                if result["traceability_found"]:
                    click.echo(
                        f"Related specifications ({len(result['related_specs'])}):"
                    )
                    for spec in result["related_specs"]:
                        if isinstance(spec, list):
                            click.echo(
                                f"  - Feature: {spec[0] if spec else 'Unknown'}"
                            )
                        else:
                            click.echo(f"  - {spec}")
                else:
                    click.echo("No specification traceability found.")
            else:
                click.echo("Trace analysis failed.")

            click.echo(f"Execution time: {result['execution_time']:.3f}s")

    except Exception as e:
        click.echo(f"Error tracing code element: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.option(
    "--category",
    type=click.Choice(["quality", "incomplete", "all"]),
    default="all",
    help="Analysis category",
)
@click.pass_context
def analyze(ctx, output_format, category):
    """Perform comprehensive code quality analysis."""
    framework = ctx.obj["framework"]

    try:
        click.echo("Running code quality analysis...")

        if category in ["quality", "all"]:
            quality_results = framework.analyze_code_quality()

            if output_format == "json":
                click.echo(json.dumps(quality_results, indent=2))
            else:
                # Display quality metrics
                metrics = quality_results.get("quality_metrics", {})
                click.echo("\nCode Quality Metrics")
                click.echo("=" * 30)
                click.echo(
                    f"Quality Score: {metrics.get('quality_score', 0)}/100"
                )
                click.echo(
                    f"Completeness Score: {metrics.get('completeness_score', 0)}/100"
                )
                click.echo(
                    f"Total Issues: {metrics.get('total_quality_issues', 0)}"
                )
                click.echo(
                    f"Incomplete Items: {metrics.get('total_incomplete_items', 0)}"
                )

                # Display specific issues
                quality_issues = quality_results.get("quality_issues", {})
                for issue_type, issue_data in quality_issues.items():
                    if issue_data["count"] > 0:
                        click.echo(f"\n{issue_type} ({issue_data['count']}):")
                        for item in issue_data["items"][:3]:  # Show first 3
                            if isinstance(item, list):
                                click.echo(
                                    f"  - {', '.join(str(x) for x in item)}"
                                )
                            else:
                                click.echo(f"  - {item}")
                        if issue_data["count"] > 3:
                            click.echo(
                                f"  ... and {issue_data['count'] - 3} more"
                            )

                # Display recommendations
                recommendations = quality_results.get("recommendations", [])
                if recommendations:
                    click.echo("\nRecommendations:")
                    for i, rec in enumerate(recommendations, 1):
                        click.echo(f"  {i}. {rec}")

    except Exception as e:
        click.echo(f"Error running analysis: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.pass_context
def stats(ctx, output_format):
    """Show knowledge graph statistics."""
    framework = ctx.obj["framework"]

    try:
        stats_data = framework.get_fact_statistics()

        if output_format == "json":
            click.echo(json.dumps(stats_data, indent=2))
        else:
            click.echo("Knowledge Graph Statistics")
            click.echo("=" * 30)
            click.echo(f"Total Facts: {stats_data.get('total_facts', 0)}")
            click.echo(
                f"Facts Size: {stats_data.get('facts_size_bytes', 0)} bytes"
            )
            click.echo(f"Cache Valid: {stats_data.get('cache_valid', False)}")
            click.echo(
                f"Cached Queries: {stats_data.get('cached_queries', 0)}"
            )

            fact_types = stats_data.get("fact_types", {})
            if fact_types:
                click.echo("\nFact Types:")
                for fact_type, count in sorted(fact_types.items()):
                    click.echo(f"  {fact_type}: {count}")

    except Exception as e:
        click.echo(f"Error getting statistics: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def clear_cache(ctx):
    """Clear all caches to force regeneration."""
    framework = ctx.obj["framework"]

    try:
        framework.invalidate_caches()
        click.echo("All caches cleared successfully.")
    except Exception as e:
        click.echo(f"Error clearing caches: {e}", err=True)
        sys.exit(1)


# Keep existing SDD commands (specify, plan, tasks)
@cli.command()
@click.argument("description")
@click.option("--context", help="Additional context for the feature")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.pass_context
def specify(ctx, description, context, output_format):
    """Create a new feature specification with enhanced analysis."""
    session_factory = ctx.obj["session_factory"]

    try:
        # Coerce --context (string) into a dict expected by the model
        ctx_dict: dict[str, object] = {}
        if context:
            try:
                loaded = json.loads(context)
                ctx_dict = (
                    loaded if isinstance(loaded, dict) else {"context": loaded}
                )
            except Exception:
                ctx_dict = {"notes": context}

        request = SpecifyRequest(user_input=description, context=ctx_dict)

        click.echo("Creating feature specification with enhanced analysis...")
        # Create session and run specify phase
        session = session_factory.for_description(description, ctx_dict)
        result = asyncio.run(session.specify(request))

        if output_format == "json":
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
            click.echo(f"Feature ID: {result.feature_id}")
            click.echo(f"Specification File: {result.artifact_path}")
            if result.metadata_path:
                click.echo(f"Next-Step Metadata: {result.metadata_path}")
            if result.guidance:
                click.echo("\nNext Steps Guidance:")
                click.echo(
                    f"  Clarifications: {len(result.guidance.clarifications)}"
                )
                click.echo(f"  Artefacts: {len(result.guidance.artefacts)}")
                click.echo(f"  Commands: {len(result.guidance.commands)}")

            # Show constitutional validation
            click.echo(
                f"\nConstitutional Compliance: {result.overall_compliance_score:.2f}"
            )
            if not result.compliance_threshold_met:
                click.echo("  ⚠️  Compliance threshold not met")

            if result.next_steps:
                click.echo("\nNext Steps:")
                for step in result.next_steps[:3]:
                    click.echo(f"  - {step}")
                if len(result.next_steps) > 3:
                    click.echo(f"  ... and {len(result.next_steps) - 3} more")

    except Exception as e:  # noqa: BLE001
        click.echo(f"Error creating specification: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("specification_path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.pass_context
def plan(ctx, specification_path, output_format):
    """Generate implementation plan with enhanced dependency analysis."""
    session_factory = ctx.obj["session_factory"]

    try:
        request = PlanRequest(specification_path=specification_path)

        click.echo("Generating implementation plan with enhanced analysis...")
        # Load existing session by deriving feature ID from spec path
        feature_id = _derive_feature_id_from_path(specification_path)
        session = session_factory.for_feature_id(feature_id)
        result = asyncio.run(session.plan(request))

        if output_format == "json":
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
            click.echo(f"Feature ID: {result.feature_id}")
            click.echo(f"Implementation Plan: {result.artifact_path}")
            if result.metadata_path:
                click.echo(f"Next-Step Metadata: {result.metadata_path}")
            if result.guidance:
                click.echo("\nNext Steps Guidance:")
                click.echo(
                    f"  Clarifications: {len(result.guidance.clarifications)}"
                )
                click.echo(f"  Artefacts: {len(result.guidance.artefacts)}")
                click.echo(f"  Commands: {len(result.guidance.commands)}")

            # Show constitutional validation
            click.echo(
                f"\nConstitutional Compliance: {result.overall_compliance_score:.2f}"
            )
            if not result.compliance_threshold_met:
                click.echo("  ⚠️  Compliance threshold not met")

            if result.next_steps:
                click.echo("\nNext Steps:")
                for step in result.next_steps[:3]:
                    click.echo(f"  - {step}")
                if len(result.next_steps) > 3:
                    click.echo(f"  ... and {len(result.next_steps) - 3} more")

    except Exception as e:  # noqa: BLE001
        click.echo(f"Error generating plan: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("plan_path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.pass_context
def tasks(ctx, plan_path, output_format):
    """Generate implementation tasks with enhanced prioritization."""
    session_factory = ctx.obj["session_factory"]

    try:
        request = TasksRequest(plan_path=plan_path)

        click.echo(
            "Generating implementation tasks with enhanced prioritization..."
        )
        # Load existing session by deriving feature ID from plan path
        feature_id = _derive_feature_id_from_path(plan_path)
        session = session_factory.for_feature_id(feature_id)
        result = asyncio.run(session.tasks(request))

        if output_format == "json":
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
            click.echo(f"Feature ID: {result.feature_id}")
            click.echo(f"Generated {len(result.next_steps)} tasks")
            if result.metadata_path:
                click.echo(f"Next-Step Metadata: {result.metadata_path}")
            if result.guidance:
                click.echo("\nNext Steps Guidance:")
                click.echo(
                    f"  Clarifications: {len(result.guidance.clarifications)}"
                )
                click.echo(f"  Artefacts: {len(result.guidance.artefacts)}")
                click.echo(f"  Commands: {len(result.guidance.commands)}")

            # Show constitutional validation
            click.echo(
                f"\nConstitutional Compliance: {result.overall_compliance_score:.2f}"
            )
            if not result.compliance_threshold_met:
                click.echo("  ⚠️  Compliance threshold not met")

            if result.next_steps:
                click.echo("\nNext Steps:")
                for step in result.next_steps[:3]:
                    click.echo(f"  - {step}")
                if len(result.next_steps) > 3:
                    click.echo(f"  ... and {len(result.next_steps) - 3} more")

    except Exception as e:  # noqa: BLE001
        click.echo(f"Error generating tasks: {e}", err=True)
        sys.exit(1)


# Helper commands for common questions
@cli.command()
@click.pass_context
def untested(ctx):
    """Show untested functions."""
    framework = ctx.obj["framework"]
    result = framework.ask_question("what functions are untested")

    if result["success"] and result["results"]:
        click.echo(f"Untested functions ({len(result['results'])}):")
        for func in result["results"]:
            click.echo(f"  - {func[0] if isinstance(func, list) else func}")
    else:
        click.echo("No untested functions found or analysis failed.")


@cli.command()
@click.pass_context
def incomplete(ctx):
    """Show incomplete features."""
    framework = ctx.obj["framework"]
    result = framework.ask_question("what features are incomplete")

    if result["success"] and result["results"]:
        click.echo(f"Incomplete features ({len(result['results'])}):")
        for feature in result["results"]:
            click.echo(
                f"  - {feature[0] if isinstance(feature, list) else feature}"
            )
    else:
        click.echo("No incomplete features found or analysis failed.")


if __name__ == "__main__":
    cli()
