
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP, tool  # part of MCP Python SDK
from src.knowledge_graph.kg_interface import KnowledgeGraphInterface
from src.knowledge_graph.models import EntityType, KnowledgeQuery
from src.reug_runtime.message_mw import MessageContext, apply_all

from .github_tools import register_github_tools
from .tools import find_missing_docstrings, format_and_lint, refactor_to_result

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server")

app = FastMCP("myCustomPythonAgent")

_KG = KnowledgeGraphInterface()


@tool(
    name="apply_result_pattern_refactor",
    description=(
        "Refactor a Python function to a Result-returning pattern. "
        "Args: file_path (str), function_name (str), dry_run (bool, default true). "
        "Returns JSON: {'applied': bool, 'diff': str, 'error': Optional[str]}."
    ),
)
async def apply_result_pattern_refactor(
    file_path: str, function_name: str, dry_run: bool = True
) -> dict[str, Any]:
    return await refactor_to_result(
        file_path=file_path, function_name=function_name, dry_run=dry_run
    )


@tool(
    name="format_and_lint_selection",
    description=(
        "Run Ruff (fix) and Black on a path. "
        "Args: target_path (str). Returns JSON: {'stdout': str, 'stderr': str}."
    ),
)
async def format_and_lint_selection(target_path: str) -> dict[str, str]:
    return await format_and_lint(target_path=target_path)


@tool(
    name="find_missing_docstrings",
    description=(
        "Find functions missing docstrings under a root dir. "
        "Args: root (str), include_tests (bool, default false). "
        "Returns JSON: {'functions': [{'file': str, 'line': int, 'name': str}], 'count': int}."
    ),
)
async def find_missing_docstrings_tool(
    root: str, include_tests: bool = False
) -> dict[str, Any]:
    return await find_missing_docstrings(root=root, include_tests=include_tests)


@app.tool(
    name="trigger_cognitive_turn",
    description=(
        "Run the registered message middleware against a query and report the resulting transformations."
    ),
)
async def trigger_cognitive_turn(
    query: str, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    ctx = context or {}
    session_id = str(ctx.get("session_id", "mcp-session"))
    optimized, steps = apply_all(query, MessageContext(session_id=session_id))
    return {
        "query": query,
        "optimized_query": optimized,
        "middleware_steps": steps,
        "context_keys": sorted(ctx.keys()),
    }


@app.tool(
    name="query_knowledge_graph",
    description="Query the lightweight in-memory knowledge graph for planning patterns.",
)
async def query_knowledge_graph(
    query_type: str, query: str, max_results: int = 10
) -> dict[str, Any]:
    kg_query = KnowledgeQuery(goal=query, max_results=max_results)
    match query_type:
        case "semantic":
            kg_query.include_patterns = True
        case "structural":
            kg_query.include_patterns = False
            kg_query.entity_types = {EntityType.DOMAIN, EntityType.TASK}
        case "temporal":
            kg_query.include_patterns = False
        case _:
            raise ValueError("query_type must be semantic, structural, or temporal")

    result = _KG.query(kg_query)
    entities = [
        {
            "id": entity.id,
            "name": entity.name,
            "entity_type": entity.entity_type.value,
            "confidence": entity.confidence,
        }
        for entity in result.entities
    ]
    patterns = [
        {
            "id": pattern.id,
            "pattern": pattern.pattern_name,
            "domain": pattern.domain,
            "success_rate": pattern.success_rate,
            "steps": pattern.decomposition_steps,
        }
        for pattern in result.patterns
    ]
    return {
        "entities": entities,
        "patterns": patterns,
        "relevance": result.relevance_scores,
        "total_results": result.total_results,
        "query_time": result.query_time,
        "graph_stats": _KG.get_statistics(),
    }


@app.tool(
    name="get_system_telemetry",
    description="Return lightweight telemetry snapshots derived from the local workspace.",
)
async def get_system_telemetry(metric_type: str, time_range: str = "1h") -> dict[str, Any]:
    metric_type = metric_type or "all"
    snapshot: dict[str, Any] = {
        "metric_type": metric_type,
        "time_range": time_range,
        "generated_at": time.time(),
    }
    if metric_type in {"performance", "all"}:
        snapshot["performance"] = _gather_performance_metrics()
    if metric_type in {"llm_usage", "all"}:
        snapshot["llm_usage"] = _gather_llm_usage()
    if metric_type in {"events", "all"}:
        snapshot["events"] = _gather_event_log_metrics()
    return snapshot


def _gather_performance_metrics() -> dict[str, Any]:
    src_root = Path("src")
    tests_root = Path("tests")
    python_files = sum(1 for _ in src_root.rglob("*.py")) if src_root.exists() else 0
    test_modules = sum(1 for _ in tests_root.rglob("test_*.py")) if tests_root.exists() else 0
    return {
        "python_files": python_files,
        "test_modules": test_modules,
    }


def _gather_llm_usage() -> dict[str, str]:
    keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "LLM_MODEL"]
    status: dict[str, str] = {}
    for key in keys:
        status[key] = "configured" if os.getenv(key) else "unset"
    return status


def _gather_event_log_metrics() -> dict[str, Any]:
    events_dir = Path("logs/events")
    if not events_dir.exists():
        return {"available": False}
    files = list(events_dir.glob("*.jsonl"))
    total_size = sum(f.stat().st_size for f in files)
    return {
        "available": True,
        "files": len(files),
        "total_size_bytes": total_size,
    }


register_github_tools(app)


def main() -> None:
    transport = "stdio"  # VS Code launches this as a subprocess
    logger.info("Starting MCP server (transport=%s)", transport)
    app.run(transport=transport)


if __name__ == "__main__":
    main()
