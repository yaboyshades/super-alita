#!/usr/bin/env python3
import functools
import hashlib
import inspect
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

# Remove the workspace from the Python path to avoid conflicts
sys.path = [p for p in sys.path if "super-alita-clean" not in p and "ATLAI" not in p]

# Add the virtual environment paths explicitly (resolve relative to repo root)
REPO_ROOT = Path(__file__).resolve().parent
venv_base = str(REPO_ROOT / ".venv")
sys.path.extend(
    [
        os.path.join(venv_base, "Lib", "site-packages"),
        os.path.join(venv_base, "Lib", "site-packages", "win32"),
        os.path.join(venv_base, "Lib", "site-packages", "win32", "lib"),
        os.path.join(venv_base, "Lib", "site-packages", "Pythonwin"),
    ]
)

# Import the MCP modules first
from mcp.server.fastmcp import FastMCP

# Restore the path so we can import local modules. For 'import src.*', the
# parent directory of 'src' must be on sys.path. Add both repo root and src.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
src_dir = REPO_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server")

# Try to import tools, provide fallbacks if not available
try:
    from mcp_server.tools import (
        find_missing_docstrings,
        format_and_lint,
        refactor_to_result,
    )

    logger.info("Successfully imported MCP tools")
except ImportError as e:
    logger.error(f"Failed to import MCP tools: {e}")

    # Provide fallback implementations
    async def find_missing_docstrings(
        root: str, include_tests: bool = False
    ) -> dict[str, Any]:
        return {"functions": [], "count": 0, "error": "MCP tools not available"}

    async def format_and_lint(target_path: str) -> dict[str, str]:
        return {"stdout": "", "stderr": "MCP tools not available"}

    async def refactor_to_result(
        file_path: str, function_name: str, dry_run: bool = True
    ) -> dict[str, Any]:
        return {"applied": False, "diff": "", "error": "MCP tools not available"}


app = FastMCP("myCustomPythonAgent")


TELEMETRY_FILE = Path(os.environ.get("SUPER_ALITA_TELEMETRY_FILE", "telemetry.jsonl"))


def _emit_event(event_type: str, **data: Any) -> None:
    TELEMETRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TELEMETRY_FILE.open("a", encoding="utf-8") as fp:
        json.dump({"type": event_type, **data}, fp)
        fp.write("\n")


def _telemetry_wrapper(
    name: str,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(func)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            span_id = uuid.uuid4().hex
            args_hash = hashlib.sha256(
                json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True).encode()
            ).hexdigest()
            _emit_event(
                "AbilityCalled", tool=name, span_id=span_id, args_hash=args_hash
            )
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration_ms = int((time.perf_counter() - start) * 1000)
                output_json = json.dumps(result, sort_keys=True)
                output_bytes = output_json.encode()
                output_hash = hashlib.sha256(output_bytes).hexdigest()
                if len(output_bytes) > 200_000:
                    sha = hashlib.sha256(output_bytes).hexdigest()
                    artifact_id = sha[:8]
                    artifact_path = TELEMETRY_FILE.with_name(
                        f"artifact_{artifact_id}.json"
                    )
                    artifact_path.write_bytes(output_bytes)
                    _emit_event(
                        "ArtifactCreated",
                        tool=name,
                        artifact_bytes=len(output_bytes),
                        sha256=sha,
                    )
                    result = {
                        "_artifact": {
                            "artifact_id": artifact_id,
                            "sha256": sha,
                            "bytes": len(output_bytes),
                        }
                    }
                _emit_event(
                    "AbilitySucceeded",
                    tool=name,
                    span_id=span_id,
                    duration_ms=duration_ms,
                    output_hash=output_hash,
                )
                return result
            except Exception as e:  # pragma: no cover - error path
                duration_ms = int((time.perf_counter() - start) * 1000)
                _emit_event(
                    "AbilityFailed",
                    tool=name,
                    span_id=span_id,
                    duration_ms=duration_ms,
                    error=str(e),
                )
                raise

        wrapped.__signature__ = inspect.signature(func, eval_str=False)
        return wrapped

    return decorator


_original_tool = FastMCP.tool


def _instrumented_tool(self: FastMCP, *t_args: Any, **t_kwargs: Any):
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        name = t_kwargs.get("name", func.__name__)
        return _original_tool(self, *t_args, **t_kwargs)(_telemetry_wrapper(name)(func))

    return decorator


FastMCP.tool = _instrumented_tool  # type: ignore[assignment]


@app.tool(
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


@app.tool(
    name="format_and_lint_selection",
    description=(
        "Run Ruff (fix) and Black on a path. "
        "Args: target_path (str). Returns JSON: {'stdout': str, 'stderr': str}."
    ),
)
async def format_and_lint_selection(target_path: str) -> dict[str, str]:
    return await format_and_lint(target_path=target_path)


@app.tool(
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


def main() -> None:
    transport = os.environ.get("MCP_TRANSPORT", "stdio")  # Support SSE via env var

    logger.info("Starting MCP server (transport=%s)", transport)
    if transport == "sse":
        host = os.environ.get("MCP_HOST", "127.0.0.1")
        port = int(os.environ.get("MCP_PORT", "8001"))
        logger.info("SSE server will be available at http://%s:%s", host, port)
        # Note: FastMCP may use environment variables for SSE config
        app.run(transport=transport)
    else:
        app.run(transport=transport)


if __name__ == "__main__":
    main()

# --- Decision Policy Bridge ---

from src.core.decision_policy_v1 import DecisionPolicyEngine  # noqa: E402
from src.mcp_local.registry import ToolRegistry as LocalToolRegistry  # noqa: E402


class MCPBridge:
    def __init__(self):
        self.decision_policy = DecisionPolicyEngine()
        self.mcp_registry = LocalToolRegistry()

    async def register_mcp_tools_as_capabilities(self):
        for tool_name in self.mcp_registry.list_tools():
            cap = self._convert_mcp_to_capability(
                {
                    "name": tool_name,
                    "description": f"MCP tool {tool_name}",
                    "inputSchema": {},
                }
            )
            if hasattr(self.decision_policy, "register_capability"):
                self.decision_policy.register_capability(cap)  # type: ignore[attr-defined]

    def _create_mcp_executor(self, name: str):
        async def _exec(**kwargs):
            return await self.mcp_registry.ainvoke(name, kwargs)

        return _exec

    def _convert_mcp_to_capability(self, tool_spec: dict) -> dict:
        return {
            "name": tool_spec["name"],
            "description": tool_spec.get("description", ""),
            "parameters": tool_spec.get("inputSchema", {}),
            "type": "mcp_tool",
            "executor": self._create_mcp_executor(tool_spec["name"]),
        }
