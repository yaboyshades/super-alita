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
    # Import the actual functions from the MCP server tools
    sys.path.insert(0, str(REPO_ROOT / "mcp_server" / "src"))

    # Try importing directly from specific modules to avoid dynamic import issues
    from mcp_server.tools.format_and_scan import (
        find_missing_docstrings,
        format_and_lint_selection,
    )

    # Rename to avoid conflicts with our tool wrappers
    format_and_lint = format_and_lint_selection

    logger.info("Successfully imported MCP tools")
except ImportError as e:
    logger.error(f"Failed to import MCP tools: {e}")

    # Provide fallback implementations
    async def find_missing_docstrings(
        root: str, include_tests: bool = False
    ) -> dict[str, Any]:
        return {"functions": [], "count": 0, "error": "MCP tools not available"}

    async def format_and_lint_selection(target_path: str) -> dict[str, str]:
        return {"stdout": "", "stderr": "MCP tools not available"}

    # Alias for consistency
    format_and_lint = format_and_lint_selection


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

        try:
            wrapped.__signature__ = inspect.signature(func, eval_str=False)  # type: ignore
        except (AttributeError, TypeError):
            # Some function types don't support signature assignment
            pass
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
    name="format_and_lint_selection",
    description=(
        "Run Ruff (fix) and Black on a path. "
        "Args: target_path (str). Returns JSON: {'stdout': str, 'stderr': str}."
    ),
)
async def format_and_lint_selection_tool(target_path: str) -> dict[str, str]:
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

# ---- Filesystem helpers (repo-first, secure) ----
REPO_ROOT = Path(__file__).resolve().parent

def _resolve_path(p: str) -> Path:
  base = REPO_ROOT
  target = (base / p).resolve()
  if not str(target).startswith(str(base)):
    raise ValueError("Path escapes repository root")
  return target

@app.tool(
    name="read_file",
    description="Read a text file from the repository. Args: path (str), encoding (str, optional)."
)
async def read_file(path: str, encoding: str = "utf-8") -> dict[str, Any]:
    fp = _resolve_path(path)
    data = fp.read_text(encoding=encoding)
    return {"path": str(fp), "content": data}

@app.tool(
    name="create_directory",
    description="Create a directory (and parents). Args: path (str)."
)
async def create_directory(path: str) -> dict[str, Any]:
    fp = _resolve_path(path)
    fp.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "path": str(fp)}

@app.tool(
    name="create_file",
    description=(
        "Create a new file with content. Args: path (str), content (str), create_dirs (bool, default true), encoding (str)."
    ),
)
async def create_file(path: str, content: str, create_dirs: bool = True, encoding: str = "utf-8") -> dict[str, Any]:
    fp = _resolve_path(path)
    if create_dirs:
        fp.parent.mkdir(parents=True, exist_ok=True)
    if fp.exists():
        return {"ok": False, "error": "exists", "path": str(fp)}
    fp.write_text(content, encoding=encoding)
    return {"ok": True, "path": str(fp), "bytes": len(content.encode(encoding))}

@app.tool(
    name="edit_file",
    description=(
        "Overwrite or create a file with new content. Args: path (str), content (str), create_dirs (bool, default true), encoding (str)."
    ),
)
async def edit_file(path: str, content: str, create_dirs: bool = True, encoding: str = "utf-8") -> dict[str, Any]:
    fp = _resolve_path(path)
    if create_dirs:
        fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content, encoding=encoding)
    return {"ok": True, "path": str(fp), "bytes": len(content.encode(encoding))}

@app.tool(
    name="delete_file",
    description="Delete a file. Args: path (str)."
)
async def delete_file(path: str) -> dict[str, Any]:
    fp = _resolve_path(path)
    if not fp.exists():
        return {"ok": False, "error": "missing", "path": str(fp)}
    fp.unlink()
    return {"ok": True, "path": str(fp)}

@app.tool(
    name="rename_file",
    description="Rename or move a file. Args: src (str), dest (str), create_dirs (bool, default true)."
)
async def rename_file(src: str, dest: str, create_dirs: bool = True) -> dict[str, Any]:
    sp = _resolve_path(src)
    dp = _resolve_path(dest)
    if create_dirs:
        dp.parent.mkdir(parents=True, exist_ok=True)
    sp.replace(dp)
    return {"ok": True, "from": str(sp), "to": str(dp)}

@app.tool(
    name="list_directory",
    description="List directory contents. Args: path (str). Returns names and types."
)
async def list_directory(path: str) -> dict[str, Any]:
    fp = _resolve_path(path)
    if not fp.exists() or not fp.is_dir():
        return {"ok": False, "error": "not_a_directory", "path": str(fp)}
    items = []
    for child in fp.iterdir():
        try:
            items.append({"name": child.name, "is_dir": child.is_dir(), "size": child.stat().st_size})
        except Exception:
            items.append({"name": child.name, "is_dir": child.is_dir(), "size": None})
    return {"ok": True, "path": str(fp), "items": items}


# ---- Comprehensive MANGLE Reasoning Tools (stubs, repo-first) ----
@app.tool(
    name="mangle_spec_reason",
    description=(
        "Apply MANGLE reasoning to specifications. "
        "Args: spec_content(str), reasoning_type(str: validate|enhance|analyze), domain_rules(list[str])."
    ),
)
async def mangle_spec_reason(
    spec_content: str,
    reasoning_type: str = "analyze",
    domain_rules: list[str] | None = None,
) -> dict[str, Any]:
    """Stubbed reasoning tool. Echoes inputs and notes missing MANGLE CLI.

    This integrates with repo-first workflow so Copilot can call the tool.
    Replace with real integration using src/core/proc.py to invoke MANGLE.
    """
    return {
        "reasoning_type": reasoning_type,
        "facts_extracted": 0,
        "rules_applied": len(domain_rules or []),
        "insights": [
            {
                "title": "Reasoning stub",
                "detail": "MANGLE CLI not integrated; returning placeholder analysis.",
            }
        ],
        "echo": {"spec_content": spec_content[:500], "domain_rules": domain_rules or []},
    }


@app.tool(
    name="mangle_plan_validate",
    description=(
        "Validate planning decisions using deductive reasoning. "
        "Args: plan_facts(list[str]), spec_constraints(list[str]), validation_rules(list[str])."
    ),
)
async def mangle_plan_validate(
    plan_facts: list[str] | None = None,
    spec_constraints: list[str] | None = None,
    validation_rules: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "feasible": False,
        "violations": [],
        "summary": "Reasoning stub — provide MANGLE to enable real validation.",
        "echo": {
            "plan_facts": plan_facts or [],
            "spec_constraints": spec_constraints or [],
            "validation_rules": validation_rules or [],
        },
    }


@app.tool(
    name="mangle_task_optimize",
    description=(
        "Optimize task sequences using MANGLE reasoning. "
        "Args: tasks(list[dict]), dependencies(list[str]), constraints(list[str])."
    ),
)
async def mangle_task_optimize(
    tasks: list[dict[str, Any]] | None = None,
    dependencies: list[str] | None = None,
    constraints: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "optimal_sequence": [t.get("id") for t in (tasks or [])],
        "ready_to_start": [t.get("id") for t in (tasks or [])[:1]],
        "notes": [
            "Reasoning stub — plug in MANGLE for real optimization.",
        ],
        "echo": {
            "tasks": tasks or [],
            "dependencies": dependencies or [],
            "constraints": constraints or [],
        },
    }


@app.tool(
    name="mangle_cross_phase_verify",
    description=(
        "Verify consistency across development phases. "
        "Args: phase1_facts(list[str]), phase2_facts(list[str]), consistency_rules(list[str])."
    ),
)
async def mangle_cross_phase_verify(
    phase1_facts: list[str] | None = None,
    phase2_facts: list[str] | None = None,
    consistency_rules: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "consistent": True,
        "issues": [],
        "echo": {
            "phase1_facts": phase1_facts or [],
            "phase2_facts": phase2_facts or [],
            "consistency_rules": consistency_rules or [],
        },
        "note": "Reasoning stub — implement rule checks via MANGLE to enable.",
    }


@app.tool(
    name="mangle_living_doc_update",
    description=(
        "Update living documents with reasoning insights. "
        "Args: document_path(str), current_facts(list[str]), reasoning_updates(object)."
    ),
)
async def mangle_living_doc_update(
    document_path: str,
    current_facts: list[str] | None = None,
    reasoning_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "document_path": document_path,
        "updated": False,
        "changes": [],
        "note": "Reasoning stub — wire to updater once MANGLE is available.",
        "echo": {"current_facts": current_facts or [], "reasoning_updates": reasoning_updates or {}},
    }


def main() -> None:
    transport_env = os.environ.get("MCP_TRANSPORT", "stdio")
    # Ensure transport is a valid literal
    transport = "stdio" if transport_env not in ["stdio", "sse"] else transport_env

    logger.info("Starting MCP server (transport=%s)", transport)
    if transport == "sse":
        host = os.environ.get("MCP_HOST", "127.0.0.1")
        port = int(os.environ.get("MCP_PORT", "8001"))
        logger.info("SSE server will be available at http://%s:%s", host, port)

    app.run(transport=transport)  # type: ignore


if __name__ == "__main__":
    main()
