"""FastAPI routes exposing tool-style endpoints for the REUG runtime.

Alignment additions (ALITA minimal predefinition + maximal self‑evolution):
- /tools/mcp/brainstorm: propose lightweight MCP-like tool specs for a task
- /tools/mcp/register: persist and dynamically register a new tool spec

These endpoints allow the runtime to evolve capabilities at runtime without
predefining large toolsets. Generated specs are persisted under ./.mcp_box.

This module has been refactored to use the centralized ToolCatalogService
for consistent tool catalog management and dynamic registration.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import SETTINGS
from .loop import execute_turn
from .mcp_abstractor import abstract_mcp_box
from .tools.service import ToolCatalogService

tools = APIRouter(prefix="/tools", tags=["tools"])

# Optional ability execution router (compat alias for external scripts)
ability = APIRouter(prefix="/ability", tags=["ability"])

_STREAMS: dict[str, Any] = {}

# Shared catalog service instance
_catalog_service = ToolCatalogService()


@tools.get("/catalog")
async def get_catalog(request: Request) -> JSONResponse:
    """Return the tool catalog including both static and dynamic tools."""
    try:
        app = request.app
        registry = getattr(app.state, "ability_registry", None)
        catalog = _catalog_service.list_tools(registry)
        return JSONResponse(catalog)
    except Exception as e:
        # If there's any error accessing tools, return empty catalog with error info
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to load tool catalog: {str(e)}", "tools": []}
        )


@tools.post("/reug_start_turn")
async def reug_start_turn(
    request: Request,
    body: dict[str, Any] = Body(...),  # noqa: B008
) -> dict[str, Any]:
    """Start a new streaming turn.

    Args:
        request: Incoming FastAPI request object.
        body: JSON body containing ``message`` and optional ``session_id``.

    Returns:
        Metadata about the started run including the ``run_id``.
    """
    message = body["message"]
    session_id = body.get("session_id", "default")
    state = cast(Any, request.app.state)
    gen = execute_turn(
        message,
        session_id,
        event_bus=state.event_bus,
        registry=state.ability_registry,
        kg=state.kg,
        model=state.llm_model,
        output_validator=getattr(state, "output_validator", None),
    )
    run_id = f"run_{hash((message, session_id)) & 0xffff_ffff:x}"
    _STREAMS[run_id] = gen.__aiter__()
    return {"run_id": run_id, "stream_begun": True}


@tools.post("/reug_stream_next")
async def reug_stream_next(
    body: dict[str, Any] = Body(...),  # noqa: B008
) -> dict[str, Any]:
    """Fetch the next streamed chunks for an active run.

    Args:
        body: JSON body containing the ``run_id``.

    Returns:
        A dictionary with streamed ``chunks`` and a ``finished`` flag.
    """
    run_id = body["run_id"]
    it = _STREAMS.get(run_id)
    if it is None:
        raise HTTPException(status_code=404, detail="unknown run_id")
    chunks: list[str] = []
    finished = False
    try:
        chunk = await asyncio.wait_for(
            anext(it), timeout=SETTINGS.model_stream_timeout_s
        )
        chunks.append(chunk)
        if "<final_answer>" in chunk:
            finished = True
            _STREAMS.pop(run_id, None)
    except StopAsyncIteration:
        finished = True
        _STREAMS.pop(run_id, None)
    except TimeoutError:
        pass
    return {"chunks": chunks, "finished": finished}


@tools.post("/pytest_run")
async def pytest_run(
    body: dict[str, Any] = Body(default={}),  # noqa: B008
) -> dict[str, Any]:
    """Execute pytest inside the runtime container.

    Args:
        body: Optional JSON body specifying target path, markers, or quiet mode.

    Returns:
        A dictionary describing the exit code and captured output.
    """
    target = body.get("target")
    markers = body.get("markers")
    quiet = body.get("quiet", True)
    cmd = [sys.executable, "-m", "pytest"]
    if quiet:
        cmd.append("-q")
    if markers:
        cmd.extend(["-m", markers])
    if target:
        cmd.append(target)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": stdout.decode(),
        "stderr": stderr.decode(),
    }


@tools.post("/fs_read")
async def fs_read(
    body: dict[str, Any] = Body(...),  # noqa: B008
) -> dict[str, Any]:
    """Read a UTF-8 text file.

    Args:
        body: JSON body containing the ``path`` of the file to read.

    Returns:
        The file contents.
    """
    path = body["path"]
    try:
        content = await asyncio.to_thread(Path(path).read_text, encoding="utf-8")
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail="file not found") from err
    return {"content": content}


@tools.post("/fs_write")
async def fs_write(
    body: dict[str, Any] = Body(...),  # noqa: B008
) -> dict[str, Any]:
    """Write UTF-8 text to a file.

    Args:
        body: JSON body containing ``path`` and ``content`` fields.

    Returns:
        ``{"ok": True}`` when the write succeeds.
    """
    path = body["path"]
    content = body["content"]
    await asyncio.to_thread(Path(path).write_text, content, encoding="utf-8")
    return {"ok": True}


@tools.post("/git_apply_patch")
async def git_apply_patch(
    body: dict[str, Any] = Body(...),  # noqa: B008
) -> dict[str, Any]:
    """Apply a unified diff patch to the repository.

    Args:
        body: JSON body containing the ``patch`` string.

    Returns:
        A dictionary with ``ok`` and captured command output.
    """
    patch = body["patch"].encode()
    proc = await asyncio.create_subprocess_exec(
        "git",
        "apply",
        "--whitespace=nowarn",
        "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(patch)
    return {
        "ok": proc.returncode == 0,
        "stdout": stdout.decode(),
        "stderr": stderr.decode(),
    }


@tools.post("/execute")
async def execute_tool(
    request: Request,
    body: dict[str, Any] = Body(...),  # noqa: B008
) -> dict[str, Any]:
    """Execute a tool via the ability registry.

    Body should be an object with:
      - tool_id: string identifier
      - args: dict of arguments for the tool
    """
    tid = (body.get("tool_id") or body.get("name") or "").strip()
    if not tid:
        raise HTTPException(status_code=400, detail="missing tool_id")
    args = body.get("args") or {}
    if not isinstance(args, dict):
        raise HTTPException(status_code=400, detail="args must be an object")

    state = cast(Any, request.app.state)
    registry = state.ability_registry
    # Allow longer time for heavier tools (e.g., consensus)
    _timeout = SETTINGS.tool_timeout_s
    if tid == "deepconf_consensus":
        # Scale timeout with requested workload to avoid premature 504s.
        ns = 0
        mt = 0
        try:
            ns = int(args.get("num_samples", 1) or 1)
        except Exception:
            ns = 1
        try:
            mt = int(args.get("max_tokens", 256) or 256)
        except Exception:
            mt = 256
        # Base 60s + 60s per sample, cap additional by token budget
        scaled = 60.0 + 60.0 * max(1, ns)
        scaled += min(120.0, mt / 2.0)
        _timeout = max(_timeout, 90.0, scaled)
    try:
        result = await asyncio.wait_for(
            registry.execute(tid, args),
            timeout=_timeout,
        )
        return {"ok": True, "tool": tid, "result": result}
    except TimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail={"error": "tool_timeout", "tool": tid, "timeout": _timeout},
        ) from e
    except Exception as e:  # pragma: no cover - runtime errors
        raise HTTPException(status_code=500, detail=str(e)) from e


@tools.post("/execute/{tool_id}")
async def execute_tool_path(
    request: Request,
    tool_id: str,
    body: dict[str, Any] = Body(default={}),  # noqa: B008
) -> dict[str, Any]:
    args = body.get("args") if isinstance(body, dict) else None
    if args is None:
        args = body
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise HTTPException(status_code=400, detail="args must be an object")
    state = cast(Any, request.app.state)
    registry = state.ability_registry
    _timeout = SETTINGS.tool_timeout_s
    if tool_id == "deepconf_consensus":
        ns = 0
        mt = 0
        try:
            ns = int(args.get("num_samples", 1) or 1)
        except Exception:
            ns = 1
        try:
            mt = int(args.get("max_tokens", 256) or 256)
        except Exception:
            mt = 256
        scaled = 60.0 + 60.0 * max(1, ns)
        scaled += min(120.0, mt / 2.0)
        _timeout = max(_timeout, 90.0, scaled)
    try:
        result = await asyncio.wait_for(
            registry.execute(tool_id, args),
            timeout=_timeout,
        )
        return {"ok": True, "tool": tool_id, "result": result}
    except TimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail={"error": "tool_timeout", "tool": tool_id, "timeout": _timeout},
        ) from e
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e)) from e


# Compatibility alias: /ability/execute/{tool_id}
@ability.post("/execute/{tool_id}")
async def ability_execute(
    request: Request,
    tool_id: str,
    body: dict[str, Any] = Body(default={}),  # noqa: B008
) -> dict[str, Any]:
    return await execute_tool_path(request, tool_id, body)


# --------- MCP self-evolution helpers (brainstorm + dynamic registration) ---------


@tools.post("/mcp/brainstorm")
async def mcp_brainstorm(
    body: dict[str, Any] = Body(...),
) -> JSONResponse:  # noqa: B008
    """Propose lightweight MCP-like tool specs for a given task description.

    Input: {"task": str}
    Output: {"proposals": [spec, ...]}
    """
    task = (body.get("task") or "").lower()
    proposals: list[dict[str, Any]] = []

    # Heuristic proposals to keep predefinition minimal
    if any(k in task for k in ("url", "web", "http")):
        proposals.append(
            {
                "tool_id": "url_text_extractor",
                "description": "Fetch a URL and return UTF-8 text (best-effort).",
                "action": "fetch_url_text",
                "input_schema": {
                    "type": "object",
                    "required": ["url"],
                    "properties": {
                        "url": {"type": "string"},
                        "truncate": {"type": "integer"},
                    },
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "truncated": {"type": "boolean"},
                    },
                },
            }
        )
    if any(k in task for k in ("github", "readme", "repo")):
        proposals.append(
            {
                "tool_id": "github_file_fetch",
                "description": "Fetch a raw file from GitHub by owner/repo/path.",
                "action": "fetch_github_raw",
                "input_schema": {
                    "type": "object",
                    "required": ["owner", "repo", "path"],
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "path": {"type": "string"},
                        "ref": {"type": "string"},
                    },
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "url": {"type": "string"},
                        "truncated": {"type": "boolean"},
                        "error": {"type": "string"},
                    },
                },
            }
        )
    if not proposals:
        # Fallback generic executor
        proposals.append(
            {
                "tool_id": "echo_plan",
                "description": "Echo a plan for the task in a structured list.",
                "action": "echo_plan",
                "input_schema": {
                    "type": "object",
                    "required": ["task"],
                    "properties": {"task": {"type": "string"}},
                },
                "output_schema": {
                    "type": "object",
                    "properties": {"steps": {"type": "array"}},
                },
            }
        )

    return JSONResponse({"proposals": proposals})


@tools.post("/mcp/register")
async def mcp_register(
    request: Request, spec: dict[str, Any] = Body(...)
) -> JSONResponse:  # noqa: B008
    """Persist and register a dynamic tool spec with a minimal executor.

    Supported actions:
      - fetch_url_text: GET the URL and return text (urllib)
      - fetch_github_raw: reuse the existing dynamic executor via registry
      - echo_plan: split task into 3 bullet steps
    """
    action = spec.get("action") or ""
    box_dir = os.getenv("MCP_BOX_DIR", ".mcp_box") or ".mcp_box"

    # Check if a canonical tool already exists for this action and schema
    # First, run the abstractor to ensure index.json is up to date
    abstract_mcp_box(box_dir)

    # Load the index to check for canonical tools
    index_path = Path(box_dir) / "index.json"
    canonical_tool_id = None

    try:
        if index_path.exists():
            index_data = json.loads(index_path.read_text(encoding="utf-8"))

            # Get signature for this spec
            from .mcp_abstractor import _compute_signature, _normalize_spec

            norm_spec = _normalize_spec(spec)
            sig = _compute_signature(norm_spec)

            # Check if there's an existing canonical tool with this signature
            for tool in index_data.get("tools", []):
                if tool.get("signature") == sig:
                    canonical_tool_id = tool.get("tool_id")
                    print(
                        f"Found canonical tool {canonical_tool_id} for signature {sig}"
                    )
                    break
    except Exception as e:
        print(f"Warning: Error checking canonical tools: {e}")

    # Use canonical tool_id if found, otherwise persist as new spec
    if canonical_tool_id:
        tool_id = canonical_tool_id
        print(f"Using existing canonical tool_id: {tool_id}")
    else:
        tool_id = _catalog_service.register_dynamic_tool(spec)
        print(f"Persisted new tool spec: {tool_id}")

    state = cast(Any, request.app.state)
    registry = state.ability_registry

    async def _exec(args: dict[str, Any]) -> dict[str, Any]:
        if action == "fetch_url_text":
            import urllib.request

            url = args.get("url")
            if not isinstance(url, str) or not url:
                return {"error": "missing url"}
            truncate = int(args.get("truncate") or 4000)
            try:
                with urllib.request.urlopen(url, timeout=8) as resp:  # nosec B310
                    raw = resp.read()
                text = raw.decode("utf-8", errors="replace")
                truncated = False
                if len(text) > truncate:
                    text = text[:truncate]
                    truncated = True
                return {"content": text, "truncated": truncated}
            except Exception as e:  # pragma: no cover - network variability
                return {"content": "", "truncated": False, "error": str(e)}
        if action == "fetch_github_raw":
            # Delegate to the built-in dynamic executor registered in SimpleAbilityRegistry
            result = await registry.execute(
                "fetch_github_raw",
                {
                    "owner": args.get("owner"),
                    "repo": args.get("repo"),
                    "path": args.get("path"),
                    **({"ref": args.get("ref")} if args.get("ref") else {}),
                },
            )
            return cast(dict[str, Any], result)
        if action == "echo_plan":
            task = (args.get("task") or "").strip()
            steps = [f"Understand: {task}", "Identify resources", "Execute and verify"]
            return {"steps": steps}
        # Unknown action fallback
        return {"ok": True, "args": args}

    contract = {
        "tool_id": tool_id,
        "description": spec.get("description", "runtime-registered tool"),
        "input_schema": spec.get("input_schema", {"type": "object"}),
        "output_schema": spec.get("output_schema", {"type": "object"}),
    }

    registry.register_tool(contract=contract, executor=_exec)
    # Include both "registered" and "tool_id" for compatibility with callers
    return JSONResponse(
        {
            "ok": True,
            "registered": tool_id,
            "tool_id": tool_id,
            "action": action,
        }
    )


@tools.post("/mcp/abstract")
async def mcp_abstract(
    body: dict[str, Any] = Body(default={}),
) -> JSONResponse:  # noqa: B008
    """Normalize, deduplicate and index specs in MCP Box.

    Input (optional): {"mcp_box_dir": ".mcp_box"}
    Output: index summary (as written to index.json)
    """
    raw_dir = body.get("mcp_box_dir") if isinstance(body, dict) else None
    box_dir: str | Path
    if raw_dir is None:
        box_dir = os.getenv("MCP_BOX_DIR", ".mcp_box") or ".mcp_box"
    elif isinstance(raw_dir, (str, Path)):
        box_dir = raw_dir
    else:
        raise HTTPException(status_code=400, detail="mcp_box_dir must be a path")
    result = abstract_mcp_box(box_dir)
    return JSONResponse(result)


@tools.get("/mcp/catalog")
async def mcp_catalog() -> JSONResponse:
    """Get the tool catalog from the MCP Box.

    Returns the lightweight catalog.json for direct tool loading.
    If catalog.json doesn't exist, automatically abstracts and generates it.
    """
    box_dir = os.getenv("MCP_BOX_DIR", ".mcp_box") or ".mcp_box"
    catalog_path = Path(box_dir) / "catalog.json"

    if not catalog_path.exists():
        # Generate catalog by running abstractor
        abstract_mcp_box(box_dir)

    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        return JSONResponse(catalog)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to read catalog: {str(e)}"}
        )
