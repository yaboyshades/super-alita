"""FastAPI routes exposing tool-style endpoints for the REUG runtime."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import SETTINGS
from .router import execute_turn

TOOL_CATALOG = [
    {
        "name": "reug_start_turn",
        "description": "Start a single-turn REUG agent run with streaming. Returns a run_id and initial stream chunk(s).",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "session_id": {"type": "string"},
            },
            "required": ["message"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "stream_begun": {"type": "boolean"},
            },
            "required": ["run_id", "stream_begun"],
        },
    },
    {
        "name": "reug_stream_next",
        "description": "Fetch next streamed chunk(s) for a run. Ends when final answer emitted.",
        "input_schema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "chunks": {"type": "array", "items": {"type": "string"}},
                "finished": {"type": "boolean"},
            },
            "required": ["chunks", "finished"],
        },
    },
    {
        "name": "pytest_run",
        "description": "Run pytest with optional target path and markers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "markers": {"type": "string"},
                "quiet": {"type": "boolean", "default": True},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "exit_code": {"type": "integer"},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
            },
            "required": ["ok", "exit_code"],
        },
    },
    {
        "name": "fs_read",
        "description": "Read a UTF-8 text file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        "output_schema": {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        },
    },
    {
        "name": "fs_write",
        "description": "Write UTF-8 content to a file (creates/overwrites).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        "output_schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
    },
    {
        "name": "git_apply_patch",
        "description": "Apply a unified diff patch to the repo.",
        "input_schema": {
            "type": "object",
            "properties": {"patch": {"type": "string"}},
            "required": ["patch"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
            },
            "required": ["ok"],
        },
    },
]


tools = APIRouter(prefix="/tools", tags=["tools"])

_STREAMS: dict[str, Any] = {}


@tools.get("/catalog")
async def get_catalog() -> JSONResponse:
    """Return the static tool catalog."""
    return JSONResponse(TOOL_CATALOG)


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
    gen = execute_turn(
        message,
        session_id,
        event_bus=request.app.state.event_bus,
        registry=request.app.state.ability_registry,
        kg=request.app.state.kg,
        model=request.app.state.llm_model,
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
        chunk = await asyncio.wait_for(anext(it), timeout=SETTINGS.model_stream_timeout_s)
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
