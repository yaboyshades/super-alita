#!/usr/bin/env python3
"""FastAPI entrypoint for the REUG runtime."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.request
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from logging.config import dictConfig
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

# Add conditional imports for FastAPI dependencies
try:
    import uvicorn
    from fastapi import (
        APIRouter,
        Body,
        Depends,
        FastAPI,
        HTTPException,
        Query,
        Request,
        Response,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    FASTAPI_AVAILABLE = True
except ImportError:
    # Create stub classes when FastAPI is not available
    FastAPI = None  # type: ignore
    Request = None  # type: ignore
    APIRouter = None  # type: ignore
    Body = None  # type: ignore
    CORSMiddleware = None  # type: ignore
    JSONResponse = None  # type: ignore
    StreamingResponse = None  # type: ignore
    uvicorn = None  # type: ignore
    FASTAPI_AVAILABLE = False
    BaseModel = object  # type: ignore

# --- Resolve reug_runtime from local src if not installed ---
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
# Ensure the PROJECT ROOT (parent of 'src') is on sys.path so that
# package imports like 'src.core.events' resolve. Previously we only
# inserted the 'src' directory itself which makes top-level packages
# (core, agents, etc.) importable, but breaks fully-qualified
# 'src.*' imports used throughout the codebase.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# (Optionally also ensure direct 'src' path for simpler 'core.*' imports)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Event bus imports (moved after path setup to avoid import errors)
from reug_runtime.event_bus import (  # noqa: E402
    BaseEventBus,
    FileEventBus,
    make_event_bus,
)
from reug_runtime.llm_client import LLMClient, get_llm_client  # noqa: E402
from src.core.events import create_event  # noqa: E402
from src.gui.router import router as gui_router  # noqa: E402
from src.security.api_key_store import APIKeyStore  # noqa: E402
from src.telemetry.mcp_broadcaster import MCPTelemetryBroadcaster  # noqa: E402

try:
    from src.security.unified_security import RateLimiter  # type: ignore
except Exception:
    # Lightweight fallback to avoid optional deps at import time
    class RateLimiter:  # type: ignore
        def __init__(self, redis_client: None = None) -> None:
            self.local_cache: dict[str, list[float]] = {}

        async def is_allowed(self, identifier: str, limit: int, window: int):
            now = time.time()
            bucket = self.local_cache.setdefault(identifier, [])
            # Drop old entries
            cutoff = now - window
            bucket[:] = [t for t in bucket if t >= cutoff]
            allowed = len(bucket) < limit
            if allowed:
                bucket.append(now)
            info = {
                "remaining": max(0, limit - len(bucket)),
                "reset_in": max(
                    0, int(window - (now - bucket[0]) if bucket else window)
                ),
            }
            return allowed, info


# ---------------- Minimal API Key Auth (opt-in) ---------------- #
if FASTAPI_AVAILABLE:

    class _APISettings:
        """Read simple API key config from env.

        - ALITA_REQUIRE_API_KEY: 'true'/'false' (default 'false')
        - ALITA_API_KEY: single key value
        - ALITA_API_KEYS: comma-separated list of keys
        - ALITA_API_HEADER: header name to read (default 'Authorization')
        - ALITA_API_QUERY_PARAM: query param name (default 'api_key')
        """

        def __init__(self) -> None:
            self.require: bool = os.getenv(
                "ALITA_REQUIRE_API_KEY", "false"
            ).lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            keys: list[str] = []
            k1 = os.getenv("ALITA_API_KEY", "").strip()
            if k1:
                keys.append(k1)
            k_many = os.getenv("ALITA_API_KEYS", "")
            if k_many:
                keys.extend([x.strip() for x in k_many.split(",") if x.strip()])
            self.keys: set[str] = set(keys)
            self.header_name: str = os.getenv("ALITA_API_HEADER", "Authorization")
            self.query_param: str = os.getenv("ALITA_API_QUERY_PARAM", "api_key")

    _api_settings = _APISettings()
    _admin_key = os.getenv("ALITA_ADMIN_KEY", "").strip()

    def _get_api_store() -> APIKeyStore:
        store = getattr(globals().get("app", None), "state", object()).__dict__.get(
            "api_key_store", None
        )
        if store is None:
            store = APIKeyStore.from_env()
            with contextlib.suppress(Exception):
                app.state.api_key_store = store  # type: ignore[attr-defined]
        return store

    _rl_enabled = os.getenv("ALITA_RATE_LIMIT_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    _rl_default_limit = int(os.getenv("ALITA_RATE_LIMIT", "60") or 60)
    _rl_default_window = int(os.getenv("ALITA_RATE_WINDOW", "60") or 60)
    _redis_url = os.getenv("ALITA_REDIS_URL", "").strip()

    def _get_rate_limiter() -> RateLimiter:
        rl = getattr(globals().get("app", None), "state", object()).__dict__.get(
            "rate_limiter", None
        )
        if rl is None:
            if _redis_url:
                with contextlib.suppress(Exception):
                    from redis import asyncio as redis  # type: ignore

                    from src.security.rate_limit_redis import (
                        RedisRateLimiter,  # type: ignore
                    )

                    client = redis.from_url(_redis_url)
                    rl = RedisRateLimiter(client)  # type: ignore[assignment]
            if rl is None:
                rl = RateLimiter()
            with contextlib.suppress(Exception):
                app.state.rate_limiter = rl  # type: ignore[attr-defined]
        return rl

    async def require_api_key(request: Request) -> None:  # type: ignore
        """FastAPI dependency enforcing API key if enabled.

        Accepts one of:
        - Authorization: Bearer <key>
        - X-API-Key: <key> (if ALITA_API_HEADER is set to X-API-Key)
        - Query parameter ?api_key=<key> (configurable)
        """
        if not _api_settings.require:
            return None
        # Accept header
        key: str | None = None
        hdr_val = request.headers.get(_api_settings.header_name)
        if hdr_val:
            # Support Bearer and raw key
            if hdr_val.lower().startswith("bearer "):
                key = hdr_val[7:].strip()
            else:
                key = hdr_val.strip()
        # Fallback to query param
        if not key:
            key = request.query_params.get(_api_settings.query_param)  # type: ignore

        if not key:
            raise HTTPException(status_code=401, detail="Valid API key required")

        # Accept if env whitelist contains key
        if _api_settings.keys and key in _api_settings.keys:
            return None
        # Otherwise check persistent store
        with contextlib.suppress(Exception):
            store = _get_api_store()
            if store.verify(key):
                return None
        raise HTTPException(status_code=401, detail="Invalid API key")
        return None

    _open_reg = os.getenv("ALITA_AUTH_OPEN_REG", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    async def require_admin(request: Request) -> None:  # type: ignore
        hdr = request.headers.get(_api_settings.header_name, "").strip()
        candidate = hdr[7:].strip() if hdr.lower().startswith("bearer ") else hdr
        if _admin_key and candidate == _admin_key:
            return None
        if _open_reg:
            # Allow when open registration explicitly enabled
            return None
        raise HTTPException(status_code=403, detail="admin required")

    async def enforce_rate_limit(request: Request, response: Response) -> None:  # type: ignore
        if not _rl_enabled:
            return None
        rl = _get_rate_limiter()
        # Identify by API key when present, else by IP
        hdr_val = request.headers.get(_api_settings.header_name, "")
        ident: str | None = None
        if hdr_val:
            tok = (
                hdr_val[7:].strip()
                if hdr_val.lower().startswith("bearer ")
                else hdr_val.strip()
            )
            if tok:
                ident = f"key:{tok[:8]}"
        if not ident:
            ip = (
                (
                    request.headers.get("x-forwarded-for")
                    or (request.client.host if request.client else "unknown")
                )
                .split(",")[0]
                .strip()
            )
            ident = f"ip:{ip}"
        allowed, info = await rl.is_allowed(
            ident, _rl_default_limit, _rl_default_window
        )
        # Surface rate window to request for downstream emitters (SSE/JSON)
        try:
            request.state.rate_limit_info = {**info, "limit": _rl_default_limit}
        except Exception:
            pass
        # Add rate limit headers if response is available (non-SSE endpoints)
        try:
            if response is not None and hasattr(response, "headers"):
                response.headers["X-RateLimit-Limit"] = str(_rl_default_limit)
                response.headers["X-RateLimit-Remaining"] = str(
                    max(0, info.get("remaining", 0))
                )
                if "reset_in" in info:
                    response.headers["X-RateLimit-Reset-In"] = str(info["reset_in"])
        except Exception:
            pass
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail={"error": "rate_limited", **info},
                headers={
                    "Retry-After": str(
                        max(1, int(info.get("reset_in", _rl_default_window)))
                    ),
                    "X-RateLimit-Limit": str(_rl_default_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset-In": str(
                        info.get("reset_in", _rl_default_window)
                    ),
                },
            )
        return None


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - simple
        data = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(data, ensure_ascii=False)


def _configure_logging() -> Path:
    log_dir = Path(os.getenv("REUG_LOG_DIR", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "runtime.log"
    dictConfig(
        {
            "version": 1,
            "formatters": {"json": {"()": JsonFormatter}},
            "handlers": {
                "file": {
                    "class": "logging.FileHandler",
                    "filename": str(log_file),
                    "formatter": "json",
                    "encoding": "utf-8",
                },
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                },
            },
            "root": {
                "level": os.getenv("REUG_LOG_LEVEL", "INFO"),
                "handlers": ["file", "console"],
            },
        }
    )
    return log_file


def _hash_json(obj: Any) -> str:
    with contextlib.suppress(Exception):
        h = hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()
        return h[:16]
    return "na"


# Initialize routers - set defaults first
if FASTAPI_AVAILABLE:
    autogen_router = APIRouter(prefix="/autogen", tags=["autogen"])  # type: ignore
else:
    autogen_router = None  # type: ignore

# REUG runtime routers (streaming agent + toolbox)
try:
    from reug_runtime.config import SETTINGS
    from reug_runtime.router import router as agent_router
    from reug_runtime.router_tools import ability as ability_router
    from reug_runtime.router_tools import tools as tools_router
except Exception as e:  # pragma: no cover
    # Fallback: minimal routers to allow boot/health during development
    print("[WARN] reug_runtime import failed; falling back to minimal routers:", e)

    if not FASTAPI_AVAILABLE:
        # Create minimal stubs when FastAPI is not available
        agent_router = None  # type: ignore
        tools_router = None  # type: ignore
        SETTINGS = type("Settings", (), {"api_prefix": ""})()  # type: ignore
    else:
        agent_router = APIRouter(prefix="/v1", tags=["agent"])  # type: ignore

        @agent_router.post("/chat/stream")  # type: ignore
        async def chat_stream(request: Request) -> StreamingResponse:  # type: ignore
            try:
                body = await request.json()
                message = body.get("message", "")
                session_id = body.get("session_id", "default")

                async def gen() -> AsyncGenerator[str, None]:
                    # Get model identity
                    llm = getattr(
                        globals().get("app", None), "state", object()
                    ).__dict__.get("llm_model", None)
                    model_identity = {"model": "unknown", "provider": "unknown"}
                    if llm and hasattr(llm, "identify"):
                        try:
                            identity = await llm.identify()
                            model_identity.update(identity)
                        except Exception:
                            pass

                    # Send initial response with model identity
                    start_data = json.dumps(
                        {"type": "start", "content": "", "model": model_identity}
                    )
                    yield f"data: {start_data}\n\n"

                    # Process the message and generate response
                    response_content = await process_chat_message(message, session_id)

                    # Stream the response in chunks
                    for chunk in response_content.split():
                        content_data = json.dumps(
                            {"type": "content", "content": chunk + " "}
                        )
                        yield f"data: {content_data}\n\n"
                        # Small delay for natural typing effect
                        await asyncio.sleep(0.05)

                    # Send completion signal
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"

                return StreamingResponse(gen(), media_type="text/plain")  # type: ignore
            except Exception:

                async def error_gen() -> AsyncGenerator[str, None]:
                    error_msg = f"Sorry, I encountered an error: {str(ex)}"
                    error_data = json.dumps({"type": "content", "content": error_msg})
                    yield f"data: {error_data}\n\n"
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"

                return StreamingResponse(error_gen(), media_type="text/plain")  # type: ignore

        tools_router = APIRouter(prefix="/tools", tags=["tools"])  # type: ignore

        @tools_router.get("/health")  # type: ignore
        async def tools_health() -> dict[str, str]:
            return {"status": "ok"}

        # Create minimal settings
        SETTINGS = type("Settings", (), {"api_prefix": ""})()  # type: ignore

# Define autogen router endpoints if FastAPI is available
if FASTAPI_AVAILABLE and autogen_router is not None:

    @autogen_router.post("/trigger")  # type: ignore
    async def trigger_autogen(
        description: str = Body(..., embed=True),  # type: ignore
    ) -> dict[str, Any]:  # type: ignore
        """Manually trigger autogen capability creation."""
        try:
            from src.pipelines.autogen_pipeline import autogen_any

            result = await autogen_any(description=description)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @autogen_router.get("/capabilities")  # type: ignore
    async def list_capabilities() -> dict[str, Any]:  # type: ignore
        """List detected capability patterns."""
        from src.policies.need_detector import NeedDetector

        return {
            "status": "success",
            "capability_kinds": list(NeedDetector.KINDS.keys()),
            "patterns": {
                kind: [p.pattern for p in patterns]
                for kind, patterns in NeedDetector.KINDS.items()
            },
        }

    @autogen_router.post("/detect")  # type: ignore
    async def detect_needs(
        description: str = Body(..., embed=True),  # type: ignore
    ) -> dict[str, Any]:  # type: ignore
        """Detect capability needs from description."""
        from src.policies.need_detector import NeedDetector

        detector = NeedDetector()
        needs = detector.detect(description)
        return {
            "status": "success",
            "description": description,
            "detected_needs": needs,
        }


# --- Ability registry (minimal adapter; replace with your real one) ---
class SimpleAbilityRegistry:
    """
    Minimal, schema-friendly registry:
      - knows(): does this tool exist?
      - validate_args(): shallow "type-ish" validation
      - register(): dynamic tool creation (contract-first)
      - execute(): your dispatch to MCP / SDK / code
    """

    def __init__(self) -> None:
        # Seed with initial tools
        self._known: set[str] = {
            "echo",
            "brainstorm_mcp_stub",
            "fetch_github_raw",
            "secure_scan_code",
            "full_cycle_prototype",
        }
        self._contracts: dict[str, dict[str, Any]] = {
            "echo": {
                "tool_id": "echo",
                "description": "Echo back the provided payload",
                "input_schema": {
                    "type": "object",
                    "properties": {"payload": {"type": "string"}},
                },
                "output_schema": {"type": "object"},
            },
            "brainstorm_mcp_stub": {
                "tool_id": "brainstorm_mcp_stub",
                "description": (
                    "Lightweight brainstorming helper returning a structured "
                    "idea list for a task."
                ),
                "input_schema": {
                    "type": "object",
                    "required": ["task"],
                    "properties": {"task": {"type": "string"}},
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "ideas": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "fetch_github_raw": {
                "tool_id": "fetch_github_raw",
                "description": (
                    "Fetch a raw file from GitHub (best-effort; graceful "
                    "fallback if network disabled)."
                ),
                "input_schema": {
                    "type": "object",
                    "required": ["owner", "repo", "path"],
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "path": {"type": "string"},
                        "ref": {"type": "string"},
                        "truncate": {"type": "integer"},
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
            },
            "secure_scan_code": {
                "tool_id": "secure_scan_code",
                "description": (
                    "Naive static scan (Bandit-inspired heuristics) for "
                    "obvious risky patterns."
                ),
                "input_schema": {
                    "type": "object",
                    "required": ["code"],
                    "properties": {"code": {"type": "string"}},
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "issues": {"type": "array", "items": {"type": "object"}},
                        "issue_count": {"type": "integer"},
                    },
                },
            },
            "full_cycle_prototype": {
                "tool_id": "full_cycle_prototype",
                "description": (
                    "Synthetic build-run cycle: optional fetch then produce "
                    "run instructions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "path": {"type": "string"},
                    },
                },
                "output_schema": {"type": "object"},
            },
        }
        # Dynamic executors registered at runtime (tool_name -> async executor(args)->dict)
        self._executors: dict[str, Callable[[dict[str, Any]], Any]] = {}

    def register_tool(
        self, *, contract: dict[str, Any], executor: Callable[[dict[str, Any]], Any]
    ) -> None:
        """Register a dynamic tool with schema contract and async executor.

        The executor should accept a single dict of args and return a JSON-serializable
        result (may be awaitable). The contract must include a unique "tool_id".
        """
        tid = contract.get("tool_id")
        if not tid or not isinstance(tid, str):
            raise ValueError("contract.tool_id must be a non-empty string")
        self._contracts[tid] = contract
        self._known.add(tid)
        self._executors[tid] = executor

    def get_available_tools_schema(self) -> list[dict[str, Any]]:
        return list(self._contracts.values())

    def knows(self, tool_name: str) -> bool:
        return tool_name in self._known

    def validate_args(self, tool_name: str, args: dict[str, Any]) -> bool:
        # Simple: require "payload" string for echo; otherwise permissive
        if tool_name == "echo":
            return isinstance(args.get("payload"), str)
        # If dynamically registered, best-effort accept and rely on executor errors
        if tool_name in self._executors:
            return True
        return self.knows(tool_name)

    async def health_check(self, _: dict[str, Any]) -> bool:
        # In real setups, ping MCP, SDK, HTTP endpoint, etc.
        return True

    async def register(self, contract: dict[str, Any]) -> None:
        tid = contract["tool_id"]
        self._contracts[tid] = contract
        self._known.add(tid)

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        # Prefer dynamically registered executors
        exec_fn = self._executors.get(tool_name)
        if exec_fn is not None:
            res = exec_fn(args)
            if hasattr(res, "__await__"):
                return await res  # type: ignore[return-value]
            return res  # type: ignore[return-value]

        # Built-ins (fallbacks)
        # Implement your actual bindings here (MCP, HTTP APIs, Python functions).
        if tool_name == "echo":
            return {"echo": args.get("payload", "")}
        if tool_name == "brainstorm_mcp_stub":
            task = (args.get("task") or "").strip()
            base = task or "unspecified task"
            ideas = [
                f"Outline {base}",
                f"List core primitives for {base}",
                f"Write minimal happy-path example for {base}",
                f"Add edge case tests for {base}",
                f"Refactor {base} for clarity",
            ]
            return {"task": task, "ideas": ideas}
        if tool_name == "fetch_github_raw":
            owner = args.get("owner")
            repo = args.get("repo")
            path = args.get("path")
            ref = args.get("ref") or "main"
            truncate_at = int(args.get("truncate") or 4000)
            url_main = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
            content = ""
            err: str | None = None
            truncated = False
            for candidate_ref in [ref, "main", "master"]:
                try:
                    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{candidate_ref}/{path}"
                    with urllib.request.urlopen(url, timeout=5) as resp:  # nosec B310
                        raw_bytes = resp.read()
                    content = raw_bytes.decode("utf-8", errors="replace")
                    if len(content) > truncate_at:
                        content = content[:truncate_at]
                        truncated = True
                    url_main = url
                    break
                except Exception as e:  # pragma: no cover - network variability
                    err = str(e)
            return {
                "content": content,
                "url": url_main,
                "truncated": truncated,
                **({"error": err} if err and not content else {}),
            }
        if tool_name == "secure_scan_code":
            code = args.get("code") or ""
            patterns: list[tuple[str, str, str]] = [
                (r"\beval\(", "HIGH", "Use of eval() can be unsafe"),
                (r"\bexec\(", "HIGH", "Use of exec() can be unsafe"),
                (
                    r"subprocess\.Popen\(",
                    "MEDIUM",
                    "subprocess.Popen without sanitization",
                ),
                (r"os\.system\(", "MEDIUM", "os.system() call detected"),
                (r"pickle\.loads\(", "MEDIUM", "Untrusted pickle.loads()"),
            ]
            issues: list[dict[str, Any]] = []
            for pat, severity, message in patterns:
                for m in re.finditer(pat, code):
                    line = code.count("\n", 0, m.start()) + 1
                    issues.append(
                        {
                            "severity": severity,
                            "message": message,
                            "line": line,
                            "pattern": pat,
                        }
                    )
            return {"issues": issues, "issue_count": len(issues)}
        if tool_name == "full_cycle_prototype":
            # Combine brainstorming + optional fetch to produce a synthetic plan
            task = (args.get("task") or "hello world").strip()
            owner = args.get("owner")
            repo = args.get("repo")
            path = args.get("path")
            fetched: dict[str, Any] | None = None
            if owner and repo and path:
                fetched = await self.execute(
                    "fetch_github_raw",
                    {"owner": owner, "repo": repo, "path": path},
                )
            brainstorm = await self.execute("brainstorm_mcp_stub", {"task": task})
            run_instructions = [
                "# 1. Create virtual environment",
                "python -m venv .venv",
                "# 2. Activate venv",
                "# (Windows) .venv\\Scripts\\activate",
                "# 3. Write prototype code (hello.py)",
                'print("Hello World")',
                "# 4. Run it",
                "python hello.py",
            ]
            return {
                "task": task,
                "brainstorm": brainstorm,
                "fetched": fetched,
                "plan": run_instructions,
            }
        # Fallback generic - echo contract
        return {"ok": True, "tool": tool_name, "args": args}


# --- Knowledge graph (minimal; replace with your store/driver) ---
class SimpleKG:
    def __init__(self) -> None:
        self.atoms: list[dict[str, Any]] = []
        self.bonds: list[dict[str, Any]] = []

    async def retrieve_relevant_context(self, user_message: str) -> str:
        return f"(context for: {user_message[:48]}...)"

    async def get_goal_for_session(self, session_id: str) -> dict[str, Any]:
        return {
            "id": f"goal_{session_id}",
            "description": f"Assist session {session_id}",
        }

    async def create_atom(self, atom_type: str, content: Any) -> dict[str, Any]:
        atom: dict[str, Any] = {
            "id": f"atom_{len(self.atoms)}",
            "type": atom_type,
            "content": content,
        }
        self.atoms.append(atom)
        return atom

    async def create_bond(
        self, bond_type: str, source_atom_id: str, target_atom_id: str
    ) -> None:
        self.bonds.append(
            {"type": bond_type, "src": source_atom_id, "tgt": target_atom_id}
        )


# --- Chat Message Processing ---
async def process_chat_message(message: str, session_id: str) -> str:
    """Process a chat message and return a response.

    The logic intentionally remains simple/fast; we avoid model calls here so
    the UI always has an immediate, helpful reply while any deeper pipeline
    can run in parallel.
    """
    try:  # noqa: C901 (keep flat/simple even if slightly long)
        message_lower = message.lower().strip()

        greetings = [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
        ]
        if any(g in message_lower for g in greetings):
            return (
                "Hello! I'm Super Alita, your AI agent. I'm ready to help with "
                "code generation, Paper2Code (research paper implementation), "
                "web scraping, and more. What would you like to work on today?"
            )

        wellbeing_questions = [
            "how are you",
            "how do you do",
            "what's up",
        ]
        if any(q in message_lower for q in wellbeing_questions):
            return (
                "I'm running smoothly—event bus, ability registry, knowledge "
                "graph, and LLM components are all healthy. How can I assist "
                "you today?"
            )

        capability_queries = [
            "what can you do",
            "capabilities",
            "help",
            "what are your abilities",
        ]
        if any(q in message_lower for q in capability_queries):
            return (
                "I can help with: Paper2Code (implement papers into code); "
                "web scraping; tool/plugin/code generation; DeepCode-based "
                "analysis; multi-armed bandit optimization; and extension via "
                "custom plugins. Describe what you want and I'll build it."
            )

        if "paper" in message_lower and any(
            w in message_lower for w in ["implement", "code", "build"]
        ):
            return (
                "Sure—name the paper or architecture (e.g., 'ResNet', "
                "'Transformer encoder') and I'll generate an implementation."
            )

        if any(w in message_lower for w in ["scrape", "scraper", "website", "web"]):
            return (
                "I can scaffold a custom web scraper. Tell me the site and the "
                "data fields you need. I'll produce a robust, parse-ready "
                "scraper with error handling."
            )

        if message_lower:
            return (
                "I see you're asking about: '"
                + message
                + "'. Provide a bit more detail on the goal or artifact you "
                "need (tool, model, scraper, plugin), and I'll proceed."
            )
        return "How can I help you build or implement something today?"
    except Exception as exc:  # pragma: no cover - defensive path
        return (
            "I encountered an error processing your message: "
            f"{exc}. Please try again!"
        )


# ---------------- New Chat Router (SSE + JSON) ---------------- #
if FASTAPI_AVAILABLE:
    chat_router = APIRouter(prefix="/v1/chat", tags=["chat"])  # type: ignore
    # In-memory session store: session_id -> list[message dict]
    _CHAT_SESSIONS: dict[str, list[dict[str, str]]] = {}

    def _get_session_messages(session_id: str) -> list[dict[str, str]]:
        return _CHAT_SESSIONS.setdefault(session_id, [])

    @chat_router.get("/history")  # type: ignore
    async def get_history(
        session: str | None = Query(None),
        _auth: None = Depends(require_api_key),  # type: ignore
        _rl: None = Depends(enforce_rate_limit),  # type: ignore
    ) -> JSONResponse:  # type: ignore
        sid = session or "default"
        return JSONResponse({"session": sid, "messages": _get_session_messages(sid)})  # type: ignore

    @chat_router.delete("/history")  # type: ignore
    async def clear_history(
        session: str | None = Query(None),
        _auth: None = Depends(require_api_key),  # type: ignore
        _rl: None = Depends(enforce_rate_limit),  # type: ignore
    ) -> JSONResponse:  # type: ignore
        sid = session or "default"
        _CHAT_SESSIONS[sid] = []
        return JSONResponse({"session": sid, "cleared": True})  # type: ignore

    async def generate_reply_chunks(prompt: str, session_id: str, llm: Any):  # type: ignore
        """Stream tokens from the configured llm_model.

        Falls back to simple echo tokens if model lacks stream_chat.
        Maintains conversation context per session.
        """
        history = _get_session_messages(session_id)
        # Build messages list (system + history + new user)
        messages: list[dict[str, str]] = (
            [
                {
                    "role": "system",
                    "content": "You are Super Alita. Be concise and helpful.",
                }
            ]
            + history
            + [{"role": "user", "content": prompt}]
        )

        # Build tool schemas for LLMs that support tool calls (e.g., OpenAI)
        def _openai_tools_from_registry(reg: Any) -> list[dict[str, Any]]:
            tools: list[dict[str, Any]] = []
            try:
                contracts = reg.get_available_tools_schema() if reg else []
                for c in contracts:
                    name = c.get("tool_id")
                    if not name:
                        continue
                    desc = c.get("description") or ""
                    params = (
                        c.get("input_schema")
                        or c.get("parameters")
                        or {
                            "type": "object",
                            "additionalProperties": True,
                        }
                    )
                    tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": name,
                                "description": desc,
                                "parameters": params,
                            },
                        }
                    )
            except Exception:
                return []
            return tools

        ability_reg = getattr(
            globals().get("app", None), "state", object()
        ).__dict__.get("ability_registry", None)
        llm_tools = _openai_tools_from_registry(ability_reg)

        # Try direct Ollama connection first for gpt-oss:20b model
        try:
            ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
            llm_model = os.getenv("LLM_MODEL", "")

            # Check if we're configured for Ollama
            if llm_model.startswith("ollama:") or "gpt-oss" in llm_model:
                ollama_model = llm_model.replace("ollama:", "")
                if not ollama_model:
                    ollama_model = "gpt-oss:20b"  # default fallback

                print(f"🔄 Using Ollama: {ollama_model} at {ollama_host}")

                async with httpx.AsyncClient(timeout=60) as client:
                    # Use Ollama chat API for better conversation handling
                    payload = {
                        "model": ollama_model,
                        "messages": messages,
                        "stream": True,
                    }

                    async with client.stream(
                        "POST", f"{ollama_host}/api/chat", json=payload
                    ) as response:
                        if response.status_code == 200:
                            print(f"✅ Ollama connected ({response.status_code})")
                            content_received = False
                            async for line in response.aiter_lines():
                                if not line.strip():
                                    continue
                                try:
                                    data = json.loads(line)
                                    msg_content = data.get("message", {}).get("content")
                                    if msg_content:
                                        content_received = True
                                        yield msg_content
                                    if data.get("done"):
                                        if content_received:
                                            print("✅ Ollama stream completed")
                                        return
                                except json.JSONDecodeError:
                                    continue
                        else:
                            print(f"❌ Ollama HTTP error: {response.status_code}")

        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            # Continue to other methods...

        if hasattr(llm, "stream_chat"):
            try:
                # First pass: allow LLM to decide on tools (if supported)
                try:
                    stream_iter = llm.stream_chat(messages, tools=llm_tools)  # type: ignore[attr-defined]
                except TypeError:
                    stream_iter = llm.stream_chat(messages)  # type: ignore[attr-defined]
                async for chunk in stream_iter:
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get("type") == "content":
                        token = chunk.get("content", "")
                        if token:
                            # Yield typed content event for SSE routing
                            yield {"type": "content", "content": token}
                    # Handle tool calls: execute and then ask for a follow-up
                    if chunk.get("type") == "tool_calls":
                        tool_calls = chunk.get("tool_calls") or []
                        # Execute tools and append results as messages
                        for call in tool_calls:
                            try:
                                fn = call.get("function", {})
                                tname = fn.get("name")
                                arg_str = fn.get("arguments") or "{}"
                                args: dict[str, Any]
                                try:
                                    args = json.loads(arg_str)
                                except Exception:
                                    args = {}
                                # Emit a tool start hint for UI
                                if tname:
                                    yield {
                                        "type": "tool_start",
                                        "tool": tname,
                                        "args": args,
                                    }
                                if ability_reg and tname:
                                    result = await ability_reg.execute(tname, args)  # type: ignore
                                else:
                                    result = {"error": "no_ability_registry"}
                                # Emit tool result for UI
                                if tname:
                                    yield {
                                        "type": "tool_result",
                                        "tool": tname,
                                        "result": result,
                                    }
                                # Append tool result for a follow-up completion
                                messages.append(
                                    {
                                        "role": "tool",
                                        "content": json.dumps(
                                            result, ensure_ascii=False
                                        ),
                                        "tool_call_id": call.get("id"),
                                    }
                                )
                            except Exception:
                                continue
                        # Second pass: ask the LLM to produce the final answer using tool results
                        try:
                            stream_iter2 = llm.stream_chat(messages, tools=llm_tools)  # type: ignore[attr-defined]
                        except TypeError:
                            stream_iter2 = llm.stream_chat(messages)  # type: ignore[attr-defined]
                        async for chunk2 in stream_iter2:
                            if not isinstance(chunk2, dict):
                                continue
                            if chunk2.get("type") == "content":
                                token2 = chunk2.get("content", "")
                                if token2:
                                    yield {"type": "content", "content": token2}
                        return
                return
            except Exception:
                # Fall back to naive tokenization on error
                pass

        # Fallback: conversational responses when LLM is unavailable
        greeting_patterns = ["hi", "hello", "hey", "greetings"]
        help_patterns = ["help", "what can you do", "capabilities", "abilities"]

        prompt_lower = prompt.lower().strip()

        # Check for greetings
        if any(pattern in prompt_lower for pattern in greeting_patterns):
            response = (
                "Hello! I'm Super Alita, your AI assistant. I can help you with "
                "code generation, research paper implementation, web scraping, "
                "and much more. What would you like to work on today?"
            )
        # Check for standalone help requests
        elif prompt_lower in help_patterns or prompt_lower == "what can you do?":
            response = (
                "I can assist you with:\n"
                "• Code generation and refactoring\n"
                "• Research paper implementation (Paper2Code)\n"
                "• Web scraping and data extraction\n"
                "• Plugin development\n"
                "• General programming questions\n\n"
                "Just tell me what you'd like to work on!"
            )
        elif len(prompt_lower) < 5:
            response = (
                "I'd be happy to help! Could you please tell me more about "
                "what you need assistance with?"
            )
        else:
            response = (
                f"I understand you're asking about: {prompt}. I'm currently "
                "running in fallback mode. For the best experience, please "
                "ensure the LLM model is properly configured. However, I can "
                "still try to help - could you provide more details about "
                "what you need?"
            )

        # Stream the response token by token
        words = response.split()
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.03)
            yield {"type": "content", "content": token}

    def _sse_pack(event_type: str, payload: dict, ev_id: str | None = None) -> str:
        parts: list[str] = []
        if event_type:
            parts.append(f"event: {event_type}")
        if ev_id:
            parts.append(f"id: {ev_id}")
        parts.append(f"data: {json.dumps(payload, ensure_ascii=False)}")
        return "\n".join(parts) + "\n\n"

    @chat_router.get("/stream")  # type: ignore
    async def chat_stream_endpoint(
        req: Request,  # type: ignore
        q: str = Query(...),
        session: str | None = Query(None),
        _auth: None = Depends(require_api_key),
        _rl: None = Depends(enforce_rate_limit),
    ):  # type: ignore
        async def event_source():  # type: ignore
            ev_id = str(uuid4())
            last_heartbeat = time.time()
            sid = session or "default"
            llm = (
                getattr(chat_stream_endpoint, "_llm", None)
                or getattr(chat_router, "app_llm", None)
                or getattr(globals().get("app", None), "state", object()).__dict__.get(
                    "llm_model", None
                )
            )
            accumulated: list[str] = []

            # Get model identity early
            model_identity = {"model": "unknown", "provider": "unknown"}
            if llm and hasattr(llm, "identify"):
                try:
                    identity = await llm.identify()
                    model_identity.update(identity)
                except Exception:
                    pass

            start_payload = {"id": ev_id, "session": sid, "model": model_identity}
            rl_info = getattr(req.state, "rate_limit_info", None)
            if isinstance(rl_info, dict):
                start_payload["rate_limit"] = rl_info
            yield _sse_pack("start", start_payload, ev_id)
            # Track user turn in history (streaming mode)
            try:
                _get_session_messages(sid).append({"role": "user", "content": q})
            except Exception:
                pass
            async for chunk in generate_reply_chunks(q, sid, llm):
                # Support both legacy string tokens and typed dict events
                if isinstance(chunk, str):
                    accumulated.append(chunk)
                    yield _sse_pack("content", {"content": chunk}, ev_id)
                elif isinstance(chunk, dict):
                    et = chunk.get("type") or "content"
                    # Normalize content payload
                    if et == "content":
                        tok = chunk.get("content", "")
                        if isinstance(tok, str) and tok:
                            accumulated.append(tok)
                        payload = {"content": tok}
                    else:
                        payload = {k: v for k, v in chunk.items() if k != "type"}
                    yield _sse_pack(str(et), payload, ev_id)
                else:
                    # Fallback: stringify unknown chunks
                    yield _sse_pack("content", {"content": str(chunk)}, ev_id)
                now = time.time()
                if now - last_heartbeat > 15:
                    # Heartbeat comment frame to keep connection alive behind proxies
                    yield ": heartbeat\n\n"
                    last_heartbeat = now
            # Append assistant message to session history
            history = _get_session_messages(sid)
            try:
                full = "".join(accumulated)
                history.append(
                    {"role": "assistant", "content": full or "(response streamed)"}
                )
            except Exception:
                history.append({"role": "assistant", "content": "(response streamed)"})
            yield _sse_pack("done", {"reason": "complete"}, ev_id)

        return StreamingResponse(  # type: ignore
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @chat_router.post("")  # type: ignore
    async def chat_fallback(req: Request, _auth: None = Depends(require_api_key), _rl: None = Depends(enforce_rate_limit)):  # type: ignore
        body = await req.json()
        prompt = (body.get("q") or body.get("message") or "").strip()
        session_id = body.get("session") or "default"
        if not prompt:
            return JSONResponse({"error": "missing 'q' or 'message'"}, status_code=400)  # type: ignore
        llm = getattr(app, "state", object()).__dict__.get("llm_model", None)

        # Get model identity
        model_identity = {"model": "unknown", "provider": "unknown"}
        if llm and hasattr(llm, "identify"):
            try:
                identity = await llm.identify()
                model_identity.update(identity)
            except Exception:
                pass

        # Track user turn
        _get_session_messages(session_id).append({"role": "user", "content": prompt})
        full = "".join(
            [chunk async for chunk in generate_reply_chunks(prompt, session_id, llm)]
        )
        # Store assistant reply
        _get_session_messages(session_id).append({"role": "assistant", "content": full})
        return JSONResponse(
            {
                "type": "message",
                "content": full,
                "session": session_id,
                "model": model_identity,
            }
        )  # type: ignore

    # --------------- Unified API Gateway (minimal) --------------- #
    api_router = APIRouter(prefix="/api/v1", tags=["api"])  # type: ignore

    class QueryRequest(BaseModel):  # type: ignore[misc,valid-type]
        prompt: str
        mode: str = "hybrid"  # accepted but not enforced in this minimal gateway
        session: str | None = None
        stream: bool = False
        max_tokens: int | None = None

    @api_router.get("/health")  # type: ignore
    async def api_health() -> dict[str, str]:  # type: ignore
        return {"status": "ok"}

    @api_router.post("/query")  # type: ignore
    async def api_query(
        req: QueryRequest,  # type: ignore
        request: Request,  # type: ignore
        response: Response,  # type: ignore
        _auth: None = Depends(require_api_key),  # type: ignore
        _rl: None = Depends(enforce_rate_limit),  # type: ignore
    ) -> Any:  # type: ignore
        """Unified non-breaking query endpoint.

        Streams via SSE when stream=True, or returns a JSON payload otherwise.
        Delegates to the existing chat generation logic to avoid duplication.
        """
        sid = req.session or "default"
        llm = getattr(app, "state", object()).__dict__.get("llm_model", None)

        # Identify model used
        model_identity = {"model": "unknown", "provider": "unknown"}
        if llm and hasattr(llm, "identify"):
            try:
                identity = await llm.identify()
                model_identity.update(identity)
            except Exception:
                pass

        if req.stream:

            async def event_source() -> AsyncGenerator[str, None]:  # type: ignore
                ev_id = str(uuid4())
                start_payload = {"id": ev_id, "session": sid, "model": model_identity}
                rl_info = getattr(request.state, "rate_limit_info", None)
                if isinstance(rl_info, dict):
                    start_payload["rate_limit"] = rl_info
                yield _sse_pack("start", start_payload, ev_id)
                async for chunk in generate_reply_chunks(req.prompt, sid, llm):
                    yield _sse_pack("content", {"content": chunk}, ev_id)
                yield _sse_pack("done", {"reason": "complete"}, ev_id)

            return StreamingResponse(  # type: ignore[return-value]
                event_source(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming path: join chunks
        full = "".join(
            [chunk async for chunk in generate_reply_chunks(req.prompt, sid, llm)]
        )
        _get_session_messages(sid).append({"role": "user", "content": req.prompt})
        _get_session_messages(sid).append({"role": "assistant", "content": full})
        payload = {
            "answer": full,
            "session": sid,
            "mode": req.mode,
            "model": model_identity,
        }
        rl_info = getattr(request.state, "rate_limit_info", None)
        if isinstance(rl_info, dict):
            payload["rate_limit"] = rl_info
        return payload

    # ---------------------- Patch 0003: Developer Experience & Team Integration ---------------------- #
    
    # Import team orchestrator at module level (will add this after creating the routes)
    team_orchestrator: Any = None  # Global state for team orchestrator
    
    class DeveloperActionRequest(BaseModel):  # type: ignore[misc,valid-type]
        user_id: str
        action: str
        context: dict[str, Any]
    
    @api_router.post("/developer-action")  # type: ignore
    async def handle_developer_action(
        req: DeveloperActionRequest,  # type: ignore
        _auth: None = Depends(require_api_key),  # type: ignore
        _rl: None = Depends(enforce_rate_limit),  # type: ignore
    ) -> Any:  # type: ignore
        """Handles developer actions from VS Code bridge."""
        try:
            # Get the orchestrator from app state if available
            orchestrator = getattr(app.state, 'ecosystem_orchestrator', None)
            if not orchestrator:
                # Create a basic orchestrator for demonstration
                from src.ecosystem.master_orchestrator import EcosystemOrchestrator
                orchestrator = EcosystemOrchestrator()
            
            # Process the developer action through the orchestrator
            result = await orchestrator.handle_developer_action(
                req.user_id, req.action, req.context
            )
            
            # Feed data to team orchestrator if available
            global team_orchestrator
            if team_orchestrator and req.action == "todo_detected":
                team_orchestrator.consume_event(
                    "workflow.todo_resolution.completed",
                    {"context": req.context}
                )
            
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @api_router.get("/team/health")  # type: ignore
    async def get_team_health() -> dict[str, Any]:  # type: ignore
        """Returns a health summary and optimization suggestions for the team."""
        global team_orchestrator
        if not team_orchestrator:
            from src.ecosystem.team_orchestrator import TeamProductivityOrchestrator
            team_orchestrator = TeamProductivityOrchestrator()
            
            # Manually feed some sample data for the demo
            team_orchestrator.consume_event(
                "workflow.todo_resolution.completed", 
                {"context": {"todo_text": "Refactor the authentication logic"}}
            )
            team_orchestrator.consume_event(
                "workflow.todo_resolution.completed", 
                {"context": {"todo_text": "Add new authentication endpoint"}}
            )
            
        return team_orchestrator.generate_team_health_summary()

    # ---------------------- Auth management ---------------------- #
    auth_router = APIRouter(prefix="/api/v1/auth", tags=["auth"])  # type: ignore

    class KeyCreateRequest(BaseModel):  # type: ignore[misc,valid-type]
        owner: str
        metadata: dict[str, Any] | None = None
        ttl_hours: int | None = None

    @auth_router.post("/keys")  # type: ignore
    async def create_key(
        req: KeyCreateRequest,  # type: ignore
        _admin: None = Depends(require_admin),  # type: ignore
    ) -> Any:  # type: ignore
        store = _get_api_store()
        result = store.add(req.owner, req.metadata or {}, req.ttl_hours)
        return {"status": "created", **result}

    @auth_router.post("/keys/rotate")  # type: ignore
    async def rotate_key(request: Request) -> Any:  # type: ignore
        # Requires presenting a valid existing key in Authorization header
        hdr = request.headers.get(_api_settings.header_name, "")
        key = hdr[7:].strip() if hdr.lower().startswith("bearer ") else hdr.strip()
        if not key:
            raise HTTPException(status_code=401, detail="present API key to rotate")
        store = _get_api_store()
        if not store.verify(key):
            raise HTTPException(status_code=401, detail="invalid API key")
        rotated = store.rotate(key)
        if not rotated:
            raise HTTPException(status_code=400, detail="rotation failed")
        return {"status": "rotated", **rotated}

    class KeyRevokeRequest(BaseModel):  # type: ignore[misc,valid-type]
        key: str | None = None
        key_id: str | None = None

    @auth_router.post("/keys/revoke")  # type: ignore
    async def revoke_key(
        body: KeyRevokeRequest,  # type: ignore
        _admin: None = Depends(require_admin),  # type: ignore
    ) -> Any:  # type: ignore
        store = _get_api_store()
        if body.key:
            ok = store.revoke(body.key)
        elif body.key_id:
            ok = store.revoke_by_id(body.key_id)
        else:
            raise HTTPException(status_code=400, detail="key or key_id required")
        return {"status": "revoked" if ok else "noop"}

    @auth_router.get("/keys/me")  # type: ignore
    async def whoami(request: Request) -> Any:  # type: ignore
        hdr = request.headers.get(_api_settings.header_name, "")
        key = hdr[7:].strip() if hdr.lower().startswith("bearer ") else hdr.strip()
        if not key:
            raise HTTPException(status_code=401, detail="present API key")
        store = _get_api_store()
        rec = store.get_by_raw(key)
        if not rec or rec.revoked_at:
            raise HTTPException(status_code=401, detail="invalid API key")
        pub = rec.to_public()
        pub["key_id"] = rec.key_id
        return pub

    @auth_router.get("/keys")  # type: ignore
    async def list_keys(_admin: None = Depends(require_admin)) -> Any:  # type: ignore
        store = _get_api_store()
        return {"keys": store.list_public()}


# --- FastAPI factory ---
def create_app(*, event_bus: BaseEventBus | None = None) -> Any:
    """Create FastAPI app or return None if FastAPI not available."""
    if not FASTAPI_AVAILABLE:
        print(
            "[ERROR] FastAPI not available - install with: "
            "'pip install fastapi uvicorn'"
        )
        return None

    _configure_logging()
    app = FastAPI(title="REUG Runtime", version="0.2.0")  # type: ignore

    # CORS (tweak as needed)
    app.add_middleware(  # type: ignore
        CORSMiddleware,  # type: ignore
        allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    # Health for Dockerfile/compose
    from reug_runtime.health import check_health

    @app.get("/healthz")  # type: ignore
    async def health_check() -> JSONResponse:  # type: ignore
        status = await check_health(
            app.state.event_bus,  # type: ignore
            app.state.ability_registry,  # type: ignore
            app.state.kg,  # type: ignore
            app.state.llm_model,  # type: ignore
        )
        code = 200 if status["status"] == "healthy" else 503
        return JSONResponse(status_code=code, content=status)  # type: ignore

    # Alternative health endpoint
    @app.get("/health")  # type: ignore
    async def health_check_alt() -> JSONResponse:  # type: ignore
        status = await check_health(
            app.state.event_bus,  # type: ignore
            app.state.ability_registry,  # type: ignore
            app.state.kg,  # type: ignore
            app.state.llm_model,  # type: ignore
        )
        code = 200 if status["status"] == "healthy" else 503
        if (
            isinstance(app.state.event_bus, FileEventBus)  # type: ignore
            and isinstance(app.state.ability_registry, SimpleAbilityRegistry)  # type: ignore
            and isinstance(app.state.kg, SimpleKG)  # type: ignore
            and isinstance(app.state.llm_model, LLMClient)  # type: ignore
        ):
            minimal: dict[str, Any] = {
                "status": status["status"],
                "service": "super-alita",
            }
            return JSONResponse(status_code=code, content=minimal)  # type: ignore
        return JSONResponse(status_code=code, content=status)  # type: ignore

    # Inject dependencies for the REUG router
    app.state.event_bus = event_bus if event_bus is not None else make_event_bus()  # type: ignore
    app.state.ability_registry = SimpleAbilityRegistry()  # type: ignore
    app.state.kg = SimpleKG()  # type: ignore
    app.state.llm_model = get_llm_client(os.getenv("LLM_MODEL"))  # type: ignore

    # Lifespan handler (replaces deprecated on_event startup/shutdown)
    app.state.plugins = []  # type: ignore
    app.state.mcp_broadcaster = None  # type: ignore
    app.state._orig_emit = None  # type: ignore

    @asynccontextmanager
    async def _lifespan(_: Any) -> AsyncGenerator[None, None]:  # type: ignore
        # Optionally enable MCP telemetry broadcasting and wrap event bus emit
        broadcaster: MCPTelemetryBroadcaster | None = None
        mcp_enabled = os.getenv("MCP_BROADCAST_ENABLED", "").strip().lower()
        if mcp_enabled in {"1", "true", "yes"}:
            try:
                broadcaster = MCPTelemetryBroadcaster()
                await broadcaster.start()
                app.state.mcp_broadcaster = broadcaster  # type: ignore

                # Wrap event_bus.emit to fan-out to MCP after successful emit
                orig_emit = app.state.event_bus.emit  # type: ignore[attr-defined]
                app.state._orig_emit = orig_emit  # type: ignore

                async def _emit_and_broadcast(event: dict[str, Any]) -> dict[str, Any]:
                    result = await orig_emit(event)
                    try:
                        # Normalize event shape
                        etype = (
                            event.get("event_type")
                            or event.get("type")
                            or event.get("kind")
                            or "event"
                        )
                        source = (
                            event.get("source")
                            or event.get("source_plugin")
                            or event.get("plugin")
                            or "runtime"
                        )
                        session_id = event.get("session_id")
                        conversation_id = event.get("conversation_id")
                        meta = event.get("metadata") or {}
                        # Avoid duplicating top-level fields in data
                        data = {
                            k: v
                            for k, v in event.items()
                            if k
                            not in {
                                "event_type",
                                "type",
                                "kind",
                                "source",
                                "source_plugin",
                                "plugin",
                                "session_id",
                                "conversation_id",
                                "metadata",
                            }
                        }
                        await broadcaster.broadcast_event(
                            event_type=str(etype),
                            source=str(source),
                            data=data,  # type: ignore[arg-type]
                            session_id=str(session_id) if session_id else None,
                            conversation_id=(
                                str(conversation_id) if conversation_id else None
                            ),
                            metadata=meta if isinstance(meta, dict) else None,
                        )
                    except Exception:
                        # Never block event path due to telemetry issues
                        pass
                    return result

                app.state.event_bus.emit = _emit_and_broadcast  # type: ignore[attr-defined]
            except Exception:
                # If broadcaster setup fails, continue without telemetry
                app.state.mcp_broadcaster = None  # type: ignore
                app.state._orig_emit = None  # type: ignore
        # Startup: initialize enhanced plugin system with YAML configuration
        with contextlib.suppress(Exception):
            from src.core.enhanced_plugin_system import (
                initialize_enhanced_plugin_system,
            )

            # Initialize enhanced plugin system
            loaded_plugins = await initialize_enhanced_plugin_system(app)

            # Log plugin initialization results
            plugin_count = len(loaded_plugins)
            plugin_names = list(loaded_plugins.keys())
            print(f"🔌 Enhanced plugin system: {plugin_count} plugins loaded")
            if plugin_names:
                print(f"   Loaded: {', '.join(plugin_names[:5])}")
                if len(plugin_names) > 5:
                    print(f"   ... and {len(plugin_names) - 5} more")

        # Register Enhanced / external Consensus sampling tool
        try:
            print("🔧 DEBUG: Starting consensus tool registration (adapter-aware)...")
            ability_reg = app.state.ability_registry  # type: ignore

            # Decide whether to use adapter (env-driven)
            use_adapter = os.getenv("CONSENSUS_SERVICE_MODE") is not None
            if use_adapter:
                from src.adapters.consensus_client import ConsensusClient

                consensus_provider = ConsensusClient(
                    {
                        "base_url": "http://localhost:11434/v1",
                        "model_name": "gpt-oss:20b",
                        "timeout": 60.0,
                        "grpc_url": os.getenv("CONSENSUS_GRPC_URL", "localhost:50051"),
                    }
                )
                await consensus_provider.initialize()  # local provider init
                print(
                    "🔧 DEBUG: Consensus adapter initialized (mode="
                    f"{os.getenv('CONSENSUS_SERVICE_MODE','local')})"
                )
            else:
                from src.abilities.enhanced_consensus_ability import (
                    EnhancedConsensusProvider,
                )

                consensus_provider = EnhancedConsensusProvider(
                    {
                        "base_url": "http://localhost:11434/v1",
                        "model_name": "gpt-oss:20b",
                        "timeout": 60.0,
                    }
                )
                await consensus_provider.initialize()
                print("🔧 DEBUG: Local enhanced consensus provider initialized")

            consensus_contract = {
                "tool_id": "deepconf_consensus",
                "description": "Enhanced consensus sampling with multiple aggregation methods",
                "input_schema": {
                    "type": "object",
                    "required": ["prompt"],
                    "properties": {
                        "prompt": {"type": "string"},
                        "num_samples": {"type": "integer", "default": 3},
                        "temperature": {"type": "number", "default": 0.7},
                        "max_tokens": {"type": "integer", "default": 512},
                        "method": {
                            "type": "string",
                            "default": "weighted_vote",
                            "enum": [
                                "simple_vote",
                                "weighted_vote",
                                "confidence_based",
                                "semantic_similarity",
                                "ensemble_ranking",
                            ],
                        },
                        "confidence_threshold": {"type": "number", "default": 0.7},
                        "temperature_range": {"type": "number", "default": 0.2},
                    },
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "consensus_text": {"type": "string"},
                        "consensus_confidence": {"type": "number"},
                        "aggregation_method": {"type": "string"},
                        "individual_responses": {"type": "array"},
                        "confidence_scores": {"type": "array"},
                        "metadata": {"type": "object"},
                    },
                },
            }

            async def consensus_executor(args: dict[str, Any]) -> dict[str, Any]:
                return await consensus_provider.consensus_sampling(
                    prompt=args["prompt"],
                    num_samples=args.get("num_samples", 3),
                    temperature=args.get("temperature", 0.7),
                    max_tokens=args.get("max_tokens", 512),
                    method=args.get("method", "weighted_vote"),
                    confidence_threshold=args.get("confidence_threshold", 0.7),
                    temperature_range=args.get("temperature_range", 0.2),
                )

            ability_reg.register_tool(
                contract=consensus_contract, executor=consensus_executor
            )
            print("✅ DEBUG: Consensus tool registered (adapter-aware)")
        except Exception as e:  # noqa: BLE001
            print(f"❌ DEBUG: Failed to register consensus tool: {e}")
            import traceback

            traceback.print_exc()

        # Register Mangle integration
        try:
            from src.abilities.mangle.register import (
                register_mangle_abilities,
                register_mangle_plugin,
            )

            print("🔧 DEBUG: Starting Mangle integration registration...")

            # Register Mangle abilities
            mangle_config = {
                "mangle": {
                    "binary_path": os.getenv("MANGLE_BIN_PATH", "mangle"),
                    "timeout": 30,
                    "knowledge_base_dir": "./data/mangle",
                }
            }

            register_mangle_abilities(app.state.ability_registry, mangle_config)  # type: ignore
            register_mangle_plugin(None, mangle_config)  # plugin_registry not needed
            print("✅ DEBUG: Mangle integration registered successfully!")

        except Exception as e:
            print(f"❌ DEBUG: Failed to register Mangle integration: {e}")
            import traceback

            traceback.print_exc()

        # Optional ability auto-discovery (best-effort)
        try:
            if os.getenv("ALITA_AUTO_DISCOVER_ABILITIES", "false").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                from src.abilities.registry_auto import auto_register_abilities

                await auto_register_abilities(app.state.ability_registry)  # type: ignore
        except Exception:
            pass

        # Emit startup events
        try:
            corr = str(uuid4())
            logging.getLogger().info("runtime startup")
            await app.state.event_bus.emit(  # type: ignore
                {
                    "type": "STATE_TRANSITION",
                    "from": "BOOT",
                    "to": "READY",
                    "correlation_id": corr,
                }
            )
            await app.state.event_bus.emit(  # type: ignore
                {
                    "type": "TaskStarted",
                    "correlation_id": corr,
                    "goal": "startup",
                    "user_msg_hash": _hash_json("startup"),
                }
            )
        except Exception:
            pass  # best-effort; keep service up even if telemetry fails

        yield

        # Shutdown: stop plugins gracefully
        with contextlib.suppress(Exception):
            for p in getattr(app.state, "plugins", []):  # type: ignore
                stop = getattr(p, "stop", None)
                if callable(stop):
                    await stop()

        # Restore original event bus emit and stop broadcaster
        with contextlib.suppress(Exception):
            if app.state._orig_emit is not None:  # type: ignore
                app.state.event_bus.emit = app.state._orig_emit  # type: ignore[attr-defined]
                app.state._orig_emit = None  # type: ignore
        with contextlib.suppress(Exception):
            bc = getattr(app.state, "mcp_broadcaster", None)  # type: ignore
            if bc is not None:
                await bc.stop()
                app.state.mcp_broadcaster = None  # type: ignore

    # Register lifespan context (Starlette/FastAPI)
    app.router.lifespan_context = _lifespan  # type: ignore[attr-defined]

    # Mount routers
    prefix = SETTINGS.api_prefix
    if prefix and prefix != "/":
        if not prefix.startswith("/"):
            prefix = f"/{prefix}"
        prefix = prefix.rstrip("/")
        app.include_router(agent_router, prefix=prefix)  # type: ignore
        app.include_router(tools_router, prefix=prefix)  # type: ignore
        with contextlib.suppress(Exception):
            app.include_router(ability_router, prefix=prefix)  # type: ignore
        app.include_router(autogen_router, prefix=prefix)  # type: ignore
        # Register chat router (SSE + JSON) under API prefix (best-effort)
        with contextlib.suppress(Exception):
            app.include_router(chat_router, prefix=prefix)  # type: ignore
        # Register unified API gateway
        with contextlib.suppress(Exception):
            app.include_router(api_router, prefix=prefix)  # type: ignore
        # Register auth router
        with contextlib.suppress(Exception):
            app.include_router(auth_router, prefix=prefix)  # type: ignore
        # GUI router under prefix
        with contextlib.suppress(Exception):
            app.include_router(gui_router, prefix=prefix)  # type: ignore

        # Automatic message optimization middleware (HTTP level)
        @app.middleware("http")  # type: ignore
        async def _optimize_incoming(
            request: Request,
            call_next: Callable,
        ) -> Any:
            with contextlib.suppress(Exception):
                # Only process JSON chat route
                if (
                    request.headers.get("content-type", "").startswith(
                        "application/json"
                    )
                    and "/chat/stream" in request.url.path
                ):
                    with contextlib.suppress(Exception):
                        from reug_runtime.config import (
                            SETTINGS as RT_SETTINGS,
                        )
                        from reug_runtime.message_mw import (
                            MessageContext,
                            apply_all,
                        )

                        # Ensure amplifier is registered when enabled
                        if RT_SETTINGS.message_optimizer_enabled:
                            with contextlib.suppress(Exception):
                                import src.plugins.message_amplifier_plugin  # noqa: F401

                    # If optimizer is enabled, attempt to rewrite body
                    if (
                        RT_SETTINGS is not None
                        and getattr(RT_SETTINGS, "message_optimizer_enabled", False)
                        and apply_all is not None
                    ):
                        raw = await request.body()  # type: ignore
                        with contextlib.suppress(Exception):
                            payload = json.loads(raw.decode("utf-8") or "{}")  # type: ignore
                        msg = payload.get("message")
                        if isinstance(msg, str) and msg:
                            session_id = payload.get("session_id") or "default"
                            optimized, steps = apply_all(  # type: ignore
                                msg,
                                MessageContext(session_id=session_id),  # type: ignore
                            )
                            if getattr(
                                RT_SETTINGS, "message_optimizer_emit_telemetry", True
                            ):
                                with contextlib.suppress(Exception):
                                    await app.state.event_bus.emit(  # type: ignore
                                        {
                                            "type": "MessageOptimized",
                                            "correlation_id": f"http-{session_id}",
                                            "len_in": len(msg),
                                            "len_out": len(optimized),
                                            "steps": steps,
                                            "source": "http_mw",
                                        }
                                    )
                            max_len = getattr(
                                RT_SETTINGS, "message_optimizer_max_len", 6000
                            )
                            if len(optimized) > max_len:
                                optimized = optimized[:max_len]
                            payload["message"] = optimized
                            new_body = json.dumps(payload).encode("utf-8")

                            # Rebuild request with new body
                            async def _receive() -> dict[str, Any]:
                                return {
                                    "type": "http.request",
                                    "body": new_body,
                                    "more_body": False,
                                }

                            request = Request(request.scope, _receive)  # type: ignore
            return await call_next(request)

    else:
        app.include_router(agent_router)  # type: ignore
        app.include_router(tools_router)  # type: ignore
        with contextlib.suppress(Exception):
            app.include_router(ability_router)  # type: ignore
        app.include_router(autogen_router)  # type: ignore
        # Register chat router (SSE + JSON) without prefix (best-effort)
        with contextlib.suppress(Exception):
            app.include_router(chat_router)  # type: ignore
        # Register unified API gateway without prefix
        with contextlib.suppress(Exception):
            app.include_router(api_router)  # type: ignore
        # Register auth router without prefix
        with contextlib.suppress(Exception):
            app.include_router(auth_router)  # type: ignore
        # GUI router without prefix
        with contextlib.suppress(Exception):
            app.include_router(gui_router)  # type: ignore

    # Startup events are handled via lifespan; on_event is deprecated

    # DeepCode trigger endpoint (fire-and-forget). Accept generic JSON to
    # reduce tight coupling / avoid Pydantic forward issues across versions.
    @app.post("/deepcode/request")  # type: ignore
    async def deepcode_request(
        req: Request,  # type: ignore
        _auth: None = Depends(require_api_key),  # type: ignore
        _rl: None = Depends(enforce_rate_limit),  # type: ignore
    ) -> dict[str, Any]:  # type: ignore
        body = {}  # type: ignore[var-annotated]
        with contextlib.suppress(Exception):
            body = await req.json()  # type: ignore
        payload: dict[str, Any] = {
            "source_plugin": "http_gateway",
            "task_kind": body.get("task_kind", "generic"),
            "requirements": body.get("requirements", ""),
            "repo_path": body.get("repo_path", "."),
            "conversation_id": body.get("conversation_id"),
        }
        evt = create_event("deepcode_request", **payload)
        # Use publish if available to notify orchestrator
        if hasattr(app.state.event_bus, "publish"):  # type: ignore
            await app.state.event_bus.publish(evt.model_dump())  # type: ignore
        else:
            await app.state.event_bus.emit(evt.model_dump())  # type: ignore
        return {"status": "accepted", "request": payload}

    # Generic ability execution endpoint (internal tools registry exposure)
    _abilities_admin_only = os.getenv(
        "ALITA_ABILITIES_ADMIN_ONLY", "false"
    ).lower() in {"1", "true", "yes", "on"}
    _ability_whitelist: set[str] = set(
        [
            x.strip()
            for x in (os.getenv("ALITA_ABILITY_WHITELIST", "") or "").split(",")
            if x.strip()
        ]
    )

    @app.post("/ability/execute/{tool_id}")  # type: ignore
    async def execute_ability(
        tool_id: str,
        req: Request,  # type: ignore
        _auth: None = Depends(require_api_key),  # type: ignore
        _rl: None = Depends(enforce_rate_limit),  # type: ignore
    ) -> JSONResponse:  # type: ignore
        from src.core.api_models import api_error

        args: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            parsed = await req.json()  # type: ignore
            if isinstance(parsed, dict):
                args = parsed
        registry: SimpleAbilityRegistry = app.state.ability_registry  # type: ignore
        # Policy enforcement
        if _abilities_admin_only:
            # Require admin token
            try:
                await require_admin(req)  # type: ignore
            except Exception as e:  # pragma: no cover - simple mapping
                return JSONResponse(status_code=403, content=api_error("admin_required", "ADMIN", {"detail": str(e)}))  # type: ignore
        elif _ability_whitelist and tool_id not in _ability_whitelist:
            return JSONResponse(status_code=403, content=api_error("ability_not_allowed", "ABILITY", {"tool": tool_id}))  # type: ignore
        if not registry.knows(tool_id):
            return JSONResponse(status_code=404, content=api_error("unknown_tool", "ABILITY", {"tool": tool_id}))  # type: ignore
        if not registry.validate_args(tool_id, args):
            return JSONResponse(status_code=400, content=api_error("invalid_args", "ABILITY", {"tool": tool_id, "args": args}))  # type: ignore
        result = await registry.execute(tool_id, args)
        return JSONResponse(  # type: ignore
            status_code=200,
            content={"tool": tool_id, "result": result},
        )

    # Bandit stats snapshot (consumed by IDE agent)
    @app.get("/bandit/stats")  # type: ignore
    async def bandit_stats() -> JSONResponse:  # type: ignore
        try:
            from cortex.bandit_stats_store import get_snapshot

            snap = get_snapshot()
            return JSONResponse(
                status_code=200,
                content={
                    "tools": snap.tools,
                    "generated_at": snap.generated_at,
                },
            )  # type: ignore
        except Exception as e:  # pragma: no cover
            return JSONResponse(status_code=200, content={"tools": [], "error": str(e)})  # type: ignore

    # DeepCode latest retrieval
    @app.get("/deepcode/latest")  # type: ignore
    async def deepcode_latest(
        _auth: None = Depends(require_api_key),  # type: ignore
        _rl: None = Depends(enforce_rate_limit),  # type: ignore
    ) -> JSONResponse:  # type: ignore
        plugin = None
        for p in getattr(app.state, "plugins", []):  # type: ignore
            if getattr(p, "name", None) == "deepcode_orchestrator":
                plugin = p
                break
        if not plugin or not hasattr(plugin, "get_latest"):
            return JSONResponse(
                status_code=404, content={"error": "orchestrator_not_ready"}
            )  # type: ignore
        latest = plugin.get_latest()  # type: ignore
        if not latest:
            return JSONResponse(status_code=404, content={"error": "no_latest"})  # type: ignore
        return JSONResponse(status_code=200, content=latest)  # type: ignore

    # Bandit endpoints: decide + feedback
    @app.post("/bandit/decide")  # type: ignore
    async def bandit_decide(req: Request) -> JSONResponse:  # type: ignore
        body = await req.json()  # type: ignore
        policy_id = body.get("policy_id", "default")
        from cortex.bandit_service import decide as bandit_decide_impl

        result = bandit_decide_impl(policy_id)
        return JSONResponse(status_code=200, content=result)  # type: ignore

    @app.post("/bandit/feedback")  # type: ignore
    async def bandit_feedback(req: Request) -> JSONResponse:  # type: ignore
        body = await req.json()  # type: ignore
        decision_id = body.get("decision_id")
        reward = float(body.get("reward", 0.0))
        source = body.get("source")
        if not decision_id:
            return JSONResponse(
                status_code=400, content={"error": "missing decision_id"}
            )  # type: ignore
        from cortex.bandit_service import feedback as bandit_feedback_impl

        result = bandit_feedback_impl(decision_id, reward, source)
        return JSONResponse(status_code=200, content=result)  # type: ignore

    # DeepCode apply endpoint (delegates to orchestrator; guardian applies elsewhere)
    @app.post("/deepcode/apply")  # type: ignore
    async def deepcode_apply(
        req: Request,  # type: ignore
        _auth: None = Depends(require_api_key),  # type: ignore
        _rl: None = Depends(enforce_rate_limit),  # type: ignore
    ) -> JSONResponse:  # type: ignore
        raw = await req.json()  # type: ignore
        body = raw if isinstance(raw, dict) else {}
        filter_paths = (
            body.get("paths") if isinstance(body.get("paths"), list) else None
        )
        plugin = None
        for p in getattr(app.state, "plugins", []):  # type: ignore
            if getattr(p, "name", None) == "deepcode_orchestrator":
                plugin = p
                break
        if not plugin or not hasattr(plugin, "apply_latest"):
            return JSONResponse(
                status_code=404, content={"error": "orchestrator_not_ready"}
            )  # type: ignore
        result = await plugin.apply_latest(filter_paths)  # type: ignore
        return JSONResponse(status_code=200, content=result)  # type: ignore

    # Serve static chat UI if available
    try:
        static_dir = ROOT / "static"
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")  # type: ignore

            @app.get("/")  # type: ignore
            async def index() -> Any:  # type: ignore
                index_path = static_dir / "index.html"
                if index_path.exists():
                    return FileResponse(str(index_path))  # type: ignore
                return JSONResponse({"status": "ok", "message": "Super Alita runtime"})  # type: ignore

    except Exception:
        # Static serving is optional; ignore errors to keep runtime resilient
        pass

    # Simplified minimal health (no deep dependency checks)
    @app.get("/health/simple")  # type: ignore
    async def health_simple() -> dict[str, object]:  # type: ignore
        # minimal health with a timestamp for clients/tests
        return {"status": "ok", "timestamp": int(time.time())}

    # Route enumeration (debug only)
    @app.get("/routes")  # type: ignore
    async def list_routes() -> list[dict[str, str]]:  # type: ignore
        info: list[dict[str, str]] = []
        for r in app.router.routes:  # type: ignore[attr-defined]
            path = getattr(r, "path", "")
            methods = ",".join(sorted(getattr(r, "methods", []) or []))
            if path:
                info.append({"path": path, "methods": methods})
        return info

    # MCP health mirror: checks for recent telemetry events
    @app.get("/mcp/health")  # type: ignore
    async def mcp_health() -> dict[str, object]:  # type: ignore
        tf = os.getenv(
            "SUPER_ALITA_TELEMETRY_FILE",
            os.path.join(str(ROOT), "logs", "mcp_telemetry.jsonl"),
        )
        exists = os.path.exists(tf)
        last_event: dict[str, object] | None = None
        try:
            if exists:
                with open(tf, encoding="utf-8") as f:
                    # Read last non-empty JSON line (best-effort)
                    for line in reversed(f.readlines()[-200:]):  # limit tail scan
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                            last_event = {
                                k: evt.get(k)
                                for k in ("type", "tool", "duration_ms", "error")
                            }
                            break
                        except Exception:
                            continue
        except Exception:
            pass
        status = "unknown"
        if exists:
            status = (
                "ok"
                if last_event
                and last_event.get("type") in {"AbilitySucceeded", "AbilityCalled"}
                else "degraded"
            )
        return {
            "status": status,
            "telemetry_file": tf,
            "exists": exists,
            "last_event": last_event or {},
            "timestamp": int(time.time()),
        }

    # Plugin system health endpoint
    @app.get("/plugins/health")  # type: ignore
    async def plugin_health() -> dict[str, object]:  # type: ignore
        """Get health status of all loaded plugins."""
        try:
            from src.core.enhanced_plugin_system import get_plugin_health_status

            return get_plugin_health_status(app)
        except Exception as e:
            return {
                "plugin_count": 0,
                "plugins": {},
                "status": "error",
                "error": str(e),
            }

    # Mount static files for chat interface
    static_dir = ROOT / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")  # type: ignore

        # Serve chat interface at root
        @app.get("/")  # type: ignore
        async def serve_chat():  # type: ignore
            return FileResponse(str(static_dir / "index.html"))  # type: ignore

    return app


app = create_app()

# Optional CLI entry (e.g., python src/main.py --no-chat just validates startup)
if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("[ERROR] FastAPI dependencies not available. Install with:")
        print("pip install fastapi uvicorn")
        sys.exit(1)

    if app is None:
        print("[ERROR] Failed to create app")
        sys.exit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--no-chat",
        action="store_true",
        help="Boot only; don't open sockets beyond uvicorn",
    )
    ap.add_argument(
        "--reload",
        action="store_true",
        help="Reload server on code changes (dev mode)",
    )
    args = ap.parse_args()

    async def _dependency_health() -> dict[str, bool]:
        results: dict[str, bool] = {}
        with contextlib.suppress(Exception):
            await app.state.event_bus.emit({"event": "health_check"})  # type: ignore
            results["event_bus"] = True
        with contextlib.suppress(Exception):
            contract = app.state.ability_registry.get_available_tools_schema()[0]  # type: ignore
            results["ability_registry"] = await app.state.ability_registry.health_check(
                contract
            )  # type: ignore
        with contextlib.suppress(Exception):
            await app.state.kg.get_goal_for_session("health")  # type: ignore
            results["kg"] = True
        with contextlib.suppress(Exception):
            stream_gen = app.state.llm_model.stream_chat([], timeout=1)  # type: ignore
            await stream_gen.__anext__()
            results["llm_model"] = True
        return results

    if args.no_chat:
        checks = asyncio.run(_dependency_health())
        print(json.dumps(checks))
        raise SystemExit(0)

    # Start the ASGI server; pass the actual app object to avoid module path issues
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)  # type: ignore
