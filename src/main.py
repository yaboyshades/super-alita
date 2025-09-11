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
from contextlib import suppress  # noqa: E402

from reug_runtime.event_bus import (  # noqa: E402
    BaseEventBus,
    FileEventBus,
    make_event_bus,
)
from reug_runtime.llm_client import LLMClient, get_llm_client  # noqa: E402
from src.constitutional_gateway import constitutional_router  # noqa: E402
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
        store = getattr(globals().get("app"), "state", object()).__dict__.get(
            "api_key_store"
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
        rl = getattr(globals().get("app"), "state", object()).__dict__.get(
            "rate_limiter"
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
                    llm = getattr(globals().get("app"), "state", object()).__dict__.get(
                        "llm_model"
                    )
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
        # Validate args against contract schema (best-effort JSON Schema)
        try:
            contract = self._contracts.get(tool_name, {})
            schema = contract.get("input_schema")
            if isinstance(schema, dict):
                try:
                    # Optional dependency: jsonschema
                    import jsonschema  # type: ignore

                    jsonschema.validate(args, schema)  # type: ignore[arg-type]
                except ModuleNotFoundError:
                    # Fallback: check required fields only
                    required = schema.get("required", []) or []
                    missing = [k for k in required if k not in args]
                    if missing:
                        return {
                            "error": "invalid_args",
                            "message": f"missing required fields: {', '.join(missing)}",
                            "missing": missing,
                        }
                except Exception as e:  # pragma: no cover - schema errors
                    return {"error": "invalid_args", "message": str(e)}
        except Exception:
            # Never block execution due to validator crash
            pass
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
                # Placeholder/mock detection to enforce no-mock policy
                (
                    r"\bNotImplementedError\b",
                    "MEDIUM",
                    "NotImplementedError placeholder",
                ),
                (r"\bTODO\b", "LOW", "TODO found in generated code"),
                (r"\bFIXME\b", "LOW", "FIXME found in code"),
                (r"\bplaceholder\b", "LOW", "Placeholder marker found"),
                (r"\bdummy\b", "LOW", "Dummy marker found"),
                (r"\bmock\b", "LOW", "Mock marker found"),
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

        ability_reg = getattr(globals().get("app"), "state", object()).__dict__.get(
            "ability_registry"
        )
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
                or getattr(globals().get("app"), "state", object()).__dict__.get(
                    "llm_model"
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
        llm = getattr(app, "state", object()).__dict__.get("llm_model")

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
        llm = getattr(app, "state", object()).__dict__.get("llm_model")

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

    # Mount unified chat static assets (best-effort)
    try:  # noqa: SIM105
        static_dir = Path("static")
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")  # type: ignore

            @app.get("/", include_in_schema=False)  # type: ignore[misc]
            async def _root_ui() -> Response:  # type: ignore
                index_path = static_dir / "index.html"
                if index_path.exists():
                    return FileResponse(index_path)  # type: ignore
                return JSONResponse({"message": "Super Alita running"})  # type: ignore

    except Exception as e:  # pragma: no cover - non critical
        print(f"⚠️  Static mount failed: {e}")

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

    # Include unified chat API if available
    try:
        from src.api.chat_endpoints import router as chat_router

        app.include_router(chat_router)  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  Unified chat endpoints not loaded: {e}")

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

        # Register Task Planner ability
        try:
            print("🔧 DEBUG: Starting task planner registration...")
            from src.abilities.planner_ability import plan_task

            planner_contract = {
                "tool_id": "task_planner",
                "description": "Decompose objectives into atomic, tool-oriented steps",
                "input_schema": {
                    "type": "object",
                    "required": ["prompt"],
                    "properties": {
                        "prompt": {"type": "string"},
                        "max_steps": {
                            "type": "integer",
                            "default": 6,
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "action": {"type": "string"},
                                    "rationale": {"type": "string"},
                                },
                            },
                        },
                        "summary": {"type": "string"},
                        "source": {
                            "type": "string",
                            "enum": ["llm", "heuristic", "fallback"],
                        },
                    },
                },
            }

            async def planner_executor(args: dict[str, Any]) -> dict[str, Any]:
                return plan_task(
                    prompt=args["prompt"],
                    max_steps=args.get("max_steps", 6),
                )

            ability_reg.register_tool(
                contract=planner_contract, executor=planner_executor
            )
            print("✅ DEBUG: Task planner tool registered")
        except Exception as e:  # noqa: BLE001
            print(f"❌ DEBUG: Failed to register task planner tool: {e}")
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

        # Register Repository + Paper MCP-style tools (dynamic abilities)
        try:
            from pathlib import Path as _Path

            from src.vscode_integration.paper_ingestion_tool import (
                PaperIngestionTool,
            )
            from src.vscode_integration.repo_mcp_tool import (
                RepositoryMCPTool,
            )

            ability_reg = app.state.ability_registry  # type: ignore[attr- defined]

            # Initialize tool instances
            repo_tool = RepositoryMCPTool(repo_root=_Path.cwd())
            await repo_tool.initialize(app.state.event_bus)  # type: ignore[attr-defined]

            paper_tool = PaperIngestionTool()
            await paper_tool.initialize(app.state.event_bus)  # type: ignore[attr-defined]

            # Repository: list files
            async def _repo_list_files(args: dict[str, Any]) -> dict[str, Any]:
                return await repo_tool.list_files(
                    directory=args.get("directory", ""),
                    pattern=args.get("pattern", "**/*"),
                    exclude_patterns=args.get("exclude_patterns"),
                )

            ability_reg.register_tool(
                contract={
                    "tool_id": "repo_list_files",
                    "description": "List files in repository directory with filtering",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string", "default": ""},
                            "pattern": {"type": "string", "default": "**/*"},
                            "exclude_patterns": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_repo_list_files,
            )

            # Repository: read file
            async def _repo_read_file(args: dict[str, Any]) -> dict[str, Any]:
                return await repo_tool.read_file(
                    file_path=args.get("file_path", ""),
                    max_lines=args.get("max_lines"),
                )

            ability_reg.register_tool(
                contract={
                    "tool_id": "repo_read_file",
                    "description": "Read file content from repository",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "max_lines": {"type": "integer"},
                        },
                        "required": ["file_path"],
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_repo_read_file,
            )

            # Repository: write file
            async def _repo_write_file(args: dict[str, Any]) -> dict[str, Any]:
                return await repo_tool.write_file(
                    file_path=args.get("file_path", ""),
                    content=args.get("content", ""),
                    create_dirs=bool(args.get("create_dirs", True)),
                )

            ability_reg.register_tool(
                contract={
                    "tool_id": "repo_write_file",
                    "description": "Write content to repository file",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"},
                            "create_dirs": {"type": "boolean", "default": True},
                        },
                        "required": ["file_path", "content"],
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_repo_write_file,
            )

            # Repository: search code
            async def _repo_search_code(args: dict[str, Any]) -> dict[str, Any]:
                return await repo_tool.search_code(
                    query=args.get("query", ""),
                    file_pattern=args.get("file_pattern", "**/*.py"),
                    context_lines=int(args.get("context_lines", 3)),
                )

            ability_reg.register_tool(
                contract={
                    "tool_id": "repo_search_code",
                    "description": "Search for code patterns in repository",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "file_pattern": {"type": "string", "default": "**/*.py"},
                            "context_lines": {"type": "integer", "default": 3},
                        },
                        "required": ["query"],
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_repo_search_code,
            )

            # Repository: git history
            async def _repo_git_history(args: dict[str, Any]) -> dict[str, Any]:
                return await repo_tool.get_git_history(
                    file_path=args.get("file_path"),
                    limit=int(args.get("limit", 10)),
                )

            ability_reg.register_tool(
                contract={
                    "tool_id": "repo_git_history",
                    "description": "Get git commit history for file or repository",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "limit": {"type": "integer", "default": 10},
                        },
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_repo_git_history,
            )

            # Paper: extract text
            async def _paper_extract_text(args: dict[str, Any]) -> dict[str, Any]:
                return await paper_tool.extract_text_from_pdf(args.get("pdf_path", ""))

            ability_reg.register_tool(
                contract={
                    "tool_id": "paper_extract_text",
                    "description": "Extract text content from PDF research papers",
                    "input_schema": {
                        "type": "object",
                        "properties": {"pdf_path": {"type": "string"}},
                        "required": ["pdf_path"],
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_paper_extract_text,
            )

            # Paper: summary
            async def _paper_generate_summary(args: dict[str, Any]) -> dict[str, Any]:
                return await paper_tool.generate_paper_summary(
                    pdf_path=args.get("pdf_path", ""),
                    focus_areas=args.get("focus_areas"),
                )

            ability_reg.register_tool(
                contract={
                    "tool_id": "paper_generate_summary",
                    "description": "Generate focused summary of research paper",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "pdf_path": {"type": "string"},
                            "focus_areas": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["pdf_path"],
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_paper_generate_summary,
            )

            # Paper: download
            async def _paper_download(args: dict[str, Any]) -> dict[str, Any]:
                return await paper_tool.download_paper(
                    url=args.get("url", ""),
                    filename=args.get("filename"),
                )

            ability_reg.register_tool(
                contract={
                    "tool_id": "paper_download",
                    "description": "Download research paper from URL",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "filename": {"type": "string"},
                        },
                        "required": ["url"],
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_paper_download,
            )

            # Paper: local search
            async def _paper_search(args: dict[str, Any]) -> dict[str, Any]:
                return await paper_tool.search_papers(
                    query=args.get("query", ""),
                    max_results=int(args.get("max_results", 10)),
                )

            ability_reg.register_tool(
                contract={
                    "tool_id": "paper_search_local",
                    "description": "Search previously ingested/cached papers",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "default": 10},
                        },
                        "required": ["query"],
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_paper_search,
            )

            print("? DEBUG: Repository + Paper tools registered")
        except Exception as e:
            print(f"? DEBUG: Failed to register repo/paper tools: {e}")
            import traceback

            traceback.print_exc()

        # Register lightweight code writing abilities (no DeepCode dependency)
        try:
            ability_reg = app.state.ability_registry  # type: ignore[attr-defined]

            def _code_prompt(
                language: str,
                spec: str,
                style_guides: str | None,
                constraints: str | None,
            ) -> str:
                parts = [
                    "You are a senior software engineer. Produce ONLY code in the requested language.",
                    f"Language: {language}",
                    f"Specification: {spec}",
                ]
                if style_guides:
                    parts.append(f"Style Guides: {style_guides}")
                if constraints:
                    parts.append(f"Constraints: {constraints}")
                parts.append(
                    "Output: Provide a single complete code listing, no commentary."
                )
                parts.append(
                    "Policy: No mock/dummy/placeholder code. No TODO/FIXME/NotImplementedError. Provide a full, working implementation."
                )
                return "\n".join(parts)

            async def _code_synthesize(args: dict[str, Any]) -> dict[str, Any]:
                language = (args.get("language") or "").strip() or "python"
                spec = (args.get("spec") or "").strip()
                style_guides = (args.get("style_guides") or "").strip() or None
                constraints = (args.get("constraints") or "").strip() or None
                prompt = _code_prompt(language, spec, style_guides, constraints)
                result = await ability_reg.execute(  # type: ignore
                    "deepconf_consensus",
                    {"prompt": prompt, "method": "weighted_vote", "num_samples": 3},
                )
                code = (
                    result.get("best_response")
                    or result.get("consensus_text")
                    or result.get("consensus_result")
                    or ""
                )
                return {
                    "language": language,
                    "code": code,
                    "notes": {"method": "consensus"},
                }

            ability_reg.register_tool(
                contract={
                    "tool_id": "code_synthesize",
                    "description": "Generate code from a natural language specification using consensus prompting",
                    "input_schema": {
                        "type": "object",
                        "required": ["language", "spec"],
                        "properties": {
                            "language": {"type": "string"},
                            "spec": {"type": "string"},
                            "style_guides": {"type": "string"},
                            "constraints": {"type": "string"},
                        },
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "language": {"type": "string"},
                            "code": {"type": "string"},
                            "notes": {"type": "object"},
                        },
                    },
                },
                executor=_code_synthesize,
            )

            async def _code_synthesize_and_write(
                args: dict[str, Any],
            ) -> dict[str, Any]:
                file_path = (args.get("file_path") or "").strip()
                if not file_path:
                    return {"error": "file_path is required"}
                force_write = bool(args.get("force_write", False))
                test_first = bool(args.get("test_first", True))
                consolidate_tests = bool(args.get("consolidate_tests", True))

                # Optionally generate tests first (TDD)
                test_write: dict[str, Any] | None = None
                if test_first:
                    test_lang = (
                        args.get("test_language") or args.get("language") or "python"
                    ).strip()
                    test_spec = (
                        args.get("test_spec") or args.get("spec") or ""
                    ).strip()
                    test_file_path = (args.get("test_file_path") or "").strip()
                    if not test_file_path:
                        if consolidate_tests:
                            test_file_path = "tests/test_codegen.py"
                        else:
                            from pathlib import Path as _P

                            fp = _P(file_path)
                            rel_dirs = list(fp.parent.parts)
                            if rel_dirs and rel_dirs[0] in {"src", "app", "services"}:
                                rel_dirs = rel_dirs[1:]
                            test_file_path = (
                                _P("tests")
                                .joinpath(*rel_dirs, f"test_{fp.stem}.py")
                                .as_posix()
                            )

                    test_prompt = "\n".join(
                        [
                            "Write unit tests for the following specification.",
                            f"Target code path: {file_path}",
                            f"Language: {test_lang}",
                            "Use idiomatic testing for the language (pytest for Python).",
                            "Include realistic edge cases. Avoid placeholders like TODO/FIXME.",
                            "Prefer deterministic tests without external network calls.",
                            *([f"Specification: {test_spec}"] if test_spec else []),
                        ]
                    )
                    t_res = await ability_reg.execute(  # type: ignore
                        "deepconf_consensus",
                        {
                            "prompt": test_prompt,
                            "method": "weighted_vote",
                            "num_samples": 3,
                        },
                    )
                    t_code = (
                        t_res.get("best_response")
                        or t_res.get("consensus_text")
                        or t_res.get("consensus_result")
                        or ""
                    )
                    # Append to existing test file when present
                    prior = await ability_reg.execute("repo_read_file", {"file_path": test_file_path})  # type: ignore
                    if prior and not prior.get("error") and prior.get("content"):
                        new_content = (prior.get("content") or "") + "\n\n" + t_code
                    else:
                        new_content = (
                            f"# Auto-generated tests for {file_path}\n\n" + t_code
                        )
                    test_write = await ability_reg.execute(  # type: ignore
                        "repo_write_file",
                        {
                            "file_path": test_file_path,
                            "content": new_content,
                            "create_dirs": True,
                        },
                    )

                # 1) synthesize code
                synth = await _code_synthesize(args)
                code = synth.get("code") or ""
                # 2) quick safety scan
                scan = await ability_reg.execute(  # type: ignore
                    "secure_scan_code", {"code": code}
                )
                issues = scan.get("issues", [])
                issue_count = int(scan.get("issue_count", len(issues)))
                wrote = False
                write_res: dict[str, Any] | None = None
                if force_write or issue_count == 0:
                    write_res = await ability_reg.execute(  # type: ignore
                        "repo_write_file",
                        {"file_path": file_path, "content": code, "create_dirs": True},
                    )
                    wrote = bool(write_res and write_res.get("success"))
                return {
                    "file_path": file_path,
                    "wrote": wrote,
                    "issues": issues,
                    "issue_count": issue_count,
                    "synth": {
                        "language": synth.get("language"),
                        "notes": synth.get("notes"),
                    },
                    **({"write_result": write_res} if write_res else {}),
                    **({"test_write": test_write} if test_write else {}),
                }

            ability_reg.register_tool(
                contract={
                    "tool_id": "code_synthesize_and_write",
                    "description": "Generate code from a spec, scan it, and (optionally) write to the repo",
                    "input_schema": {
                        "type": "object",
                        "required": ["language", "spec", "file_path"],
                        "properties": {
                            "language": {"type": "string"},
                            "spec": {"type": "string"},
                            "file_path": {"type": "string"},
                            "style_guides": {"type": "string"},
                            "constraints": {"type": "string"},
                            "force_write": {"type": "boolean", "default": False},
                            "test_first": {"type": "boolean", "default": True},
                            "consolidate_tests": {"type": "boolean", "default": True},
                            "test_file_path": {"type": "string"},
                            "test_language": {"type": "string"},
                            "test_spec": {"type": "string"},
                        },
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_code_synthesize_and_write,
            )

            print("? DEBUG: Code synthesis abilities registered")
        except Exception as e:
            print(f"? DEBUG: Failed to register code synthesis abilities: {e}")
            import traceback

            traceback.print_exc()

        # Register hybrid reasoning pipeline (extract → reason → synthesize/validate)
        try:
            ability_reg = app.state.ability_registry  # type: ignore[attr-defined]

            async def _hybrid_reasoning_pipeline(
                args: dict[str, Any],
            ) -> dict[str, Any]:
                """Extract structure → reason → synthesize/validate.

                Args:
                  text: Optional raw text to extract from.
                  pdf_path: Optional paper path under ./papers to extract from.
                  file_path: Optional repository file to read as input.
                  mangle_query: Optional Mangle query to run after asserting facts/rules.
                  explain: bool, include Mangle explanation.
                  generate_code: bool, synthesize code from the spec.
                  language: target language for code generation.
                  code_spec: natural language spec for code generation (optional).
                  file_output: optional repo path to write synthesized code.
                  validation_queries: list of Mangle queries to validate code/invariants.
                """
                import json as _json
                import re as _re

                def _extract_json_block(s: str) -> dict[str, Any] | None:
                    m = _re.search(r"```json\s*(\{[\s\S]*?\})\s*```", s)
                    if not m:
                        return None
                    try:
                        return _json.loads(m.group(1))
                    except Exception:
                        return None

                def _fallback_parse(s: str) -> tuple[list[str], list[dict[str, str]]]:
                    facts: list[str] = []
                    rules: list[dict[str, str]] = []
                    for line in s.splitlines():
                        t = line.strip()
                        if not t:
                            continue
                        if ":-" in t and t.endswith("."):
                            rules.append({"name": "rule_auto", "rule": t})
                        elif t.endswith(".") and _re.match(
                            r"^[a-z_][a-zA-Z0-9_,()\s]*\.$", t
                        ):
                            facts.append(t)
                    return facts, rules

                # 1) Acquire source text
                source_text = (args.get("text") or "").strip()
                if not source_text and args.get("pdf_path"):
                    res = await ability_reg.execute("paper_extract_text", {"pdf_path": args["pdf_path"]})  # type: ignore
                    source_text = (res.get("text") or "").strip()
                if not source_text and args.get("file_path"):
                    resf = await ability_reg.execute("repo_read_file", {"file_path": args["file_path"]})  # type: ignore
                    source_text = (resf.get("content") or "").strip()
                if not source_text:
                    return {
                        "error": "no_input",
                        "hint": "Provide text, pdf_path, or file_path",
                    }

                # 2) Ask LLM to extract Mangle facts and rules
                extract_prompt = (
                    "Extract precise Mangle facts and rules from the text.\n"
                    "Return ONLY a fenced JSON block with keys: facts (array of strings, each ends with '.'),\n"
                    "rules (array of objects with fields 'name' and 'rule' where rule ends with '.').\n"
                    'Example:```json\n{"facts":["edge(a,b)."],"rules":[{"name":"reachable","rule":"reachable(X,Y) :- edge(X,Y)."}]}```\n\n'
                    f"Text:\n{source_text[:4000]}"
                )
                llm_res = await ability_reg.execute(  # type: ignore
                    "deepconf_consensus",
                    {
                        "prompt": extract_prompt,
                        "method": "weighted_vote",
                        "num_samples": 3,
                    },
                )
                raw = (
                    llm_res.get("best_response")
                    or llm_res.get("consensus_text")
                    or llm_res.get("consensus_result")
                    or ""
                )
                parsed = _extract_json_block(raw)
                facts: list[str] = []
                rules: list[dict[str, str]] = []
                if parsed and isinstance(parsed, dict):
                    facts = [
                        str(x).strip()
                        for x in parsed.get("facts", [])
                        if str(x).strip()
                    ]
                    rules = [
                        {
                            "name": str(r.get("name") or "rule_auto").strip(),
                            "rule": str(r.get("rule") or "").strip(),
                        }
                        for r in (parsed.get("rules", []) or [])
                        if isinstance(r, dict) and str(r.get("rule") or "").strip()
                    ]
                else:
                    facts, rules = _fallback_parse(raw)

                # 3) Assert into Mangle
                asserted: dict[str, Any] = {"facts": 0, "rules": 0}
                for f in facts[:200]:  # safety cap
                    await ability_reg.execute("mangle_add_fact", {"fact": f})  # type: ignore
                    asserted["facts"] += 1
                for i, r in enumerate(rules[:100]):  # safety cap
                    nm = r.get("name") or f"rule_{i+1}"
                    await ability_reg.execute("mangle_add_rule", {"name": nm, "rule": r.get("rule", "")})  # type: ignore
                    asserted["rules"] += 1

                # 4) Query / Explain
                mq = (args.get("mangle_query") or "").strip()
                query_result: dict[str, Any] | None = None
                explain_result: dict[str, Any] | None = None
                if mq:
                    query_result = await ability_reg.execute("mangle_query", {"query": mq})  # type: ignore
                    if args.get("explain"):
                        explain_result = await ability_reg.execute("mangle_explain", {"query": mq})  # type: ignore

                # 5) Optional code synthesis
                synth_result: dict[str, Any] | None = None
                if bool(args.get("generate_code")):
                    language = (args.get("language") or "python").strip()
                    spec = (
                        args.get("code_spec")
                        or "Implement the algorithm described by the extracted facts and rules."
                    ).strip()
                    file_output = (args.get("file_output") or "").strip()
                    code_args = {
                        "language": language,
                        "spec": spec,
                        "style_guides": args.get("style_guides"),
                        "constraints": args.get("constraints"),
                    }
                    if file_output:
                        code_args["file_path"] = file_output
                        synth_result = await ability_reg.execute("code_synthesize_and_write", code_args)  # type: ignore
                    else:
                        synth_result = await ability_reg.execute("code_synthesize", code_args)  # type: ignore

                # 6) Optional validation queries
                validations: list[dict[str, Any]] = []
                vqs = args.get("validation_queries") or []
                if isinstance(vqs, list):
                    for v in vqs[:20]:
                        q = str(v or "").strip()
                        if not q:
                            continue
                        vres = await ability_reg.execute("mangle_query", {"query": q})  # type: ignore
                        validations.append({"query": q, "result": vres})

                return {
                    "asserted": asserted,
                    "extraction_raw": raw,
                    "facts": facts,
                    "rules": rules,
                    **(
                        {"query_result": query_result}
                        if query_result is not None
                        else {}
                    ),
                    **(
                        {"explain_result": explain_result}
                        if explain_result is not None
                        else {}
                    ),
                    **({"synthesis": synth_result} if synth_result is not None else {}),
                    **({"validations": validations} if validations else {}),
                }

            ability_reg.register_tool(
                contract={
                    "tool_id": "hybrid_reasoning_pipeline",
                    "description": "Extract facts/rules → assert to Mangle → query/explain → optional code synth + validation",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "pdf_path": {"type": "string"},
                            "file_path": {"type": "string"},
                            "mangle_query": {"type": "string"},
                            "explain": {"type": "boolean", "default": False},
                            "generate_code": {"type": "boolean", "default": False},
                            "language": {"type": "string", "default": "python"},
                            "code_spec": {"type": "string"},
                            "file_output": {"type": "string"},
                            "style_guides": {"type": "string"},
                            "constraints": {"type": "string"},
                            "validation_queries": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_hybrid_reasoning_pipeline,
            )

            print("? DEBUG: Hybrid reasoning pipeline registered")
        except Exception as e:
            print(f"? DEBUG: Failed to register hybrid pipeline: {e}")
            import traceback

            traceback.print_exc()

        # Register GitHub discovery tools (search before building)
        try:
            ability_reg = app.state.ability_registry  # type: ignore[attr-defined]

            def _gh_headers() -> dict[str, str]:
                headers = {
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "super-alita",
                }
                token = os.getenv("GITHUB_TOKEN", "").strip()
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                return headers

            async def _github_search_code(args: dict[str, Any]) -> dict[str, Any]:
                import json
                from urllib.parse import quote_plus

                q = (args.get("q") or args.get("query") or "").strip()
                if not q:
                    return {"error": "missing query 'q'"}
                language = (args.get("language") or "").strip()
                repo = (args.get("repo") or "").strip()
                qualifiers: list[str] = []
                if language:
                    qualifiers.append(f"language:{language}")
                if repo:
                    qualifiers.append(f"repo:{repo}")
                q_full = "+".join([quote_plus(q)] + [quote_plus(x) for x in qualifiers])
                per_page = max(1, min(int(args.get("per_page", 10) or 10), 50))
                page = max(1, min(int(args.get("page", 1) or 1), 10))
                url = f"https://api.github.com/search/code?q={q_full}&per_page={per_page}&page={page}"
                req = urllib.request.Request(url, headers=_gh_headers())  # nosec B310
                try:
                    with urllib.request.urlopen(req, timeout=8) as resp:  # nosec B310
                        data = json.loads(resp.read().decode("utf-8", errors="replace"))
                except Exception as e:  # pragma: no cover - network variability
                    return {"error": str(e), "url": url}
                items = []
                for it in data.get("items", []) or []:
                    repo_obj = it.get("repository") or {}
                    items.append(
                        {
                            "name": it.get("name"),
                            "path": it.get("path"),
                            "repo": repo_obj.get("full_name") or repo,
                            "html_url": it.get("html_url"),
                            "score": it.get("score"),
                        }
                    )
                return {
                    "items": items,
                    "total_count": data.get("total_count", len(items)),
                    "incomplete_results": data.get("incomplete_results", False),
                    "url": url,
                }

            ability_reg.register_tool(
                contract={
                    "tool_id": "github_search_code",
                    "description": "Search public GitHub code for matches (uses GitHub Search API; set GITHUB_TOKEN to raise rate limits)",
                    "input_schema": {
                        "type": "object",
                        "required": ["q"],
                        "properties": {
                            "q": {"type": "string"},
                            "language": {"type": "string"},
                            "repo": {"type": "string"},
                            "per_page": {"type": "integer", "default": 10},
                            "page": {"type": "integer", "default": 1},
                        },
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_github_search_code,
            )

            async def _github_search_repos(args: dict[str, Any]) -> dict[str, Any]:
                import json
                from urllib.parse import quote_plus

                q = (args.get("q") or args.get("query") or "").strip()
                if not q:
                    return {"error": "missing query 'q'"}
                sort = (args.get("sort") or "stars").strip()
                order = (args.get("order") or "desc").strip()
                per_page = max(1, min(int(args.get("per_page", 10) or 10), 50))
                page = max(1, min(int(args.get("page", 1) or 1), 10))
                url = (
                    f"https://api.github.com/search/repositories?q={quote_plus(q)}&sort={quote_plus(sort)}&order={quote_plus(order)}"
                    f"&per_page={per_page}&page={page}"
                )
                req = urllib.request.Request(url, headers=_gh_headers())  # nosec B310
                try:
                    with urllib.request.urlopen(req, timeout=8) as resp:  # nosec B310
                        data = json.loads(resp.read().decode("utf-8", errors="replace"))
                except Exception as e:  # pragma: no cover
                    return {"error": str(e), "url": url}
                items = []
                for it in data.get("items", []) or []:
                    items.append(
                        {
                            "full_name": it.get("full_name"),
                            "html_url": it.get("html_url"),
                            "description": it.get("description"),
                            "stargazers_count": it.get("stargazers_count"),
                            "language": it.get("language"),
                        }
                    )
                return {
                    "items": items,
                    "total_count": data.get("total_count", len(items)),
                    "url": url,
                }

            ability_reg.register_tool(
                contract={
                    "tool_id": "github_search_repos",
                    "description": "Search GitHub repositories (uses GitHub Search API; set GITHUB_TOKEN to raise rate limits)",
                    "input_schema": {
                        "type": "object",
                        "required": ["q"],
                        "properties": {
                            "q": {"type": "string"},
                            "sort": {"type": "string", "default": "stars"},
                            "order": {"type": "string", "default": "desc"},
                            "per_page": {"type": "integer", "default": 10},
                            "page": {"type": "integer", "default": 1},
                        },
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_github_search_repos,
            )

            print("? DEBUG: GitHub discovery tools registered")
        except Exception as e:
            print(f"? DEBUG: Failed to register GitHub discovery tools: {e}")
            import traceback

            traceback.print_exc()

        # Register Python verification helpers and prompt evolution
        try:
            ability_reg = app.state.ability_registry  # type: ignore[attr-defined]

            async def _pytest_run(args: dict[str, Any]) -> dict[str, Any]:
                """Run pytest with optional target and markers.

                Args: target (str), markers (str), k (str), maxfail (int), quiet (bool)
                """
                from src.core.proc import arun

                target = str(args.get("target") or "tests")
                quiet = bool(args.get("quiet", True))
                markers = str(args.get("markers") or "").strip()
                k_expr = str(args.get("k") or "").strip()
                maxfail = args.get("maxfail")
                cmd: list[str] = [sys.executable, "-m", "pytest"]
                if quiet:
                    cmd.append("-q")
                if markers:
                    cmd.extend(["-m", markers])
                if k_expr:
                    cmd.extend(["-k", k_expr])
                if isinstance(maxfail, int) and maxfail > 0:
                    cmd.extend(["--maxfail", str(maxfail)])
                cmd.append(target)
                try:
                    out = await arun(cmd, timeout=300)
                    return {"ok": True, "exit_code": 0, "stdout": out, "stderr": ""}
                except (
                    Exception
                ) as e:  # pragma: no cover - returns stderr via exception
                    msg = str(e)
                    return {
                        "ok": False,
                        "exit_code": getattr(e, "returncode", 1),
                        "stderr": msg,
                        "stdout": getattr(e, "stdout", ""),
                    }

            ability_reg.register_tool(
                contract={
                    "tool_id": "pytest_run",
                    "description": "Run pytest with optional target and filters",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string", "default": "tests"},
                            "markers": {"type": "string"},
                            "k": {"type": "string"},
                            "maxfail": {"type": "integer"},
                            "quiet": {"type": "boolean", "default": True},
                        },
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_pytest_run,
            )

            async def _python_import_smoke(args: dict[str, Any]) -> dict[str, Any]:
                """Import all modules under a path (default 'src') and report failures."""
                import importlib
                import pkgutil

                base = str(args.get("path") or "src").rstrip("/\\")
                ok: list[str] = []
                fail: list[dict[str, str]] = []
                try:
                    for mod in pkgutil.walk_packages([base]):
                        name = mod.name
                        if not name.startswith("src."):
                            name = f"src.{name}"
                        try:
                            importlib.import_module(name)
                            ok.append(name)
                        except (
                            Exception
                        ) as e:  # pragma: no cover - environment dependent
                            fail.append({"module": name, "error": str(e)})
                except Exception as e:
                    return {"error": str(e)}
                return {
                    "ok_count": len(ok),
                    "fail_count": len(fail),
                    "failures": fail[:200],
                }

            ability_reg.register_tool(
                contract={
                    "tool_id": "python_import_smoke",
                    "description": "Attempt to import all modules under a base path (default 'src')",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_python_import_smoke,
            )

            async def _prompt_evolve(args: dict[str, Any]) -> dict[str, Any]:
                """Generate prompt variants and choose the best via simple heuristics.

                Inputs: prompt (str), variants (int), domain (str)
                Outputs: best_prompt, variants (list with scores)
                """
                import random

                base = (args.get("prompt") or "").strip()
                if not base:
                    return {"error": "missing prompt"}
                n = max(2, min(int(args.get("variants", 4) or 4), 8))
                domain = (args.get("domain") or "python").lower()

                def make_variant(p: str, kind: str) -> str:
                    blocks = {
                        "role": "You are a Senior Python Engineer.",
                        "constraints": "Constraints: PEP 8, full type hints, error handling, no placeholders.",
                        "examples": "Examples: Provide 1-2 concise input/output examples.",
                        "steps": "Steps: 1) Design 2) Implement 3) Tests 4) Optimize 5) Review.",
                        "format": "Output: Return only code in a single block.",
                        "tdd": "Testing: Write/append tests first, then code.",
                    }
                    order = [
                        "role",
                        "constraints",
                        "examples",
                        "steps",
                        "format",
                        "tdd",
                    ]
                    if kind == "long":
                        body = "\n".join([blocks[k] for k in order])
                    elif kind == "short":
                        body = "\n".join(
                            [blocks[k] for k in ["role", "constraints", "format"]]
                        )
                    else:  # mixed
                        random.shuffle(order)
                        body = "\n".join([blocks[k] for k in order[:4]])
                    return f"{p}\n\n{body}"

                variants: list[dict[str, Any]] = []
                for i in range(n):
                    kind = random.choice(
                        ["long", "short", "mixed"]
                    )  # non-deterministic variety
                    v = make_variant(base, kind)
                    # Heuristic score by presence of key sections
                    score = 0
                    for key in [
                        "Constraints:",
                        "Examples:",
                        "Steps:",
                        "Output:",
                        "Testing:",
                    ]:
                        if key.lower() in v.lower():
                            score += 1
                    variants.append({"prompt": v, "score": score, "kind": kind})

                best = (
                    max(variants, key=lambda x: x["score"])
                    if variants
                    else {"prompt": base, "score": 0}
                )
                return {"best_prompt": best["prompt"], "variants": variants}

            ability_reg.register_tool(
                contract={
                    "tool_id": "prompt_evolve",
                    "description": "Create structured prompt variants and pick the best by simple heuristics",
                    "input_schema": {
                        "type": "object",
                        "required": ["prompt"],
                        "properties": {
                            "prompt": {"type": "string"},
                            "variants": {"type": "integer"},
                            "domain": {"type": "string"},
                        },
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_prompt_evolve,
            )

            print("? DEBUG: Python verification + prompt evolution tools registered")
        except Exception as e:
            print(f"? DEBUG: Failed to register verification/evolution tools: {e}")
            import traceback

            traceback.print_exc()

        # Register shadow reward deployment abilities
        try:
            from src.cognitive.shadow_reward import (
                ShadowRewardDeployment,
                SimplePythonReward,
            )

            # Keep a single deployment instance on app.state
            if not hasattr(app.state, "shadow_reward"):
                stub = SimplePythonReward()

                class AltReward(
                    SimplePythonReward
                ):  # reuse logic; acts as "torch" adapter
                    async def compute_reward(self, code: str, context: dict | None = None) -> float:  # type: ignore[override]
                        # Slight variant: favor docstrings more
                        base = await super().compute_reward(code, context)
                        return min(1.0, base + 0.05)

                torch_like = AltReward()
                app.state.shadow_reward = ShadowRewardDeployment(stub, torch_like, {})  # type: ignore[attr-defined]

            async def _shadow_reward_score(args: dict[str, Any]) -> dict[str, Any]:
                code = str(args.get("code") or "")
                if not code:
                    return {"error": "missing code"}
                ctx = args.get("context") or {}
                deploy = app.state.shadow_reward  # type: ignore[attr-defined]
                score = await deploy.compute_reward_with_shadow(code, ctx)
                return {"score": score}

            ability_reg.register_tool(
                contract={
                    "tool_id": "shadow_reward_score",
                    "description": "Compute reward with shadow (alt) model in parallel; may progressively use alt based on correlation",
                    "input_schema": {
                        "type": "object",
                        "required": ["code"],
                        "properties": {
                            "code": {"type": "string"},
                            "context": {"type": "object"},
                        },
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_shadow_reward_score,
            )

            async def _shadow_reward_metrics(_: dict[str, Any]) -> dict[str, Any]:
                deploy = app.state.shadow_reward  # type: ignore[attr-defined]
                return deploy.get_metrics()

            ability_reg.register_tool(
                contract={
                    "tool_id": "shadow_reward_metrics",
                    "description": "Get shadow deployment metrics (correlation, rollout, means)",
                    "input_schema": {"type": "object", "properties": {}},
                    "output_schema": {"type": "object"},
                },
                executor=_shadow_reward_metrics,
            )

            print("? DEBUG: Shadow reward deployment registered")
        except Exception as e:
            print(f"? DEBUG: Failed to register shadow reward: {e}")
            import traceback

            traceback.print_exc()

        # Register Z3 verifier abilities (feature-flagged)
        try:
            if os.getenv("ALITA_ENABLE_Z3", "false").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                from src.cognitive.z3_verifier import ScalableZ3Verifier

                z3v = ScalableZ3Verifier(base_timeout=10, max_timeout=60)

                async def _z3_analyze_minimize(args: dict[str, Any]) -> dict[str, Any]:
                    cons = args.get("constraints") or []
                    if not isinstance(cons, list):
                        return {"error": "constraints must be a list"}
                    analysis = await z3v.analyze_constraints(cons)
                    minimized = await z3v.minimize_constraints(cons, analysis)
                    return {"analysis": analysis, "minimized": minimized}

                ability_reg.register_tool(
                    contract={
                        "tool_id": "z3_analyze_minimize",
                        "description": "Analyze constraints for complexity and propose a minimized essential set",
                        "input_schema": {
                            "type": "object",
                            "required": ["constraints"],
                            "properties": {"constraints": {"type": "array"}},
                        },
                        "output_schema": {"type": "object"},
                    },
                    executor=_z3_analyze_minimize,
                )

                async def _z3_verify(args: dict[str, Any]) -> dict[str, Any]:
                    cons = args.get("constraints") or []
                    if not isinstance(cons, list):
                        return {"error": "constraints must be a list"}
                    timeout_s = args.get("timeout_s")
                    res = await z3v.verify(
                        cons, timeout_s=int(timeout_s) if timeout_s else None
                    )
                    return res

                ability_reg.register_tool(
                    contract={
                        "tool_id": "z3_verify",
                        "description": "Verify constraints with z3 (eq/ineq relations on Int symbols)",
                        "input_schema": {
                            "type": "object",
                            "required": ["constraints"],
                            "properties": {
                                "constraints": {"type": "array"},
                                "timeout_s": {"type": "integer"},
                            },
                        },
                        "output_schema": {"type": "object"},
                    },
                    executor=_z3_verify,
                )

                print("? DEBUG: Z3 verifier abilities registered")
            else:
                print(
                    "? DEBUG: Z3 verifier disabled (set ALITA_ENABLE_Z3=true to enable)"
                )
        except Exception as e:
            print(f"? DEBUG: Failed to register Z3 verifier: {e}")
            import traceback

            traceback.print_exc()

        # Register REUG-LADDER bridging pipeline (planning + discovery + TDD codegen)
        try:
            ability_reg = app.state.ability_registry  # type: ignore[attr-defined]

            async def _ladder_reug_generate(args: dict[str, Any]) -> dict[str, Any]:
                """Integrate planning + discovery + TDD codegen in one call.

                Args:
                  goal: description of the feature to implement
                  file_path: where to write the code
                  language: target language (default python)
                  use_github_discovery: bool (default true)
                  test_first: bool (default true)
                  constraints: optional list of simple z3 constraints
                """
                goal = (args.get("goal") or args.get("prompt") or "").strip()
                if not goal:
                    return {"error": "missing goal"}
                file_path = (args.get("file_path") or "").strip()
                language = (args.get("language") or "python").strip()
                use_discovery = bool(args.get("use_github_discovery", True))
                test_first = bool(args.get("test_first", True))
                constraints = args.get("constraints") or []
                context: dict[str, Any] = args.get("context") or {}

                result: dict[str, Any] = {"goal": goal}

                # 1) Try to evolve the prompt
                try:
                    evo = await ability_reg.execute(  # type: ignore
                        "prompt_evolve", {"prompt": goal, "variants": 4}
                    )
                    best_prompt = evo.get("best_prompt") or goal
                except Exception:
                    best_prompt = goal
                result["prompt"] = best_prompt

                # 2) Optional GitHub discovery to augment spec
                refs: list[dict[str, Any]] = []
                if use_discovery:
                    try:
                        q = f"{goal} language:{language}"
                        gh = await ability_reg.execute(  # type: ignore
                            "github_search_code", {"q": q, "per_page": 5}
                        )
                        refs = (gh.get("items") or [])[:5]
                    except Exception:
                        refs = []
                result["github_refs"] = refs

                # 3) Optional constraints verification (if enabled)
                z3_summary: dict[str, Any] | None = None
                if constraints and os.getenv("ALITA_ENABLE_Z3", "false").lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }:
                    try:
                        analysis = await ability_reg.execute(  # type: ignore
                            "z3_analyze_minimize", {"constraints": constraints}
                        )
                        ver = await ability_reg.execute(  # type: ignore
                            "z3_verify",
                            {
                                "constraints": analysis.get("minimized") or constraints,
                                "timeout_s": 10,
                            },
                        )
                        z3_summary = {"analysis": analysis, "verify": ver}
                    except Exception as e:  # pragma: no cover
                        z3_summary = {"error": str(e)}
                if z3_summary is not None:
                    result["z3"] = z3_summary

                # 4) Assemble spec for code synth
                spec_lines = [best_prompt]
                if refs:
                    spec_lines.append("References:")
                    for r in refs[:3]:
                        spec_lines.append(f" - {r.get('html_url')}")
                spec = "\n".join(spec_lines)

                # 5) TDD code generation and write
                synth_args: dict[str, Any] = {
                    "language": language,
                    "spec": spec,
                    "file_path": file_path,
                    "test_first": test_first,
                    "consolidate_tests": True,
                }
                synth = await ability_reg.execute("code_synthesize_and_write", synth_args)  # type: ignore
                result["codegen"] = synth
                return result

            ability_reg.register_tool(
                contract={
                    "tool_id": "ladder_reug_generate",
                    "description": "Plan + discover + TDD codegen pipeline. Uses prompt evolution, optional GitHub search, and writes code/tests.",
                    "input_schema": {
                        "type": "object",
                        "required": ["goal", "file_path"],
                        "properties": {
                            "goal": {"type": "string"},
                            "file_path": {"type": "string"},
                            "language": {"type": "string", "default": "python"},
                            "use_github_discovery": {
                                "type": "boolean",
                                "default": True,
                            },
                            "test_first": {"type": "boolean", "default": True},
                            "constraints": {"type": "array"},
                            "context": {"type": "object"},
                        },
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_ladder_reug_generate,
            )

            print("? DEBUG: LADDER–REUG bridging ability registered")
        except Exception as e:
            print(f"? DEBUG: Failed to register ladder bridge: {e}")
            import traceback

            traceback.print_exc()

        # Register Unified Orchestration ability + endpoint
        try:
            ability_reg = app.state.ability_registry  # type: ignore[attr-defined]

            from src.orchestration.unified_orchestrator import (
                UnifiedOrchestrator,
                UnifiedRunConfig,
            )

            orchestrator_instance = UnifiedOrchestrator(
                ability_reg, app.state.event_bus  # type: ignore[attr-defined]
            )

            async def _unified_execute(args: dict[str, Any]) -> dict[str, Any]:
                prompt = (args.get("prompt") or args.get("goal") or "").strip()
                if not prompt:
                    return {"error": "missing prompt"}
                cfg = UnifiedRunConfig.from_args(prompt, args)
                # Run non-streaming for tool invocation (aggregated result)
                return await orchestrator_instance.run(cfg)

            ability_reg.register_tool(  # type: ignore
                contract={
                    "tool_id": "unified_execute",
                    "description": (
                        "Run unified orchestration pipeline "
                        "(spec/plan/consensus/code/validate/score)"
                    ),
                    "input_schema": {
                        "type": "object",
                        "required": ["prompt"],
                        "properties": {
                            "prompt": {"type": "string"},
                            "run_id": {"type": "string"},
                            "session_id": {"type": "string"},
                            "file_path": {"type": "string"},
                            "language": {
                                "type": "string",
                                "default": "python",
                            },
                            "enable_specification": {
                                "type": "boolean",
                                "default": False,
                            },
                            "enable_planning": {
                                "type": "boolean",
                                "default": True,
                            },
                            "enable_tasks": {
                                "type": "boolean",
                                "default": False,
                            },
                            "enable_consensus": {
                                "type": "boolean",
                                "default": True,
                            },
                            "enable_code_generation": {
                                "type": "boolean",
                                "default": False,
                            },
                            "enable_validation": {
                                "type": "boolean",
                                "default": False,
                            },
                            "enable_scoring": {
                                "type": "boolean",
                                "default": False,
                            },
                            "test_first": {"type": "boolean", "default": True},
                            "timeout_s": {"type": "integer", "default": 120},
                        },
                    },
                    "output_schema": {"type": "object"},
                },
                executor=_unified_execute,
            )

            # Add streaming endpoint (best-effort) under existing app
            from fastapi import APIRouter  # type: ignore
            from fastapi.responses import StreamingResponse  # type: ignore

            unified_router = APIRouter(  # type: ignore
                prefix="/v1/unified", tags=["unified"]
            )

            def _sse_pack(event_type: str, payload: dict[str, Any]) -> str:
                data = json.dumps(payload, ensure_ascii=False)
                return f"event: {event_type}\ndata: {data}\n\n"

            @unified_router.post("/stream")  # type: ignore
            async def unified_stream_post(request: Request):  # type: ignore
                body = await request.json()
                prompt = (body.get("prompt") or body.get("goal") or "").strip()
                if not prompt:
                    from fastapi.responses import JSONResponse  # type: ignore

                    return JSONResponse(  # type: ignore
                        {"error": "missing prompt"}, status_code=400
                    )
                cfg = UnifiedRunConfig.from_args(prompt, body)

                async def gen():  # type: ignore
                    async for ev in orchestrator_instance.run_stream(cfg):
                        et = ev.get("type", "message")
                        mapping = {
                            "UnifiedRunStarted": "start",
                            "UnifiedStageStarted": "stage_start",
                            "UnifiedStageSucceeded": "stage_result",
                            "UnifiedStageFailed": "stage_error",
                            "UnifiedRunCompleted": "done",
                        }
                        yield _sse_pack(mapping.get(et, "message"), ev)

                return StreamingResponse(
                    gen(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache, no-transform",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )

            @unified_router.get("/stream")  # type: ignore
            async def unified_stream_get(request: Request):  # type: ignore
                params = request.query_params  # type: ignore[attr-defined]
                prompt = (params.get("q") or params.get("prompt") or "").strip()
                if not prompt:
                    from fastapi.responses import JSONResponse  # type: ignore

                    return JSONResponse(  # type: ignore
                        {"error": "missing prompt"}, status_code=400
                    )
                cfg_args = dict(params)
                cfg = UnifiedRunConfig.from_args(prompt, cfg_args)

                async def gen():  # type: ignore
                    async for ev in orchestrator_instance.run_stream(cfg):
                        et = ev.get("type", "message")
                        mapping = {
                            "UnifiedRunStarted": "start",
                            "UnifiedStageStarted": "stage_start",
                            "UnifiedStageSucceeded": "stage_result",
                            "UnifiedStageFailed": "stage_error",
                            "UnifiedRunCompleted": "done",
                        }
                        yield _sse_pack(mapping.get(et, "message"), ev)

                return StreamingResponse(
                    gen(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache, no-transform",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )

            app.include_router(unified_router)  # Already has prefix="/v1/unified"
            print("? DEBUG: Unified orchestration ability + endpoints registered")
        except Exception as e:  # noqa: BLE001
            print(f"? DEBUG: Failed to register unified orchestration: {e}")
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
        # Constitutional Gateway router under prefix
        with contextlib.suppress(Exception):
            app.include_router(constitutional_router, prefix=prefix)  # type: ignore
        # SDD Framework router under prefix
        with contextlib.suppress(Exception):
            from src.sdd.router import create_sdd_router

            sdd_router = create_sdd_router()
            app.include_router(sdd_router, prefix=prefix)  # type: ignore

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
        # Constitutional Gateway router without prefix
        with contextlib.suppress(Exception):
            app.include_router(constitutional_router)  # type: ignore
        # SDD Framework router without prefix
        with contextlib.suppress(Exception):
            from src.sdd.router import create_sdd_router

            sdd_router = create_sdd_router()
            app.include_router(sdd_router)  # type: ignore

        # Register reasoning endpoints (consensus/deepconf/mangle) if available
        with suppress(Exception):
            from src.reasoning.endpoints import (
                register_reasoning_endpoints,  # type: ignore
            )

            register_reasoning_endpoints(app)  # type: ignore[arg-type]

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

    # ---------------- SDD (Spec-Driven Development) Endpoints ---------------- #
    class SDDSpecificationRequest(BaseModel):  # type: ignore[misc, valid-type]
        spec_id: str  # External ID from IDE
        title: str
        description: str
        requirements: list[str]
        constraints: list[str]

    class SDDPlanRequest(BaseModel):  # type: ignore[misc, valid-type]
        plan_id: str
        specification_id: str
        tech_stack: list[str]
        architecture: str
        dependencies: list[str]

    class SDDTasksRequest(BaseModel):  # type: ignore[misc, valid-type]
        plan_id: str
        tasks: list[dict]

    @app.post("/sdd/specify")  # type: ignore
    async def sdd_specify(
        req: SDDSpecificationRequest,  # type: ignore
        _auth: None = Depends(require_api_key),  # type: ignore
        _rl: None = Depends(enforce_rate_limit),  # type: ignore
    ) -> dict[str, object]:  # type: ignore
        payload = req.model_dump()
        with contextlib.suppress(Exception):
            evt = create_event("sdd_specify", **payload)
            if hasattr(app.state.event_bus, "publish"):  # type: ignore
                await app.state.event_bus.publish(evt.model_dump())  # type: ignore
            else:
                await app.state.event_bus.emit(evt.model_dump())  # type: ignore
        return {"status": "specification_processed", "spec_id": req.spec_id}

    @app.post("/sdd/plan")  # type: ignore
    async def sdd_plan(
        req: SDDPlanRequest,  # type: ignore
        _auth: None = Depends(require_api_key),  # type: ignore
        _rl: None = Depends(enforce_rate_limit),  # type: ignore
    ) -> dict[str, object]:  # type: ignore
        payload = req.model_dump()
        with contextlib.suppress(Exception):
            evt = create_event("sdd_plan", **payload)
            if hasattr(app.state.event_bus, "publish"):  # type: ignore
                await app.state.event_bus.publish(evt.model_dump())  # type: ignore
            else:
                await app.state.event_bus.emit(evt.model_dump())  # type: ignore
        return {"status": "plan_processed", "plan_id": req.plan_id}

    @app.post("/sdd/tasks")  # type: ignore
    async def sdd_tasks(
        req: SDDTasksRequest,  # type: ignore
        _auth: None = Depends(require_api_key),  # type: ignore
        _rl: None = Depends(enforce_rate_limit),  # type: ignore
    ) -> dict[str, object]:  # type: ignore
        payload = req.model_dump()
        with contextlib.suppress(Exception):
            evt = create_event("sdd_tasks", **payload)
            if hasattr(app.state.event_bus, "publish"):  # type: ignore
                await app.state.event_bus.publish(evt.model_dump())  # type: ignore
            else:
                await app.state.event_bus.emit(evt.model_dump())  # type: ignore
        return {
            "status": "tasks_processed",
            "plan_id": req.plan_id,
            "count": len(req.tasks),
        }

    # ---------------- Reasoning (DeepCode) Endpoint ---------------- #
    class CodeAnalysisRequest(BaseModel):  # type: ignore[misc, valid-type]
        code: str
        context_before: list[str] = []
        context_after: list[str] = []
        language: str
        file_name: str
        consensus_method: str = "ensemble_ranking"
        include_alternatives: bool = True
        confidence_threshold: float = 0.7

    class ReasoningStep(BaseModel):  # type: ignore[misc, valid-type]
        step: str
        confidence: float = 0.5
        evidence: list[str] = []
        alternatives: list[str] = []

    class CodeAnalysisResponse(BaseModel):  # type: ignore[misc, valid-type]
        confidence: float
        reasoning_steps: list[ReasoningStep]
        alternatives: list[str] = []
        patterns: list[dict[str, object]] = []
        insights: list[dict[str, object]] = []
        risk_assessment: dict[str, object] = {}
        suggested_improvements: list[str] = []

    def _semantic_insights(
        code: str, language: str, before: list[str], after: list[str]
    ) -> list[dict[str, object]]:
        insights: list[dict[str, object]] = []
        try:
            if (
                "class " in code or code.strip().startswith("class ")
            ) and language.lower() in {
                "python",
                "typescript",
                "javascript",
                "java",
                "csharp",
            }:
                insights.append(
                    {
                        "type": "architecture",
                        "insight": "Class definition detected",
                        "severity": "medium",
                        "suggestion": "Ensure single responsibility and proper encapsulation",
                    }
                )
            # Performance (very lightweight)
            if "for" in code and (" in " in code or ":" in code):
                insights.append(
                    {
                        "type": "performance",
                        "insight": "Loop detected - consider efficiency",
                        "severity": "low",
                        "suggestion": "Consider comprehension/vectorization where applicable",
                    }
                )
            # Security quick scan
            low = code.lower()
            if any(tok in low for tok in ["sql", "exec", "eval", "input("]):
                insights.append(
                    {
                        "type": "security",
                        "insight": "Potential security-sensitive call detected",
                        "severity": "high",
                        "suggestion": "Validate and sanitize all inputs; avoid eval/exec where possible",
                    }
                )
        except Exception:
            pass
        return insights

    def _risk_assessment(
        code: str, language: str, base_conf: float
    ) -> dict[str, object]:
        # Heuristic scores; replace with DeepConf calibration if available
        try:
            lines = code.count("\n") + 1
            words = len(code.split())
            complexity = min(1.0, (lines * max(1, words // max(1, lines))) / 500)
            coupling = code.count("import ") + code.count("from ")
            coupling_score = min(1.0, coupling / 10)
            overall = max(
                0.0,
                min(1.0, base_conf * (1 - (complexity * 0.5 + coupling_score * 0.3))),
            )
            return {
                "overall": overall,
                "factors": {
                    "complexity": complexity,
                    "coupling": coupling_score,
                    "testability": base_conf,
                    "security": 1.0 if "eval(" not in code.lower() else 0.3,
                },
                "language": language,
            }
        except Exception:
            return {"overall": base_conf, "factors": {}}

    @app.post("/reasoning/analyze-code", response_model=CodeAnalysisResponse)  # type: ignore
    async def analyze_code_with_reasoning(
        req: CodeAnalysisRequest,  # type: ignore
        _auth: None = Depends(require_api_key),  # type: ignore
        _rl: None = Depends(enforce_rate_limit),  # type: ignore
    ) -> CodeAnalysisResponse:  # type: ignore
        """Analyze code using consensus + lightweight semantic/risk heuristics.

        This endpoint composes a prompt and calls the in-process ability
        "deepconf_consensus" when available, falling back to a heuristic.
        """
        ability_registry = getattr(app.state, "ability_registry", None)  # type: ignore[attr-defined]
        consensus: dict[str, object] = {
            "consensus_text": "",
            "consensus_confidence": 0.5,
            "individual_responses": [],
            "confidence_scores": [],
        }
        if ability_registry and getattr(ability_registry, "knows", lambda *_: False)(
            "deepconf_consensus"
        ):
            prompt = (
                "Analyze this code for patterns, improvements, and reasoning.\n\n"
                f"Language: {req.language}\nFile: {req.file_name}\n"
                f"ContextBefore(last5): {req.context_before[-5:]}\nContextAfter(next5): {req.context_after[:5]}\n\n"
                f"Code:\n{req.code[:4000]}\n"
                "Return confidence, reasoning, alternatives, patterns, and risks."
            )
            args = {
                "prompt": prompt,
                "num_samples": 5,
                "temperature": 0.7,
                "max_tokens": 512,
                "method": req.consensus_method or "weighted_vote",
                "confidence_threshold": req.confidence_threshold,
                "temperature_range": 0.2,
            }
            try:
                consensus = await ability_registry.execute("deepconf_consensus", args)  # type: ignore
            except Exception as e:  # noqa: BLE001
                # Fallback remains in place
                with contextlib.suppress(Exception):
                    await app.state.event_bus.emit(
                        {  # type: ignore
                            "type": "AbilityFailed",
                            "tool": "deepconf_consensus",
                            "error": str(e),
                        }
                    )

        conf = float(consensus.get("consensus_confidence", 0.5) or 0.5)
        responses = consensus.get("individual_responses", [])
        if not isinstance(responses, list):
            responses = []
        steps: list[ReasoningStep] = []
        for i, resp in enumerate(responses[:5]):
            steps.append(
                ReasoningStep(
                    step=f"Analysis {i+1}: {str(resp)[:400]}", confidence=conf
                )
            )

        insights = _semantic_insights(
            req.code, req.language, req.context_before, req.context_after
        )
        risk = _risk_assessment(req.code, req.language, conf)

        # Simple pattern placeholder derived from insights
        patterns: list[dict[str, object]] = []
        for ins in insights:
            patterns.append(
                {
                    "pattern": str(ins.get("type")),
                    "confidence": 0.6,
                    "recommendation": str(ins.get("suggestion", "")),
                    "examples": [],
                }
            )

        # Suggested improvements: top 2 alternatives from steps if any
        improvements: list[str] = []
        for s in steps:
            improvements.extend((s.alternatives or [])[:1])
        improvements = improvements[:5]

        return CodeAnalysisResponse(
            confidence=conf,
            reasoning_steps=steps,
            alternatives=(
                list(consensus.get("individual_responses", [])[:3])
                if isinstance(consensus.get("individual_responses", []), list)
                else []
            ),
            patterns=patterns,
            insights=insights,
            risk_assessment=risk,
            suggested_improvements=improvements,
        )

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
