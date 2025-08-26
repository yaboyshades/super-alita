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
import sys
from collections.abc import AsyncGenerator, Callable
from logging.config import dictConfig
from pathlib import Path
from typing import Any
from uuid import uuid4

# Add conditional imports for FastAPI dependencies
try:
    import uvicorn
    from fastapi import APIRouter, FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    # Create stub classes when FastAPI is not available
    FastAPI = None  # type: ignore
    Request = None  # type: ignore
    APIRouter = None  # type: ignore
    CORSMiddleware = None  # type: ignore
    JSONResponse = None  # type: ignore
    StreamingResponse = None  # type: ignore
    uvicorn = None  # type: ignore
    FASTAPI_AVAILABLE = False


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


# --- Resolve reug_runtime from local src if not installed ---
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# REUG runtime routers (streaming agent + toolbox)
try:
    from reug_runtime.config import SETTINGS
    from reug_runtime.router import router as agent_router
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
        async def chat_stream(_: Request) -> StreamingResponse:  # type: ignore
            async def gen() -> AsyncGenerator[str, None]:
                yield "Thinking... "
                yield '<final_answer>{"content":"hello","citations":[]}</final_answer>'

            return StreamingResponse(gen(), media_type="text/plain")  # type: ignore

        tools_router = APIRouter(prefix="/tools", tags=["tools"])  # type: ignore

        @tools_router.get("/health")  # type: ignore
        async def tools_health() -> dict[str, str]:
            return {"status": "ok"}

        # Create minimal settings
        SETTINGS = type("Settings", (), {"api_prefix": ""})()  # type: ignore


# --- Event bus (JSONL fallback + optional Redis) ---
from reug_runtime.event_bus import (
    BaseEventBus,
    FileEventBus,
    make_event_bus,
)
from reug_runtime.llm_client import LLMClient, get_llm_client
from src.core.events import create_event


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
        # Seed with a friendly "echo" tool
        self._known: set[str] = {"echo"}
        self._contracts: dict[str, dict[str, Any]] = {
            "echo": {
                "tool_id": "echo",
                "description": "Echo back the provided payload",
                "input_schema": {
                    "type": "object",
                    "properties": {"payload": {"type": "string"}},
                },
                "output_schema": {"type": "object"},
            }
        }

    def get_available_tools_schema(self) -> list[dict[str, Any]]:
        return list(self._contracts.values())

    def knows(self, tool_name: str) -> bool:
        return tool_name in self._known

    def validate_args(self, tool_name: str, args: dict[str, Any]) -> bool:
        # Simple: require "payload" string for echo; otherwise permissive
        if tool_name == "echo":
            return isinstance(args.get("payload"), str)
        return self.knows(tool_name)

    async def health_check(self, _: dict[str, Any]) -> bool:
        # In real setups, ping MCP, SDK, HTTP endpoint, etc.
        return True

    async def register(self, contract: dict[str, Any]) -> None:
        tid = contract["tool_id"]
        self._contracts[tid] = contract
        self._known.add(tid)

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        # Implement your actual bindings here (MCP, HTTP APIs, Python functions).
        if tool_name == "echo":
            return {"echo": args.get("payload", "")}
        # Fallback generic
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


# --- FastAPI factory ---
def create_app(*, event_bus: BaseEventBus | None = None) -> Any:
    """Create FastAPI app or return None if FastAPI not available."""
    if not FASTAPI_AVAILABLE:
        print(
            "[ERROR] FastAPI not available - please install: pip install fastapi uvicorn"
        )
        return None

    _configure_logging()
    logger = logging.getLogger()
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

    # Initialize and start DeepCode plugins (bridge + orchestrator)
    with contextlib.suppress(Exception):
        from src.plugins.deepcode_generator_plugin import (
            DeepCodeGeneratorBridgePlugin,
        )
        from src.plugins.deepcode_orchestrator_plugin import (
            DeepCodeOrchestratorPlugin,
        )

        app.state.plugins = []  # type: ignore

        async def _start_plugins() -> None:
            gen = DeepCodeGeneratorBridgePlugin()
            orch = DeepCodeOrchestratorPlugin()
            await gen.setup(app.state.event_bus, store=None, config={})  # type: ignore
            await orch.setup(app.state.event_bus, store=None, config={})  # type: ignore
            await gen.start()
            await orch.start()
            app.state.plugins = [gen, orch]  # type: ignore

        # schedule startup after app is ready
        @app.on_event("startup")  # type: ignore
        async def _start_dc() -> None:
            await _start_plugins()

    # Mount routers
    prefix = SETTINGS.api_prefix
    if prefix and prefix != "/":
        if not prefix.startswith("/"):
            prefix = f"/{prefix}"
        prefix = prefix.rstrip("/")
        app.include_router(agent_router, prefix=prefix)  # type: ignore
        app.include_router(tools_router, prefix=prefix)  # type: ignore

        # Automatic message optimization middleware (HTTP level)
        @app.middleware("http")  # type: ignore
        async def _optimize_incoming(
            request: Request,
            call_next: Callable,  # type: ignore
        ) -> Any:
            with contextlib.suppress(Exception):
                # Only process JSON chat route
                if (
                    request.headers.get("content-type", "").startswith(  # type: ignore
                        "application/json"
                    )
                    and "/chat/stream" in request.url.path  # type: ignore
                ):
                    with contextlib.suppress(Exception):
                        from reug_runtime.config import (
                            SETTINGS as RT_SETTINGS,  # type: ignore
                        )
                        from reug_runtime.message_mw import (  # type: ignore
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

    @app.on_event("startup")  # type: ignore
    async def _startup() -> None:
        corr = str(uuid4())
        logger.info("runtime startup")
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

    # DeepCode trigger endpoint (fire-and-forget)
    @app.post("/deepcode/request")  # type: ignore
    async def deepcode_request(req: Request) -> dict[str, Any]:  # type: ignore
        body = await req.json()  # type: ignore
        payload: dict[str, Any] = {
            "event_type": "deepcode_request",
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

    # Just start the ASGI server; REUG handles single-turn streaming internally
    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)  # type: ignore
