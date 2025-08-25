#!/usr/bin/env python3
"""FastAPI entrypoint for the REUG runtime."""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
import logging
from logging.config import dictConfig
import json
from uuid import uuid4

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware


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
    try:
        import hashlib

        h = hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()
        return h[:16]
    except Exception:
        return "na"

# --- Resolve reug_runtime from local src if not installed ---
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# REUG runtime routers (streaming agent + toolbox)
try:
    from reug_runtime.router import router as agent_router
    from reug_runtime.router_tools import tools as tools_router
    from reug_runtime.kg import create_kg_from_env
except Exception as e:  # pragma: no cover
    print("[WARN] reug_runtime not installed; make sure it’s on PYTHONPATH.", e)
    raise


# --- Event bus (JSONL fallback + optional Redis) ---
from reug_runtime.event_bus import (
    FileEventBus,
    RedisEventBus,
    make_event_bus,
)  # noqa: F401


# --- Ability registry (minimal adapter; replace with your real one) ---
class SimpleAbilityRegistry:
    """
    Minimal, schema-friendly registry:
      - knows(): does this tool exist?
      - validate_args(): shallow "type-ish" validation
      - register(): dynamic tool creation (contract-first)
      - execute(): your dispatch to MCP / SDK / code
    """

    def __init__(self):
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
        # Simple: require "payload" string for echo; otherwise permissive (router can enforce)
        if tool_name == "echo":
            return isinstance(args.get("payload"), str)
        return self.knows(tool_name)

    async def health_check(self, contract: dict[str, Any]) -> bool:
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


# --- LLM client: choose provider by available key (Gemini > OpenAI > Claude) ---
class LLMClient:
    def __init__(self):
        self._provider = None
        self._model = None

        gemini = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic = os.getenv("ANTHROPIC_API_KEY")

        if gemini:
            self._provider = "gemini"
            # lazy import; replace with your client if desired
            from collections.abc import AsyncGenerator as _T  # noqa: F401

        elif openai_key:
            self._provider = "openai"

        elif anthropic:
            self._provider = "anthropic"

        else:
            self._provider = "mock"  # dev fallback

    async def stream_chat(
        self, messages: list[dict[str, str]], timeout: float
    ) -> AsyncGenerator[dict[str, str], None]:
        """
        IMPORTANT: This *must* yield {"content": "..."} chunks.
        The REUG streaming router will parse <tool_call> / <tool_result> / <final_answer>.
        """
        # Minimal development behavior:
        # If no tool_result yet, ask for an echo tool call; otherwise finalize.
        has_result = any(
            m["role"] == "assistant" and "<tool_result" in m["content"]
            for m in messages
        )
        if not has_result:
            yield {"content": "Thinking... "}
            yield {
                "content": '<tool_call>{"tool":"echo","args":{"payload":"hello"}}</tool_call>'
            }
        else:
            yield {
                "content": '<final_answer>{"content":"done: hello","citations":[]}</final_answer>'
            }


# --- FastAPI factory ---
def create_app() -> FastAPI:
    _configure_logging()
    logger = logging.getLogger(__name__)
    app = FastAPI(title="REUG Runtime", version="0.2.0")

    # CORS (tweak as needed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    # Health for Dockerfile/compose
    @app.get("/healthz")
    async def health_check():
        return Response(status_code=200)

    # Alternative health endpoint
    @app.get("/health")
    async def health_check_alt():
        return {"status": "healthy", "service": "super-alita"}

    # Inject dependencies for the REUG router
    app.state.event_bus = make_event_bus()
    app.state.ability_registry = SimpleAbilityRegistry()
    app.state.kg = create_kg_from_env()
    app.state.llm_model = LLMClient()

    # Mount routers
    app.include_router(agent_router)  # /v1/chat/stream
    app.include_router(
        tools_router
    )  # /tools/* (toolbox – run tests, apply patches, etc.)

    @app.on_event("startup")
    async def _startup() -> None:
        corr = str(uuid4())
        logger.info("runtime startup")
        await app.state.event_bus.emit(
            {
                "type": "STATE_TRANSITION",
                "from": "BOOT",
                "to": "READY",
                "correlation_id": corr,
            }
        )
        await app.state.event_bus.emit(
            {
                "type": "TaskStarted",
                "correlation_id": corr,
                "goal": "startup",
                "user_msg_hash": _hash_json("startup"),
            }
        )

    return app


app = create_app()

# Optional CLI entry (e.g., python src/main.py --no-chat just validates startup)
if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--no-chat",
        action="store_true",
        help="Boot only; don’t open sockets beyond uvicorn",
    )
    ap.add_argument(
        "--reload",
        action="store_true",
        help="Reload server on code changes (dev mode)",
    )
    args = ap.parse_args()

    async def _dependency_health() -> dict[str, bool]:
        results: dict[str, bool] = {}
        try:
            await app.state.event_bus.emit({"event": "health_check"})
            results["event_bus"] = True
        except Exception:
            results["event_bus"] = False
        try:
            contract = app.state.ability_registry.get_available_tools_schema()[0]
            results["ability_registry"] = await app.state.ability_registry.health_check(
                contract
            )
        except Exception:
            results["ability_registry"] = False
        try:
            await app.state.kg.get_goal_for_session("health")
            results["kg"] = True
        except Exception:
            results["kg"] = False
        try:
            agen = app.state.llm_model.stream_chat([], timeout=1)
            await agen.__anext__()
            results["llm_model"] = True
        except Exception:
            results["llm_model"] = False
        return results

    if args.no_chat:
        import asyncio
        import json

        checks = asyncio.run(_dependency_health())
        print(json.dumps(checks))
        raise SystemExit(0)

    # Just start the ASGI server; REUG handles single-turn streaming internally
    uvicorn.run(
        "main:app", host=args.host, port=args.port, reload=args.reload
    )
