"""Minimal streaming router for the REUG runtime.

This router implements a simple single-turn protocol compatible with
MockLLMClient and similar providers that emit tagged blocks:

  - <tool_call>{"tool":"name","args":{...}}</tool_call>
  - <tool_result tool="name">{...}</tool_result>
  - <final_answer>{"content":"...","citations":[]}</final_answer>

It executes tool calls via pp.state.ability_registry and streams text
chunks through to the client. This keeps the agent functional while
conflicts are resolved or provider-specific logic evolves.

This module has been refactored to focus on FastAPI routing while
delegating core orchestration to the loop module and SSE streaming
to the streaming module.
"""

from __future__ import annotations

import os
from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .loop import Orchestrator, ReasoningResult, execute_turn, parse_tool_calls

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .loop import Orchestrator, execute_turn, parse_tool_calls
from .loop import execute_turn, parse_tool_calls

from .streaming import sse_transformer
from src.agents.communication import A2AProtocol

__all__ = [
    "router",
    "execute_turn",
    "parse_tool_calls",
    "Orchestrator",
    "ReasoningResult",
]

router = APIRouter(prefix="/v1", tags=["agent"])

__all__ = [
    "chat_stream",
    "chat_stream_get",
    "parse_tool_calls",
]


def _get_a2a_protocol(state: Any) -> A2AProtocol:
    protocol = getattr(state, "_a2a_protocol", None)
    if not isinstance(protocol, A2AProtocol):
        protocol = A2AProtocol(state.event_bus)
        setattr(state, "_a2a_protocol", protocol)
    return protocol


async def _route_agent_message(
    state: Any,
    message: str,
    session_id: str,
    *,
    priority: str = "medium",
    sender_id: str = "client",
    message_type: str = "user_message",
) -> None:
    if not message:
        return
    protocol = _get_a2a_protocol(state)
    await protocol.agent_to_agent(
        sender_id=sender_id,
        recipient_id="runtime.loop",
        message_type=message_type,
        payload={"content": message, "session_id": session_id},
        priority=priority,
        correlation_id=session_id,
        metadata={"claims": {"source": "router"}},
    )


@router.post("/chat/stream", response_model=None)
async def chat_stream(request: Request) -> StreamingResponse | JSONResponse:
    # Rate limit pre-check (optional)
    try:
        if os.getenv("ALITA_RATE_LIMIT_ENABLED", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            rl = getattr(request.app.state, "rate_limiter", None)
            if rl is not None:
                limit = int(os.getenv("ALITA_RATE_LIMIT", "60") or 60)
                window = int(os.getenv("ALITA_RATE_WINDOW", "60") or 60)
                hdr = request.headers.get(
                    os.getenv("ALITA_API_HEADER", "Authorization"), ""
                )
                tok = (
                    hdr[7:].strip()
                    if hdr.lower().startswith("bearer ")
                    else hdr.strip()
                )
                client_host = request.client.host if request.client else "unknown"
                ident = f"key:{tok[:8]}" if tok else f"ip:{client_host}"
                allowed, _ = await rl.is_allowed(ident, limit, window)
                if not allowed:

                    raise HTTPException(status_code=429, detail="rate_limited")

                    return JSONResponse(
                        status_code=429, content={"error": "rate_limited"}
                    )

    except Exception:
        pass
    raw_body = await request.json()
    body: dict[str, Any] = raw_body if isinstance(raw_body, dict) else {}
    user_msg = body.get("message", "")
    session_id = body.get("session_id", "default")

    state = cast(Any, request.app.state)

    priority = str(body.get("priority", "medium")) if isinstance(body, dict) else "medium"
    await _route_agent_message(
        state,
        user_msg,
        session_id,
        priority=priority,
        sender_id=str(body.get("sender_id", "client")) if isinstance(body, dict) else "client",
        message_type=str(body.get("message_type", "user_message")) if isinstance(body, dict) else "user_message",
    )

    event_gen = execute_turn(
        user_msg,
        session_id,
        state.event_bus,
        state.ability_registry,
        state.kg,
        state.llm_model,
    )

    sse_gen = sse_transformer(event_gen)

    return StreamingResponse(
        sse_gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/chat/stream", response_model=None)
async def chat_stream_get(request: Request) -> StreamingResponse:
    """
    GET variant to support browsers using EventSource.

    Accepts query params:
      - q or message
      - session or session_id
    """
    qp = request.query_params
    user_msg = qp.get("q") or qp.get("message") or ""
    session_id = qp.get("session") or qp.get("session_id") or "default"

    state = cast(Any, request.app.state)

    await _route_agent_message(
        state,
        user_msg,
        session_id,
        priority="medium",
        sender_id="client",
        message_type="user_message",
    )

    event_gen = execute_turn(
        user_msg,
        session_id,
        state.event_bus,
        state.ability_registry,
        state.kg,
        state.llm_model,
    )

    sse_gen = sse_transformer(event_gen)

    return StreamingResponse(
        sse_gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
