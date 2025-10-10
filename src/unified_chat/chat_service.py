"""Unified chat service abstraction.

This module provides a thin orchestration layer that:
 - Manages lightweight in-memory chat sessions (id -> messages)
 - Provides helper to append user/assistant messages
 - Exposes a streaming generator that wraps the existing REUG execute_turn
 - Optionally performs a consensus refinement step (non-stream) after a turn

The goal is to supply a stable surface for API endpoints / UI without
adding heavy persistence requirements. If persistence is needed later,
an adapter can be injected implementing the same interface.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

try:  # Local import; avoid circulars at import time
    from src.reug_runtime.router import execute_turn
except Exception:  # pragma: no cover - fallback
    execute_turn = None  # type: ignore


@dataclass
class ChatMessage:
    role: str
    content: str
    meta: dict[str, Any] | None = None


@dataclass
class ChatSession:
    session_id: str
    messages: list[ChatMessage] = field(default_factory=list)

    def add(
        self, role: str, content: str, meta: dict[str, Any] | None = None
    ) -> None:
        self.messages.append(
            ChatMessage(role=role, content=content, meta=meta)
        )


class UnifiedChatService:
    """Manages chat sessions and bridges to the core streaming turn executor."""

    def __init__(self, app: Any):  # FastAPI app for accessing state objects
        self._app = app
        self._sessions: dict[str, ChatSession] = {}
        self._lock = asyncio.Lock()

    # ---- Session Management -------------------------------------------------
    async def create_session(
        self, session_id: str | None = None
    ) -> ChatSession:
        sid = session_id or str(uuid.uuid4())
        async with self._lock:
            if sid not in self._sessions:
                self._sessions[sid] = ChatSession(session_id=sid)
            return self._sessions[sid]

    async def get_session(self, session_id: str) -> ChatSession | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def list_sessions(self) -> list[str]:
        async with self._lock:
            return list(self._sessions.keys())

    # ---- Message Handling ---------------------------------------------------
    async def add_user_message(self, session_id: str, content: str) -> None:
        sess = await self.create_session(session_id)
        sess.add("user", content)

    async def add_assistant_message(
        self, session_id: str, content: str, meta: dict[str, Any] | None = None
    ) -> None:
        sess = await self.create_session(session_id)
        sess.add("assistant", content, meta)

    async def history(self, session_id: str) -> list[dict[str, Any]]:
        sess = await self.get_session(session_id)
        if not sess:
            return []
        return [
            {
                "role": m.role,
                "content": m.content,
                **({"meta": m.meta} if m.meta else {}),
            }
            for m in sess.messages
        ]

    # ---- Streaming Turn Bridge ---------------------------------------------
    async def stream_turn(
        self,
        session_id: str,
        user_message: str,
        *,
        use_consensus: bool = False,
        consensus_method: str = "weighted_vote",
        consensus_samples: int = 3,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a reasoning/acting turn.

        Yields event dicts; the caller (endpoint) is responsible for
        transforming to SSE or websocket frames.
        """
        await self.add_user_message(session_id, user_message)

        if execute_turn is None:  # safety
            yield {"type": "Error", "error": "execute_turn unavailable"}
            return

        event_bus = getattr(self._app.state, "event_bus", None)
        registry = getattr(self._app.state, "ability_registry", None)
        kg = getattr(self._app.state, "kg", None)
        model = getattr(self._app.state, "llm", None)

        final_answer_chunks: list[str] = []
        async for ev in execute_turn(  # type: ignore[misc]
            user_msg=user_message,
            session_id=session_id,
            event_bus=event_bus,
            registry=registry,
            kg=kg,
            model=model,
        ):
            et = ev.get("type")
            if et == "LLMChunk":
                txt = ev.get("data", {}).get("text", "")
                final_answer_chunks.append(txt)
            yield ev

        final_answer = "".join(final_answer_chunks).strip()
        if final_answer:
            await self.add_assistant_message(session_id, final_answer)

        # Optional consensus refinement (non-stream, appended as system note)
        if use_consensus:
            try:
                consensus_tool_id = "deepconf_consensus"
                # Basic guard: ensure tool registered
                if registry and getattr(registry, "knows", lambda *_: False)(
                    consensus_tool_id
                ):
                    result = await registry.execute(
                        consensus_tool_id,
                        {
                            "prompt": final_answer or user_message,
                            "method": consensus_method,
                            "num_samples": consensus_samples,
                        },
                    )
                    consensus_text = (
                        result.get("consensus_text")
                        or result.get("result")
                        or ""
                    )
                    await self.add_assistant_message(
                        session_id,
                        consensus_text,
                        meta={"consensus": True, "raw": result},
                    )
                    yield {
                        "type": "ConsensusResult",
                        "result": result,
                        "session_id": session_id,
                    }
            except Exception as e:  # noqa: BLE001
                yield {"type": "ConsensusError", "error": str(e)}

    # ---- Utility -----------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {"sessions": len(self._sessions)}


__all__ = [
    "UnifiedChatService",
    "ChatSession",
    "ChatMessage",
]
