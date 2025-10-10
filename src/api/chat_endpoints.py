"""Chat API endpoints (unified chat layer).

Routes under /api/chat:
 - POST /session       : create or return provided session id
 - GET  /history/{sid} : return message history
 - POST /message       : single-turn non-stream response
 - POST /stream        : Server-Sent Events streaming of a turn

Implementation notes:
 - SSE events encoded as: 'event: <Type>\n data: <json>\n\n'
 - Optional consensus refinement executed after base answer (non-stream path
     aggregates internally; stream path emits a ConsensusResult event).
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from src.unified_chat.chat_service import UnifiedChatService

router = APIRouter(prefix="/api/chat", tags=["chat"])


def get_chat_service(dep_request) -> UnifiedChatService:  # type: ignore[override]
    # FastAPI injects Request object; keep name generic to avoid import churn
    app = dep_request.app
    svc = getattr(app.state, "unified_chat_service", None)
    if svc is None:
        svc = UnifiedChatService(app)
        app.state.unified_chat_service = svc  # type: ignore[attr-defined]
    return svc


@router.post("/session")
async def create_session(
    payload: dict[str, Any],
    svc: UnifiedChatService = Depends(get_chat_service),  # noqa: B008
) -> Any:  # noqa: ANN401
    sess = await svc.create_session(payload.get("session_id"))
    return {"session_id": sess.session_id}


@router.get("/history/{session_id}")
async def get_history(
    session_id: str,
    svc: UnifiedChatService = Depends(get_chat_service),  # noqa: B008
) -> Any:  # noqa: ANN401
    hist = await svc.history(session_id)
    return {"session_id": session_id, "messages": hist}


@router.post("/message")
async def send_message(
    payload: dict[str, Any],
    svc: UnifiedChatService = Depends(get_chat_service),  # noqa: B008
) -> Any:  # noqa: ANN401
    session_id = (
        payload.get("session_id") or payload.get("session") or "default"
    )
    message = payload.get("message") or payload.get("text")
    if not isinstance(message, str) or not message.strip():  # invalid
        return JSONResponse(
            status_code=400, content={"error": "Missing message"}
        )
    use_consensus = bool(payload.get("consensus"))
    method = payload.get("consensus_method", "weighted_vote")
    samples = int(payload.get("consensus_samples", 3))

    collected: list[str] = []
    async for ev in svc.stream_turn(
        session_id,
        message,
        use_consensus=use_consensus,
        consensus_method=method,
        consensus_samples=samples,
    ):
        if ev.get("type") == "LLMChunk":
            collected.append(ev.get("data", {}).get("text", ""))
    return {"session_id": session_id, "response": "".join(collected).strip()}


@router.post("/stream")
async def stream_message(
    payload: dict[str, Any],
    svc: UnifiedChatService = Depends(get_chat_service),  # noqa: B008
) -> StreamingResponse:  # noqa: ANN401
    session_id = (
        payload.get("session_id") or payload.get("session") or "default"
    )
    message = payload.get("message") or payload.get("text")
    if not isinstance(message, str) or not message.strip():  # invalid
        return StreamingResponse(
            iter([_sse({"type": "Error", "error": "Missing message"})]),
            media_type="text/event-stream",
        )
    use_consensus = bool(payload.get("consensus"))
    method = payload.get("consensus_method", "weighted_vote")
    samples = int(payload.get("consensus_samples", 3))

    async def event_gen() -> AsyncGenerator[bytes, None]:
        try:
            async for ev in svc.stream_turn(
                session_id,
                message,
                use_consensus=use_consensus,
                consensus_method=method,
                consensus_samples=samples,
            ):
                yield _sse(ev)
        except Exception as e:  # noqa: BLE001
            yield _sse({"type": "Error", "error": str(e)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def _sse(ev: dict[str, Any]) -> bytes:
    et = ev.get("type", "event")
    data = json.dumps(ev, ensure_ascii=False)
    return (f"event: {et}\n" f"data: {data}\n\n").encode()


__all__ = ["router"]
