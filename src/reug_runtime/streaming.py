"""SSE streaming utilities for REUG runtime.

This module handles the conversion of internal event streams into
Server-Sent Events (SSE) format for HTTP streaming responses.
It provides heartbeat functionality and event name mapping.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator
from typing import Any


async def sse_transformer(
    event_generator: AsyncGenerator[dict[str, Any], None],
) -> AsyncGenerator[str, None]:
    """Transform internal events into SSE frames (with optional heartbeats)."""
    name_map = {
        "TaskStarted": "start",
        "LLMChunk": "content",
        "AbilityCalled": "tool_start",
        "AbilitySucceeded": "tool_result",
        "AbilityFailed": "tool_error",
        "TaskSucceeded": "done",
    }
    use_hb = os.getenv("ALITA_SSE_HEARTBEAT", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    hb_interval = int(os.getenv("ALITA_SSE_HEARTBEAT_INTERVAL", "15") or 15)

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def _pump() -> None:
        try:
            async for ev in event_generator:
                await queue.put(ev)
        finally:
            await queue.put(None)

    async def _pings() -> None:
        if not use_hb:
            return
        try:
            while True:
                await asyncio.sleep(max(1, hb_interval))
                await queue.put({"type": "__ping__"})
        except asyncio.CancelledError:
            return

    t1 = asyncio.create_task(_pump())
    t2 = asyncio.create_task(_pings()) if use_hb else None
    try:
        while True:
            ev = await queue.get()
            if ev is None:
                break
            et = ev.get("type", "message")
            if et == "__ping__":
                yield "event: ping\n"
                yield f"data: {json.dumps({'ts': int(time.time())})}\n\n"
                continue
            sse_name = name_map.get(et, "message")
            yield f"event: {sse_name}\n"
            if et == "LLMChunk":
                text = ev.get("data", {}).get("text", "")
                yield f"data: {json.dumps({'content': text})}\n\n"
            else:
                yield f"data: {json.dumps(ev)}\n\n"
    finally:
        t1.cancel()
        if t2:
            t2.cancel()
