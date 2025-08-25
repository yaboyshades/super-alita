from __future__ import annotations

"""Minimal streaming router for the REUG runtime.

This router implements a simple single-turn protocol compatible with
`MockLLMClient` and similar providers that emit tagged blocks:

  - <tool_call>{"tool":"name","args":{...}}</tool_call>
  - <tool_result tool="name">{...}</tool_result>
  - <final_answer>{"content":"...","citations":[]}</final_answer>

It executes tool calls via `app.state.ability_registry` and streams text
chunks through to the client. This keeps the agent functional while
conflicts are resolved or provider-specific logic evolves.
"""

import asyncio
import json
import re
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from .config import SETTINGS


router = APIRouter(prefix="/v1", tags=["agent"])


class _Parser:
    pattern = re.compile(r"<(\w+)([^>]*)>(\{.*?\})</\1>", re.DOTALL)

    def __init__(self) -> None:
        self.buffer = ""

    def feed(self, chunk: str) -> None:
        self.buffer += chunk

    def _extract(self, tag: str) -> tuple[dict[str, Any], str] | None:
        for m in self.pattern.finditer(self.buffer):
            name, attrs, payload = m.group(1), m.group(2), m.group(3)
            if name != tag:
                continue
            raw = m.group(0)
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = {"content": payload}
            self.buffer = self.buffer.replace(raw, "", 1)
            return data, attrs
        return None

    def take_tool_call(self) -> dict[str, Any] | None:
        hit = self._extract("tool_call")
        return hit[0] if hit else None

    def take_final(self) -> dict[str, Any] | None:
        hit = self._extract("final_answer")
        return hit[0] if hit else None


async def _stream_once(
    model: Any, messages: list[dict[str, str]]
) -> AsyncGenerator[str, None]:
    async for chunk in model.stream_chat(messages, timeout=SETTINGS.model_stream_timeout_s):
        text = chunk.get("content", "")
        if text:
            yield text


async def execute_turn(
    user_msg: str,
    session_id: str,
    event_bus: Any,
    registry: Any,
    kg: Any,
    model: Any,
) -> AsyncGenerator[str, None]:
    parser = _Parser()
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "Use tools when helpful. End with <final_answer>{...}</final_answer>.",
        },
        {"role": "user", "content": user_msg},
    ]
    cycles = 0
    while cycles < SETTINGS.max_tool_calls:
        cycles += 1
        tool_called = False
        async for text in _stream_once(model, messages):
            yield text
            parser.feed(text)
            call = parser.take_tool_call()
            if call:
                tool = call.get("tool", "")
                args = call.get("args", {})
                try:
                    result = await asyncio.wait_for(
                        registry.execute(tool, args), timeout=SETTINGS.tool_timeout_s
                    )
                except Exception as e:
                    yield f'<tool_error tool="{tool}">{{"error":{json.dumps(str(e))}}}</tool_error>'
                    break
                block = f'<tool_result tool="{tool}">{json.dumps(result)}</tool_result>'
                messages.append({"role": "assistant", "content": block})
                yield block
                tool_called = True
        if parser.take_final():
            return
        if not tool_called:
            break
    if not parser.take_final():
        payload = {"content": "done", "citations": []}
        yield f"<final_answer>{json.dumps(payload)}</final_answer>"


@router.post("/chat/stream")
async def chat_stream(request: Request):
    body = await request.json()
    user_msg = body.get("message", "")
    session_id = body.get("session_id", "default")
    gen = execute_turn(
        user_msg,
        session_id,
        request.app.state.event_bus,
        request.app.state.ability_registry,
        request.app.state.kg,
        request.app.state.llm_model,
    )
    return StreamingResponse(gen, media_type="text/plain")
