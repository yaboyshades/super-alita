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
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.telemetry import build_copilot_context

from .config import SETTINGS
from .message_mw import MessageContext, apply_all

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
    """Run a single streaming turn and yield chunks to the client.

    The function keeps a minimal contract compatible with the tests in
    ``tests/runtime``.  It emits telemetry events around tool execution and
    preserves the `<tool_call>`/`<tool_result>`/`<final_answer>` streaming
    protocol.

    Args:
        user_msg: The raw user message for this turn.
        session_id: Session identifier.
        event_bus: Event bus used for telemetry emission.
        registry: Ability registry capable of executing tools.
        kg: Knowledge graph handle (unused, kept for future expansion).
        model: LLM-like client implementing ``stream_chat``.

    Yields:
        Text chunks to be streamed to the client.
    """

    correlation_id = f"{session_id}-{int(time.time()*1000)}"
    # Optional message optimization/amplification
    if SETTINGS.message_optimizer_enabled:
        try:
            # Lazy import to avoid side effects when disabled
            import src.plugins.message_amplifier_plugin  # noqa: F401
        except Exception:
            # If plugin import fails, continue with raw message
            pass
        optimized, steps = apply_all(user_msg, MessageContext(session_id=session_id))
        if SETTINGS.message_optimizer_emit_telemetry:
            await event_bus.emit(
                {
                    "type": "MessageOptimized",
                    "correlation_id": correlation_id,
                    "len_in": len(user_msg),
                    "len_out": len(optimized),
                    "steps": steps,
                }
            )
        # Enforce a soft cap if configured to prevent runaway messages
        if len(optimized) > SETTINGS.message_optimizer_max_len:
            optimized = optimized[: SETTINGS.message_optimizer_max_len]
        user_msg = optimized
    await event_bus.emit({"type": "TaskStarted", "correlation_id": correlation_id, "goal": user_msg})

    system_prompt = "Use tools when helpful. End with <final_answer>{...}</final_answer>."
    if SETTINGS.copilot_context:
        span_id = str(uuid.uuid4())
        await event_bus.emit(
            {
                "type": "AbilityCalled",
                "tool": "build_copilot_context",
                "correlation_id": correlation_id,
                "span_id": span_id,
            }
        )
        ctx = build_copilot_context(user_message=user_msg, session_id=session_id)
        await event_bus.emit(
            {
                "type": "AbilitySucceeded",
                "tool": "build_copilot_context",
                "correlation_id": correlation_id,
                "span_id": span_id,
            }
        )
        system_prompt = f"{ctx}\n{system_prompt}"

    parser = _Parser()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
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
                span_id = str(uuid.uuid4())
                await event_bus.emit(
                    {
                        "type": "AbilityCalled",
                        "tool": tool,
                        "correlation_id": correlation_id,
                        "span_id": span_id,
                    }
                )
                try:
                    result = await asyncio.wait_for(
                        registry.execute(tool, args), timeout=SETTINGS.tool_timeout_s
                    )
                except Exception as e:
                    await event_bus.emit(
                        {
                            "type": "AbilityFailed",
                            "tool": tool,
                            "correlation_id": correlation_id,
                            "span_id": span_id,
                            "error": str(e),
                        }
                    )
                    yield f'<tool_error tool="{tool}">{{"error":{json.dumps(str(e))}}}</tool_error>'
                    break
                await event_bus.emit(
                    {
                        "type": "AbilitySucceeded",
                        "tool": tool,
                        "correlation_id": correlation_id,
                        "span_id": span_id,
                    }
                )
                block = f'<tool_result tool="{tool}">{json.dumps(result)}</tool_result>'
                messages.append({"role": "assistant", "content": block})
                yield block
                tool_called = True
        final = parser.take_final()
        if final:
            await event_bus.emit({"type": "TaskSucceeded", "correlation_id": correlation_id})
            yield f"<final_answer>{json.dumps(final)}</final_answer>"
            return
        if not tool_called:
            break
    if not parser.take_final():
        payload = {"content": "done", "citations": []}
        await event_bus.emit(
            {"type": "TaskFailed", "correlation_id": correlation_id, "reason": "no_final_answer"}
        )
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
