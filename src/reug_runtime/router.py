from __future__ import annotations

"""Minimal streaming router for the REUG runtime.

This router implements a simple single-turn protocol compatible with
MockLLMClient and similar providers that emit tagged blocks:

  - <tool_call>{"tool":"name","args":{...}}</tool_call>
  - <tool_result tool="name">{...}</tool_result>
  - <final_answer>{"content":"...","citations":[]}</final_answer>

It executes tool calls via pp.state.ability_registry and streams text
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


class Orchestrator:
    def __init__(self, event_bus: Any, registry: Any, model: Any, correlation_id: str):
        self.event_bus = event_bus
        self.registry = registry
        self.model = model
        self.correlation_id = correlation_id

    async def _reasoning_step(
        self, messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        llm_response_content = ""
        tool_calls = []
        async for chunk in self.model.stream_chat(messages, tools=tool_schemas):
            if chunk.get("type") == "content":
                text = chunk.get("content", "")
                llm_response_content += text
                yield {"type": "LLMChunk", "data": {"text": text}}
            elif chunk.get("type") == "tool_calls":
                tool_calls.extend(chunk.get("tool_calls", []))
        # Store the return values for later retrieval
        self._last_reasoning_result = (llm_response_content, tool_calls)

    async def _acting_step(
        self, tool_calls: list[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        tool_messages = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_call_id = tool_call.id
            span_id = str(uuid.uuid4())

            ability_called_event = {
                "type": "AbilityCalled", "tool": tool_name,
                "correlation_id": self.correlation_id, "span_id": span_id
            }
            await self.event_bus.emit(ability_called_event)
            yield ability_called_event

            try:
                result = await asyncio.wait_for(
                    self.registry.execute(tool_name, tool_args),
                    timeout=SETTINGS.tool_timeout_s
                )
                ability_succeeded_event = {
                    "type": "AbilitySucceeded", "tool": tool_name,
                    "correlation_id": self.correlation_id, "span_id": span_id, "result": result
                }
                await self.event_bus.emit(ability_succeeded_event)
                yield ability_succeeded_event
                tool_messages.append({
                    "role": "tool", "tool_call_id": tool_call_id,
                    "name": tool_name, "content": json.dumps(result)
                })
            except Exception as e:
                ability_failed_event = {
                    "type": "AbilityFailed", "tool": tool_name,
                    "correlation_id": self.correlation_id, "span_id": span_id, "error": str(e)
                }
                await self.event_bus.emit(ability_failed_event)
                yield ability_failed_event
                tool_messages.append({
                    "role": "tool", "tool_call_id": tool_call_id, "name": tool_name,
                    "content": f'{{"error": "Tool execution failed: {e}"}}'
                })
        # Store the return values for later retrieval
        self._last_acting_result = tool_messages


async def execute_turn(
    user_msg: str, session_id: str, event_bus: Any, registry: Any, kg: Any, model: Any
) -> AsyncGenerator[dict[str, Any], None]:
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
    
    orchestrator = Orchestrator(event_bus, registry, model, correlation_id)

    start_event = {"type": "TaskStarted", "correlation_id": correlation_id, "goal": user_msg}
    await event_bus.emit(start_event)
    yield start_event

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when necessary."},
        {"role": "user", "content": user_msg},
    ]

    llm_response_content = ""
    for _ in range(SETTINGS.max_tool_calls):
        tool_schemas = registry.get_available_tools_schema()

        # Run reasoning step and get results
        async for event in orchestrator._reasoning_step(messages, tool_schemas):
            yield event
        llm_response_content, tool_calls = orchestrator._last_reasoning_result

        assistant_message = {"role": "assistant", "content": llm_response_content}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)

        if not tool_calls:
            break

        # Run acting step and get results
        async for event in orchestrator._acting_step(tool_calls):
            yield event
        tool_messages = orchestrator._last_acting_result
        messages.extend(tool_messages)

    final_answer = {"content": llm_response_content or "Task complete.", "citations": []}
    task_succeeded_event = {"type": "TaskSucceeded", "correlation_id": correlation_id, "data": final_answer}
    await event_bus.emit(task_succeeded_event)
    yield task_succeeded_event


async def sse_transformer(event_generator: AsyncGenerator[dict[str, Any], None]) -> AsyncGenerator[str, None]:
    async for event in event_generator:
        yield f"data: {json.dumps(event)}\n\n"


@router.post("/chat/stream")
async def chat_stream(request: Request):
    body = await request.json()
    user_msg = body.get("message", "")
    session_id = body.get("session_id", "default")

    event_gen = execute_turn(
        user_msg,
        session_id,
        request.app.state.event_bus,
        request.app.state.ability_registry,
        request.app.state.kg,
        request.app.state.llm_model,
    )

    sse_gen = sse_transformer(event_gen)

    return StreamingResponse(sse_gen, media_type="text/event-stream")
