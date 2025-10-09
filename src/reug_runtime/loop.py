"""Core orchestration loop and turn execution logic.

This module contains the core components extracted from router.py:
- parse_tool_calls: Parse tool calls from streamed LLM text
- Orchestrator: Main orchestration class handling reasoning/acting steps
- execute_turn: Complete turn execution with streaming events

These components handle the core agent loop while being independent
of FastAPI routing and SSE streaming concerns.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from .config import SETTINGS
from .formatting import normalize_output_contract
from .message_mw import MessageContext, apply_all
from .tools.service import ToolCatalogService


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse serialized ``<tool_call>`` blocks from streamed text."""

    calls: list[dict[str, Any]] = []
    try:
        for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
            inner = match.group(1)
            payload = json.loads(inner)
            name = payload.get("tool")
            raw_args = payload.get("args", {})
            if not isinstance(name, str) or not name:
                continue
            if not isinstance(raw_args, dict):
                continue
            call: dict[str, Any] = {
                "id": str(uuid.uuid4()),
                "name": name,
                "function": {"name": name, "arguments": json.dumps(raw_args)},
            }
            calls.append(call)
    except Exception:
        # best-effort parsing only
        pass
    return calls


@dataclass(slots=True)
class ReasoningResult:
    """Snapshot of the latest reasoning output from the language model."""

    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class Orchestrator:
    def __init__(self, event_bus: Any, registry: Any, model: Any, correlation_id: str):
        self.event_bus = event_bus
        self.registry = registry
        self.model = model
        self.correlation_id = correlation_id
        self._last_reasoning_result = ReasoningResult()
        self._last_acting_result: list[dict[str, Any]] = []
        # Use shared tool catalog service for dynamic tool registration
        self._tool_service = ToolCatalogService()

    @staticmethod
    def _coerce_tool_call(raw_call: Any) -> dict[str, Any]:
        """Convert provider-specific tool call payloads into plain dicts."""

        if isinstance(raw_call, dict):
            return raw_call

        for attr in ("model_dump", "to_dict", "dict"):
            fn = getattr(raw_call, attr, None)
            if callable(fn):
                try:
                    data = fn()
                except TypeError:
                    continue
                if isinstance(data, dict):
                    return cast(dict[str, Any], data)

        result: dict[str, Any] = {}

        for attr in ("id", "name", "tool", "type"):
            value = getattr(raw_call, attr, None)
            if value is not None:
                result[attr] = value

        fn_obj = getattr(raw_call, "function", None)
        if fn_obj is not None:
            if isinstance(fn_obj, dict):
                result["function"] = fn_obj
            else:
                fn_dict: dict[str, Any] = {}
                for attr in ("name", "arguments"):
                    value = getattr(fn_obj, attr, None)
                    if value is not None:
                        fn_dict[attr] = value
                if fn_dict:
                    result["function"] = fn_dict

        arguments = getattr(raw_call, "arguments", None)
        if arguments is not None:
            result.setdefault("arguments", arguments)

        return result

    async def _reasoning_step(
        self, messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        llm_response_content = ""
        tool_calls: list[dict[str, Any]] = []

        # Call model.stream_chat with best-effort compatibility across providers
        async def _stream() -> AsyncGenerator[dict[str, Any], None]:
            try:
                # Preferred: pass tools and timeout if supported
                async for ch in self.model.stream_chat(
                    messages,
                    tools=tool_schemas,
                    timeout=SETTINGS.model_stream_timeout_s,
                ):
                    yield ch
                return
            except TypeError:
                pass
            try:
                # Fallback: pass only timeout
                async for ch in self.model.stream_chat(
                    messages,
                    timeout=SETTINGS.model_stream_timeout_s,
                ):
                    yield ch
                return
            except TypeError:
                pass
            # Last-resort: messages only
            async for ch in self.model.stream_chat(messages):
                yield ch

        async for chunk in _stream():
            if "content" in chunk:
                text = chunk.get("content", "")
                llm_response_content += text
                yield {"type": "LLMChunk", "data": {"text": text}}
            elif chunk.get("type") == "tool_calls":
                raw_tool_calls = chunk.get("tool_calls", [])
                if isinstance(raw_tool_calls, list):
                    for raw_call in raw_tool_calls:
                        normalized = self._coerce_tool_call(raw_call)
                        if normalized:
                            tool_calls.append(normalized)
        # Store the return values for later retrieval
        self._last_reasoning_result = ReasoningResult(
            text=llm_response_content, tool_calls=list(tool_calls)
        )

    async def _acting_step(
        self, tool_calls: list[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        tool_messages: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            # Support both OpenAI SDK objects and plain dicts
            if hasattr(tool_call, "function"):
                fn = tool_call.function
                tool_name = getattr(fn, "name", None)
                tool_args_raw = getattr(fn, "arguments", "{}")
                tool_call_id = getattr(tool_call, "id", str(uuid.uuid4()))
            elif isinstance(tool_call, dict):
                fn = tool_call.get("function", {})
                if isinstance(fn, dict):
                    tool_name = (
                        fn.get("name") or tool_call.get("name") or tool_call.get("tool")
                    )
                    tool_args_raw = fn.get("arguments", "{}")
                else:
                    tool_name = tool_call.get("name") or tool_call.get("tool")
                    tool_args_raw = tool_call.get("arguments", "{}")
                tool_call_id = tool_call.get("id", str(uuid.uuid4()))
            else:  # best-effort fallback
                tool_name = getattr(tool_call, "name", None)
                tool_args_raw = getattr(tool_call, "arguments", "{}")
                tool_call_id = getattr(tool_call, "id", str(uuid.uuid4()))

            if not tool_name:
                # Skip malformed tool call
                continue

            try:
                tool_args_obj: Any = (
                    json.loads(tool_args_raw)
                    if isinstance(tool_args_raw, str)
                    else (tool_args_raw or {})
                )
            except Exception:
                tool_args_obj = {}
            tool_args: dict[str, Any] = (
                tool_args_obj if isinstance(tool_args_obj, dict) else {}
            )
            span_id = str(uuid.uuid4())

            ability_called_event = {
                "type": "AbilityCalled",
                "tool": tool_name,
                "correlation_id": self.correlation_id,
                "span_id": span_id,
            }
            # include args for UI visibility
            try:
                ability_called_event["args"] = json.loads(json.dumps(tool_args))
            except Exception:
                ability_called_event["args"] = tool_args
            await self.event_bus.emit(ability_called_event)
            yield ability_called_event

            # Self-evolution: auto-register unknown tools on demand via tool service
            self._tool_service.ensure_tool_registered(tool_name or "", tool_args, self.registry)

            try:
                result = cast(
                    dict[str, Any],
                    await asyncio.wait_for(
                        self.registry.execute(tool_name, tool_args),
                        timeout=SETTINGS.tool_timeout_s,
                    ),
                )
                ability_succeeded_event = {
                    "type": "AbilitySucceeded",
                    "tool": tool_name,
                    "correlation_id": self.correlation_id,
                    "span_id": span_id,
                    "result": result,
                }
                await self.event_bus.emit(ability_succeeded_event)
                yield ability_succeeded_event
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": json.dumps(result),
                    }
                )
            except Exception as e:
                ability_failed_event = {
                    "type": "AbilityFailed",
                    "tool": tool_name,
                    "correlation_id": self.correlation_id,
                    "span_id": span_id,
                    "error": str(e),
                }
                await self.event_bus.emit(ability_failed_event)
                yield ability_failed_event
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": f'{{"error": "Tool execution failed: {e}"}}',
                    }
                )
        # Store the return values for later retrieval
        self._last_acting_result = tool_messages


async def execute_turn(
    user_msg: str, session_id: str, event_bus: Any, registry: Any, kg: Any, model: Any
) -> AsyncGenerator[dict[str, Any], None]:
    """Run a single agent turn and stream events to downstream consumers.

    Args:
        user_msg: Raw user prompt to feed into the orchestrator.
        session_id: Identifier used for correlation across telemetry artifacts.
        event_bus: Event bus responsible for emitting telemetry records.
        registry: Ability registry used to execute tool calls.
        kg: Knowledge graph adapter for context lookup and persistence.
        model: Chat model implementation providing the reasoning stream.

    Yields:
        Event payloads describing state transitions, tool usage and the final
        answer payload. Consumers forward these as SSE frames or use them for
        observability pipelines.
    """
    correlation_id = f"{session_id}-{int(time.time()*1000)}"
    llm_token_chars = 0
    ability_called = 0
    ability_succeeded = 0
    ability_failed = 0
    tools_seen: list[str] = []

    # Optional message optimization/amplification
    if SETTINGS.message_optimizer_enabled:
        # Lazy import to avoid side effects when disabled
        with contextlib.suppress(Exception):
            import src.plugins.message_amplifier_plugin  # noqa: F401
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

    start_event = {
        "type": "TaskStarted",
        "correlation_id": correlation_id,
        "goal": user_msg,
        "session_id": session_id,
    }
    await event_bus.emit(start_event)
    yield start_event

    # Build system prompt with optional output contract
    base_system = "You are a helpful assistant. Use tools when necessary."
    if os.getenv("ALITA_FORMAT_CONTRACT", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        contract = (
            "\n\nOutput Contract:\n"
            "- Always wrap multi-line code in fenced markdown with a language, "
            "e.g. ```python ...```.\n"
            "- Use inline backticks for short code.\n"
            "- Use only ASCII operators (*, /, <=, !=) and quotes; avoid typographic "
            "symbols.\n"
            "- Never omit the multiplication operator; write 5 * x * x (not 5x² or "
            "5 x x).\n"
            "- Provide complete, runnable snippets with imports when including code.\n"
            "- Brief explanation first, then exactly one fenced code block unless "
            "no code is needed.\n"
            "- Self-check and regenerate if rules are violated."
        )
        system_content = base_system + contract
    else:
        system_content = base_system

    kg_context: str | None = None
    kg_goal_id: str | None = None
    if kg is not None:
        with contextlib.suppress(Exception):
            ctx_fn = getattr(kg, "retrieve_relevant_context", None)
            if callable(ctx_fn):
                kg_context = await ctx_fn(user_msg)
            goal_fn = getattr(kg, "get_goal_for_session", None)
            if callable(goal_fn):
                kg_goal = await goal_fn(session_id)
                if isinstance(kg_goal, dict):
                    goal_candidate = kg_goal.get("id")
                    if isinstance(goal_candidate, str):
                        kg_goal_id = goal_candidate

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_content},
    ]
    if kg_context:
        messages.append(
            {
                "role": "system",
                "content": f"Relevant knowledge graph context:\n{kg_context}",
            }
        )
    messages.append({"role": "user", "content": user_msg})

    if kg_context:
        snippet = kg_context if len(kg_context) <= 512 else f"{kg_context[:509]}..."
        await event_bus.emit(
            {
                "type": "KnowledgeContextRetrieved",
                "correlation_id": correlation_id,
                "session_id": session_id,
                "snippet": snippet,
                "goal_id": kg_goal_id,
            }
        )

    llm_response_content = ""

    for _ in range(SETTINGS.max_tool_calls):
        tool_schemas: list[dict[str, Any]] = registry.get_available_tools_schema()

        # Run reasoning step and get results
        async for event in orchestrator._reasoning_step(messages, tool_schemas):
            if event.get("type") == "LLMChunk":
                text = event.get("data", {}).get("text", "")
                if isinstance(text, str):
                    llm_token_chars += len(text)
            yield event
        reasoning_snapshot = orchestrator._last_reasoning_result
        llm_response_content = reasoning_snapshot.text
        tool_calls: list[dict[str, Any]] = list(reasoning_snapshot.tool_calls)
        # Fallback: derive tool calls by parsing streamed content blocks
        if not tool_calls and "<tool_call>" in llm_response_content:
            tool_calls = parse_tool_calls(llm_response_content)

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": llm_response_content,
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)

        if not tool_calls:
            break

        # Run acting step and get results
        async for event in orchestrator._acting_step(tool_calls):
            etype = event.get("type")
            if etype == "AbilityCalled":
                ability_called += 1
                tool_name = event.get("tool")
                if isinstance(tool_name, str):
                    tools_seen.append(tool_name)
            elif etype == "AbilitySucceeded":
                ability_succeeded += 1
            elif etype == "AbilityFailed":
                ability_failed += 1
            yield event
        tool_messages = orchestrator._last_acting_result
        messages.extend(tool_messages)
        # Inject assistant-visible tool_result blocks to advance FakeLLM phase
        for tm in tool_messages:
            with contextlib.suppress(Exception):
                tname = tm.get("name")
                tcontent = tm.get("content", "")
                if tname and isinstance(tcontent, str):
                    tool_result_block = (
                        f'<tool_result tool="{tname}">' f"{tcontent}</tool_result>"
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": tool_result_block,
                        }
                    )

    content_out = llm_response_content or "Task complete."
    if os.getenv("ALITA_FORMAT_ENFORCE", "false").lower() in {"1", "true", "yes", "on"}:
        with contextlib.suppress(Exception):
            content_out = normalize_output_contract(content_out)

    final_answer = {
        "content": content_out,
        "citations": [],
    }

    created_atom_ids: list[str] = []
    pending_bonds: list[tuple[str, str, str]] = []
    bond_previews: list[dict[str, str]] = []
    final_atom_id: str | None = None

    with contextlib.suppress(Exception):
        atom_fn = getattr(kg, "create_atom", None)
        if callable(atom_fn):
            final_atom = await atom_fn("final_answer", final_answer)
            atom_id = final_atom.get("id") if isinstance(final_atom, dict) else None
            if isinstance(atom_id, str):
                final_atom_id = atom_id
                created_atom_ids.append(atom_id)
            await event_bus.emit(
                {
                    "type": "KnowledgeAtomCreated",
                    "correlation_id": correlation_id,
                    "session_id": session_id,
                    "atom_id": atom_id,
                    "atom_type": "final_answer",
                }
            )
            if atom_id and kg_goal_id:
                cb_fn = getattr(kg, "create_bond", None)
                if callable(cb_fn):
                    pending_bonds.append(("ANSWERED", kg_goal_id, atom_id))
                    bond_previews.append(
                        {
                            "type": "ANSWERED",
                            "source": kg_goal_id,
                            "target": atom_id,
                        }
                    )

    atoms_for_alignment: list[str] = []
    if isinstance(kg_goal_id, str):
        atoms_for_alignment.append(kg_goal_id)
    atoms_for_alignment.extend(created_atom_ids)
    bonds_for_alignment = bond_previews
    token_measure = max(len(content_out), llm_token_chars)
    energy_signal = max(
        0.0,
        round(0.1 * token_measure + ability_succeeded - ability_failed, 4),
    )
    todo_count = max(0, ability_called - ability_succeeded - ability_failed)
    bandit_ready = max(0, ability_succeeded)
    reward_payload = {
        "success": 1.0 if final_atom_id else 0.0,
        "tools": sorted({t for t in tools_seen if isinstance(t, str)}),
    }
    loop_alignment_event = {
        "type": "LoopAlignmentTelemetry",
        "correlation_id": correlation_id,
        "session_id": session_id,
        "atoms": atoms_for_alignment,
        "bonds": bonds_for_alignment,
        "energy": energy_signal,
        "todo": todo_count,
        "bandit": bandit_ready,
        "reward": reward_payload,
    }
    await event_bus.emit(loop_alignment_event)
    yield loop_alignment_event

    if pending_bonds and kg is not None:
        cb_fn = getattr(kg, "create_bond", None)
        if callable(cb_fn):
            for bond_type, source_atom_id, target_atom_id in pending_bonds:
                with contextlib.suppress(Exception):
                    await cb_fn(bond_type, source_atom_id, target_atom_id)
                    await event_bus.emit(
                        {
                            "type": "KnowledgeBondCreated",
                            "correlation_id": correlation_id,
                            "session_id": session_id,
                            "bond_type": bond_type,
                            "source_atom_id": source_atom_id,
                            "target_atom_id": target_atom_id,
                        }
                    )

    task_succeeded_event = {
        "type": "TaskSucceeded",
        "correlation_id": correlation_id,
        "data": final_answer,
        "session_id": session_id,
    }
    await event_bus.emit(task_succeeded_event)
    yield task_succeeded_event
