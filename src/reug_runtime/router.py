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

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
import urllib.request
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import SETTINGS
from .message_mw import MessageContext, apply_all

router = APIRouter(prefix="/v1", tags=["agent"])


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


class Orchestrator:
    def __init__(self, event_bus: Any, registry: Any, model: Any, correlation_id: str):
        self.event_bus = event_bus
        self.registry = registry
        self.model = model
        self.correlation_id = correlation_id
        self._mcp_box_dir = Path(os.getenv("MCP_BOX_DIR", ".mcp_box"))
        self._last_reasoning_result: tuple[str, list[dict[str, Any]]] = ("", [])
        self._last_reasoning_result: tuple[str, list[Any]] = ("", [])
        self._last_acting_result: list[dict[str, Any]] = []

    def _persist_mcp_spec(self, spec: dict[str, Any]) -> None:
        try:
            self._mcp_box_dir.mkdir(parents=True, exist_ok=True)
            tid = spec.get("tool_id") or spec.get("name") or "unnamed_tool"
            path = self._mcp_box_dir / f"{tid}.json"
            path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        except Exception:
            # Persistence is best-effort; never break the run
            pass

    async def _ensure_tool(self, tool_name: str, tool_args: dict[str, Any]) -> bool:
        """Auto self-evolve: if a requested tool is unknown, register a minimal one.

        Heuristics:
          - GitHub-related: proxy to existing fetch_github_raw executor
          - URL-related: minimal URL text fetcher
          - Fallback: echo_plan that returns 3 planning steps
        """
        try:
            # Prefer canonical tool_id from MCP index if available
            try:
                import json as _json
                from pathlib import Path

                idx_path = Path(os.getenv("MCP_BOX_DIR", ".mcp_box")) / "index.json"
                if idx_path.exists():
                    index = _json.loads(idx_path.read_text(encoding="utf-8"))
                    aliases = index.get("aliases", {}) or {}
                    # Build alias->canonical map
                    alias_to_canonical: dict[str, str] = {}
                    for canonical, alias_list in aliases.items():
                        for a in alias_list:
                            alias_to_canonical[a] = canonical
                    canon = alias_to_canonical.get(tool_name)
                    if canon:
                        tool_name = canon
            except Exception:
                pass
            if getattr(self.registry, "knows", lambda *_: True)(tool_name):
                return True

            # GitHub proxy
            if (
                any(k in tool_args for k in ("owner", "repo", "path"))
                or "github" in tool_name.lower()
            ):
                contract = {
                    "tool_id": tool_name,
                    "description": "Proxy to fetch a raw file from GitHub",
                    "input_schema": {
                        "type": "object",
                        "required": ["owner", "repo", "path"],
                        "properties": {
                            "owner": {"type": "string"},
                            "repo": {"type": "string"},
                            "path": {"type": "string"},
                            "ref": {"type": "string"},
                        },
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "url": {"type": "string"},
                            "truncated": {"type": "boolean"},
                            "error": {"type": "string"},
                        },
                    },
                }

                async def _exec(a: dict[str, Any]) -> dict[str, Any]:  # proxy
                    result = await self.registry.execute(
                        "fetch_github_raw",
                        {
                            "owner": a.get("owner"),
                            "repo": a.get("repo"),
                            "path": a.get("path"),
                        } | ({"ref": a.get("ref")} if a.get("ref") else {}),
                    )
                    return cast(dict[str, Any], result)

                self.registry.register_tool(contract=contract, executor=_exec)
                # Persist spec for reuse
                self._persist_mcp_spec(
                    {
                        "tool_id": tool_name,
                        "description": contract["description"],
                        "action": "fetch_github_raw",
                        "input_schema": contract["input_schema"],
                        "output_schema": contract["output_schema"],
                    }
                )
                return True

            # URL fetcher
            if ("url" in {k.lower() for k in tool_args}) or (
                any(x in tool_name.lower() for x in ("url", "http", "fetch"))
            ):
                contract = {
                    "tool_id": tool_name,
                    "description": "Fetch a URL and return UTF-8 text (best-effort)",
                    "input_schema": {
                        "type": "object",
                        "required": ["url"],
                        "properties": {
                            "url": {"type": "string"},
                            "truncate": {"type": "integer"},
                        },
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "truncated": {"type": "boolean"},
                            "error": {"type": "string"},
                        },
                    },
                }

                async def _exec(a: dict[str, Any]) -> dict[str, Any]:  # url fetch
                    url = a.get("url")
                    if not isinstance(url, str) or not url:
                        return {"error": "missing url"}
                    truncate = int(a.get("truncate") or 4000)

                    def _do_fetch() -> dict[str, Any]:
                        try:
                            with urllib.request.urlopen(url, timeout=8) as resp:  # nosec B310
                                raw = resp.read()
                            text = raw.decode("utf-8", errors="replace")
                            truncated = False
                            if len(text) > truncate:
                                text = text[:truncate]
                                truncated = True
                            return {"content": text, "truncated": truncated}
                        except Exception as e:  # pragma: no cover - network variability
                            return {"content": "", "truncated": False, "error": str(e)}

                    return await asyncio.to_thread(_do_fetch)

                self.registry.register_tool(contract=contract, executor=_exec)
                # Persist spec for reuse
                self._persist_mcp_spec(
                    {
                        "tool_id": tool_name,
                        "description": contract["description"],
                        "action": "fetch_url_text",
                        "input_schema": contract["input_schema"],
                        "output_schema": contract["output_schema"],
                    }
                )
                return True

            # Fallback planning tool
            contract = {
                "tool_id": tool_name,
                "description": "Echo a minimal plan for the task",
                "input_schema": {
                    "type": "object",
                    "properties": {"task": {"type": "string"}},
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "steps": {"type": "array", "items": {"type": "string"}}
                    },
                },
            }

            async def _exec(a: dict[str, Any]) -> dict[str, Any]:
                t = (a.get("task") or "unknown task").strip()
                return {
                    "steps": [
                        f"Understand: {t}",
                        "Identify resources",
                        "Execute and verify",
                    ]
                }

            self.registry.register_tool(contract=contract, executor=_exec)
            # Persist spec for reuse
            self._persist_mcp_spec(
                {
                    "tool_id": tool_name,
                    "description": contract["description"],
                    "action": "echo_plan",
                    "input_schema": contract["input_schema"],
                    "output_schema": contract["output_schema"],
                }
            )
            return True
        except Exception:
            return False

    async def _reasoning_step(
        self, messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        llm_response_content = ""
        tool_calls: list[Any] = []

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
                tool_calls.extend(chunk.get("tool_calls", []))
        # Store the return values for later retrieval
        self._last_reasoning_result = (llm_response_content, tool_calls)

    async def _acting_step(
        self, tool_calls: list[Any]
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
            tool_args: dict[str, Any] = tool_args_obj if isinstance(tool_args_obj, dict) else {}
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

            # Self-evolution: auto-register unknown tools on demand
            await self._ensure_tool(tool_name or "", tool_args)

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
        llm_response_content, tool_calls_raw = orchestrator._last_reasoning_result
        tool_calls: list[Any] = list(tool_calls_raw)
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

    def _enforce_output_contract_on_text(text: str) -> str:
        """Optionally normalize text to reduce formatting issues.

        - Replace common unicode operators with ASCII inside fenced blocks.
        - Leave content unchanged when enforcement is disabled.
        """
        # Quick exit if nothing to do
        if "```" not in text:
            return text
        # Process fenced blocks only
        parts: list[str] = []
        lines = text.split("\n")
        in_fence = False
        buf: list[str] = []

        def norm_line(s: str) -> str:
            s = s.replace("×", " * ").replace("·", " * ")
            s = s.replace("−", "-").replace("–", "-").replace("—", "-")
            s = s.replace("“", '"').replace("”", '"').replace("’", "'")
            return s

        for line in lines:
            if line.startswith("```"):
                if in_fence:
                    # close fence: flush normalized buffer
                    parts.append("\n".join(buf))
                    buf = []
                    in_fence = False
                    parts.append(line)
                else:
                    in_fence = True
                    parts.append(line)
                continue
            if in_fence:
                buf.append(norm_line(line))
            else:
                parts.append(line)
        # If fence left open, append remaining
        if buf:
            parts.append("\n".join(buf))
        return "\n".join(parts)

    def _normalize_code_blocks(text: str) -> str:
        if "```" not in text:
            return text
        parts: list[str] = []
        lines = text.split("\n")
        in_fence = False
        buf: list[str] = []

        def norm_line(s: str) -> str:
            s = s.replace("×", " * ").replace("·", " * ")
            s = s.replace("−", "-").replace("–", "-").replace("—", "-")
            s = s.replace("“", '"').replace("”", '"').replace("’", "'")
            return s

        for line in lines:
            if line.startswith("```"):
                if in_fence:
                    parts.append("\n".join(buf))
                    buf = []
                    in_fence = False
                    parts.append(line)
                else:
                    in_fence = True
                    parts.append(line)
                continue
            if in_fence:
                buf.append(norm_line(line))
            else:
                parts.append(line)
        if buf:
            parts.append("\n".join(buf))
        return "\n".join(parts)

    content_out = llm_response_content or "Task complete."
    if os.getenv("ALITA_FORMAT_ENFORCE", "false").lower() in {"1", "true", "yes", "on"}:
        with contextlib.suppress(Exception):
            content_out = _normalize_code_blocks(content_out)

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


@router.post("/chat/stream")
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
                    from fastapi.responses import JSONResponse

                    return JSONResponse(
                        status_code=429, content={"error": "rate_limited"}
                    )
    except Exception:
        pass
    raw_body = await request.json()
    body: dict[str, Any]
    if isinstance(raw_body, dict):
        body = raw_body
    else:
        body = {}
    user_msg = body.get("message", "")
    session_id = body.get("session_id", "default")

    state = cast(Any, request.app.state)

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


@router.get("/chat/stream")
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
