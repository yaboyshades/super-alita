import json, re, time, asyncio, math
from typing import AsyncGenerator, Optional
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from reug_runtime.config import SETTINGS

# ==== Injected dependencies via app.state ====
class EventBus: ...
class AbilityRegistry: ...
class KnowledgeGraph: ...
class LLMClient: ...  # wrapper around your provider

router = APIRouter(prefix="/v1", tags=["agent"])

# ---------- Helpers ----------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _new_corr(session_id: str) -> str:
    return f"turn_{session_id}_{int(time.time())}"

def _new_span(i: int) -> str:
    return f"span_{i}_{int(time.time())}"

def _hash_json(obj) -> str:
    try:
        import hashlib
        h = hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()
        return h[:16]
    except Exception:
        return "na"

def _backoff_ms(base_ms: int, attempt: int) -> int:
    # attempt: 0,1,2...  -> base, 2*base, 4*base ...
    return base_ms * (2 ** max(0, attempt))

# ---------- Block parser (robust to partial chunks) ----------
class BlockParser:
    """
    Stateful parser for streamed tags:
      <tool_call>{"tool":"name","args":{...}}</tool_call>
      <tool_result tool="name">{...}</tool_result>
      <final_answer>{"content":"...","citations":[...]}</final_answer>
    """
    def __init__(self):
        self.buffer = ""
        self.pattern = re.compile(r'<(\w+)([^>]*)>(\{.*?\})</\1>', re.DOTALL)

    def feed(self, chunk: str):
        self.buffer += chunk

    def _extract(self, tag: str) -> Optional[tuple[dict, str]]:
        for m in self.pattern.finditer(self.buffer):
            name, attrs, payload = m.group(1), m.group(2), m.group(3)
            if name == tag:
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    return None
                raw = m.group(0)
                # remove the first occurrence to advance
                self.buffer = self.buffer.replace(raw, "", 1)
                return data, attrs
        return None

    def get_tool_call(self) -> Optional[dict]:
        hit = self._extract("tool_call")
        return hit[0] if hit else None

    def get_final_answer(self) -> Optional[dict]:
        hit = self._extract("final_answer")
        return hit[0] if hit else None

    def reset(self):
        self.buffer = ""

# ---------- Dynamic tool synthesis (contract-first) ----------
def _synthesize_contract_from_call(tool_name: str, args: dict) -> dict:
    """
    Minimal viable contract. In a richer setup, you'd consult the KG or templates.
    """
    return {
        "tool_id": tool_name,
        "version": "0.0.1",
        "description": f"Synthesized tool for {tool_name}",
        "input_schema": {
            "type": "object",
            "properties": {
                k: {
                    "type": "string" if isinstance(v, str)
                    else "number" if isinstance(v, (int, float))
                    else "object"
                }
                for k, v in args.items()
            },
            "additionalProperties": True,
        },
        "output_schema": {"type": "object", "additionalProperties": True},
        "binding": {"type": "mcp_or_sdk", "endpoint": tool_name},
        "guard": {"pii_allowed": False},
    }

# ---------- Core orchestrator ----------
async def execute_turn(
    user_msg: str,
    session_id: str,
    event_bus: EventBus,
    registry: AbilityRegistry,
    kg: KnowledgeGraph,
    model: LLMClient,
) -> AsyncGenerator[str, None]:
    corr = _new_corr(session_id)
    parser = BlockParser()
    tool_calls = 0
    used_spans: list[dict] = []  # for KG provenance

    # REUG: STATE_TRANSITION (AWAITING_INPUT -> DECOMPOSE_TASK)
    await event_bus.emit({"type": "STATE_TRANSITION", "from": "AWAITING_INPUT", "to": "DECOMPOSE_TASK", "correlation_id": corr})

    # Enrich prompt from KG
    kg_ctx = await kg.retrieve_relevant_context(user_msg)
    goal = await kg.get_goal_for_session(session_id)
    tools = registry.get_available_tools_schema()

    system_prompt = f"""You are an agent with tools.

CONTEXT:
{kg_ctx}

GOAL:
{goal.get("description", "Assist user")}

TOOLS:
{json.dumps(tools, indent=2)}

Protocol:
- Plan silently.
- To call a tool: <tool_call>{{"tool":"name","args":{{...}}}}</tool_call>
- You will receive: <tool_result tool="name">{{...}}</tool_result> or <tool_error tool="name">{{"error":"..."}}</tool_error>
- Recover from errors when possible (fix args or choose an alternative tool).
- End with: <final_answer>{{"content":"...","citations":[]}}</final_answer>
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    await event_bus.emit({
        "type": "TaskStarted",
        "correlation_id": corr,
        "goal": goal.get("id"),
        "user_msg_hash": _hash_json(user_msg),
    })

    # REUG: DECOMPOSE_TASK -> SELECT_TOOL (we let the model select; we only instrument)
    await event_bus.emit({"type": "STATE_TRANSITION", "from": "DECOMPOSE_TASK", "to": "SELECT_TOOL", "correlation_id": corr})

    while tool_calls < SETTINGS.max_tool_calls:
        # Stream model output for this internal cycle
        try:
            async for chunk in model.stream_chat(messages, timeout=SETTINGS.model_stream_timeout_s):
                text = chunk["content"]
                # Stream through to client
                yield text
                parser.feed(text)

                # Detect tool call
                call = parser.get_tool_call()
                if not call:
                    continue

                tool_name = call.get("tool")
                tool_args = call.get("args", {})
                span = _new_span(tool_calls)
                attempt = 0
                max_attempts = SETTINGS.max_retries + 1  # first try + retries

                # REUG: SELECT_TOOL -> EXECUTE_TOOL
                await event_bus.emit({"type": "STATE_TRANSITION", "from": "SELECT_TOOL", "to": "EXECUTE_TOOL", "correlation_id": corr})

                # Unknown tool? Trigger dynamic creation
                knows = False
                try:
                    knows = getattr(registry, "knows")(tool_name)  # type: ignore[attr-defined]
                except Exception:
                    # If registry lacks .knows, optimistically try validate
                    knows = registry.validate_args(tool_name, tool_args)

                if not knows:
                    await event_bus.emit({"type": "STATE_TRANSITION", "from": "EXECUTE_TOOL", "to": "CREATING_DYNAMIC_TOOL", "correlation_id": corr})
                    contract = _synthesize_contract_from_call(tool_name, tool_args)
                    # health-check / ping
                    healthy = True
                    try:
                        healthy = await getattr(registry, "health_check")(contract)  # type: ignore[attr-defined]
                    except Exception:
                        # If no health_check, proceed but note it
                        healthy = True
                    if healthy:
                        try:
                            await getattr(registry, "register")(contract)  # type: ignore[attr-defined]
                            await event_bus.emit({
                                "type": "TOOL_REGISTERED",
                                "correlation_id": corr,
                                "tool": tool_name,
                                "contract_hash": _hash_json(contract),
                            })
                        except Exception as e:
                            # Surface as a tool_error for model recovery
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": f'<tool_error tool="{tool_name}">{{"error":"registration_failed: {str(e)}"}}</tool_error>',
                                }
                            )
                            parser.reset()
                            break  # restart outer while loop
                    else:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": f'<tool_error tool="{tool_name}">{{"error":"health_check_failed"}}</tool_error>',
                            }
                        )
                        parser.reset()
                        break

                # Execute with retries/backoff
                while attempt < max_attempts:
                    start_ms = _now_ms()
                    await event_bus.emit(
                        {
                            "type": "AbilityCalled",
                            "correlation_id": corr,
                            "span_id": span,
                            "tool": tool_name,
                            "args_hash": _hash_json(tool_args),
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts,
                        }
                    )
                    try:
                        result = await asyncio.wait_for(
                            registry.execute(tool_name, tool_args),
                            timeout=SETTINGS.tool_timeout_s,
                        )
                        dur_ms = _now_ms() - start_ms
                        await event_bus.emit(
                            {
                                "type": "AbilitySucceeded",
                                "correlation_id": corr,
                                "span_id": span,
                                "tool": tool_name,
                                "duration_ms": dur_ms,
                                "output_hash": _hash_json(result),
                            }
                        )
                        # REUG: EXECUTE_TOOL -> PROCESS_TOOL_RESULT
                        await event_bus.emit(
                            {
                                "type": "STATE_TRANSITION",
                                "from": "EXECUTE_TOOL",
                                "to": "PROCESS_TOOL_RESULT",
                                "correlation_id": corr,
                            }
                        )
                        messages.append(
                            {
                                "role": "assistant",
                                "content": f'<tool_result tool="{tool_name}">{json.dumps(result)}</tool_result>',
                            }
                        )
                        used_spans.append({"span_id": span, "tool": tool_name})
                        parser.reset()
                        tool_calls += 1
                        break  # tool execution done for this cycle
                    except asyncio.TimeoutError:
                        dur_ms = _now_ms() - start_ms
                        await event_bus.emit(
                            {
                                "type": "AbilityFailed",
                                "correlation_id": corr,
                                "span_id": span,
                                "tool": tool_name,
                                "duration_ms": dur_ms,
                                "error": f"timeout_{SETTINGS.tool_timeout_s}s",
                                "attempt": attempt + 1,
                                "max_attempts": max_attempts,
                            }
                        )
                        attempt += 1
                        if attempt >= max_attempts:
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": f'<tool_error tool="{tool_name}">{{"error":"timeout"}}</tool_error>',
                                }
                            )
                            parser.reset()
                            break
                        await asyncio.sleep(_backoff_ms(SETTINGS.retry_base_ms, attempt - 1) / 1000)
                    except Exception as e:
                        dur_ms = _now_ms() - start_ms
                        await event_bus.emit(
                            {
                                "type": "AbilityFailed",
                                "correlation_id": corr,
                                "span_id": span,
                                "tool": tool_name,
                                "duration_ms": dur_ms,
                                "error": str(e),
                                "attempt": attempt + 1,
                                "max_attempts": max_attempts,
                            }
                        )
                        attempt += 1
                        if attempt >= max_attempts:
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": f'<tool_error tool="{tool_name}">{{"error":"exception","message":{json.dumps(str(e))}}}</tool_error>',
                                }
                            )
                            parser.reset()
                            break
                        await asyncio.sleep(_backoff_ms(SETTINGS.retry_base_ms, attempt - 1) / 1000)

                # after handling this tool (success or terminal error), break the inner model stream
                break

            # After processing the chunk stream, check for final answer
            final = parser.get_final_answer()
            if final:
                # write provenance to KG
                answer_atom = await kg.create_atom("ANSWER", final)
                try:
                    await kg.create_bond("RELATES_TO", answer_atom["id"], goal["id"])
                except Exception:
                    pass
                for span_info in used_spans:
                    try:
                        await kg.create_bond("CAUSED_BY", answer_atom["id"], span_info["span_id"])
                    except Exception:
                        pass
                await event_bus.emit(
                    {
                        "type": "STATE_TRANSITION",
                        "from": "PROCESS_TOOL_RESULT",
                        "to": "RESPONDING_SUCCESS",
                        "correlation_id": corr,
                    }
                )
                await event_bus.emit(
                    {
                        "type": "TaskSucceeded",
                        "correlation_id": corr,
                        "answer_atom_id": answer_atom["id"],
                    }
                )
                return

        except asyncio.TimeoutError:
            yield "\n[ERROR: model_stream_timeout]"
            break

    # Exceeded caps or aborted
    await event_bus.emit({"type": "TaskFailed", "correlation_id": corr, "reason": "tool_cap_or_abort"})
    yield "\n[ERROR: Agent unable to complete request]"

# ---------- FastAPI endpoint ----------
@router.post("/chat/stream")
async def chat_stream(request: Request):
    body = await request.json()
    user_msg = body["message"]
    session_id = body.get("session_id", "default")
    app = request.app
    gen = execute_turn(
        user_msg,
        session_id,
        app.state.event_bus,
        app.state.ability_registry,
        app.state.kg,
        app.state.llm_model,
    )
    return StreamingResponse(
        gen,
        media_type="text/plain",
        headers={"X-Correlation-ID": f"session_{session_id}", "Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
