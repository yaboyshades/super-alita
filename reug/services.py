"""
REUG services wired to Script-of-Thought parsing and a computational environment executor.
States remain thin: all compute/IO happens here; FSM only orchestrates + emits events.
"""
from __future__ import annotations

import asyncio, os, time, random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Callable, Tuple

try:
    # Real SoT + executor (preferred if present in repo)
    from super_alita.script_of_thought.parser import ScriptParser  # type: ignore
    from super_alita.script_of_thought.interpreter import StepType as SoTStepType  # type: ignore
    from super_alita.computational_env.executor import Executor  # type: ignore
    HAVE_REAL_IMPL = True
except Exception:
    HAVE_REAL_IMPL = False

try:
    import jsonschema  # optional, for input/output schema validation
except Exception:
    jsonschema = None  # type: ignore

from .events import EventEmitter, hash_json, new_span_id

ENFORCE_SCHEMA = os.getenv("REUG_SCHEMA_ENFORCE", "true").lower() == "true"

StepKind = Literal["SEARCH","COMPUTE","ANALYZE","GENERATE","VALIDATE"]

@dataclass
class PlanStep:
    step_id: str
    kind: StepKind
    args: Dict[str, Any]

@dataclass
class Plan:
    steps: List[PlanStep]

# ---------- Fallbacks (keeps tests green without repo-local deps) ----------
class _FallbackParser:
    """Very small SoT parser fallback: expects user_input['sot'] as a list[dict]."""
    def parse(self, user_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        sot = user_input.get("sot")
        if isinstance(sot, str):
            # very simple "type: arg" lines format
            steps = []
            for i, line in enumerate([l.strip() for l in sot.splitlines() if l.strip()]):
                if ":" in line:
                    k, v = [x.strip() for x in line.split(":", 1)]
                    steps.append({"id": f"step{i+1}", "kind": k.upper(), "args": {"expr": v}})
            return steps
        if isinstance(sot, list):
            steps = []
            for i, s in enumerate(sot):
                if isinstance(s, dict):
                    steps.append({
                        "id": s.get("id", f"step{i+1}"),
                        "kind": str(s.get("kind", "COMPUTE")).upper(),
                        "args": dict(s.get("args", {})),
                    })
                elif isinstance(s, str) and ":" in s:
                    k, v = [x.strip() for x in s.split(":", 1)]
                    steps.append({"id": f"step{i+1}", "kind": k.upper(), "args": {"expr": v}})
                else:
                    raise ValueError("Invalid SoT step format")
            return steps
        raise ValueError("Fallback SoT parser requires user_input['sot'] as str or list")

class _FallbackExecutor:
    """Extremely simple safe-ish executor for COMPUTE/ANALYZE/GENERATE kinds (tests only)."""
    def run(self, tool_id: str, args: Dict[str, Any], timeout_s: float = 5.0) -> Dict[str, Any]:
        kind = args.get("_kind", "COMPUTE")
        if kind in ("COMPUTE","ANALYZE"):
            expr = args.get("expr") or args.get("code") or "None"
            # Very limited eval (no imports), only numeric/simple expressions
            allowed_names = {"__builtins__": {}}
            return {"value": eval(expr, allowed_names, {})}  # noqa: S307 (test-only)
        if kind == "GENERATE":
            return {"text": args.get("text", "ok")}
        return {"ok": True}

# ---------- Tool registry hook (optional) ----------
ToolResolver = Callable[[PlanStep, Dict[str, Any]], Tuple[str, Dict[str, Any]]]

def default_tool_resolver(step: PlanStep, _ctx: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Minimal resolver: map step.kind to pseudo-tool IDs and pass through args.
    Real systems can consult a registry by capability + schema.
    """
    tool_map = {
        "SEARCH":  "tool.search.basic",
        "COMPUTE": "tool.compute.python",
        "ANALYZE": "tool.analyze.basic",
        "GENERATE":"tool.generate.text",
        "VALIDATE":"tool.validate.schema",
    }
    return tool_map.get(step.kind, f"tool.{step.kind.lower()}.generic"), step.args

# ---------- Validation helpers ----------
def _validate(schema: Optional[Dict[str, Any]], data: Any) -> Optional[str]:
    if not ENFORCE_SCHEMA or not schema or not jsonschema:
        return None
    try:
        jsonschema.validate(instance=data, schema=schema)  # type: ignore
        return None
    except Exception as e:
        return str(e)

# ---------- Public service factory ----------
def create_services(
    emitter: EventEmitter,
    tool_resolver: ToolResolver = default_tool_resolver,
    *,
    per_tool_timeout_s: float | None = None,
    max_retries: int | None = None,
    retry_base_ms: int | None = None,
) -> Dict[str, Callable]:
    """
    Returns the four service callables expected by the FSM:
      - decompose(user_input) -> Plan
      - select_tool(step, ctx) -> {status, tool?}
      - execute(tool, args, ctx) -> {status, result?|error?}
      - process_result(ctx) -> {task_complete, more_steps_needed}
    """
    per_tool_timeout_s = float(os.getenv("REUG_EXEC_TIMEOUT_S", per_tool_timeout_s or 20.0))
    max_retries = int(os.getenv("REUG_EXEC_MAX_RETRIES", max_retries if max_retries is not None else 2))
    retry_base_ms = int(os.getenv("REUG_RETRY_BASE_MS", retry_base_ms if retry_base_ms is not None else 100))

    parser = ScriptParser() if HAVE_REAL_IMPL else _FallbackParser()
    executor = Executor() if HAVE_REAL_IMPL else _FallbackExecutor()

    async def decompose(user_input: Dict[str, Any]) -> Plan:
        # Accepts either pre-parsed plan or a raw SoT input to parse.
        if "plan" in user_input and isinstance(user_input["plan"], list):
            steps_raw = user_input["plan"]
        else:
            steps_raw = parser.parse(user_input)
        steps: List[PlanStep] = []
        for s in steps_raw:
            step_id = s.get("id") or s.get("step_id") or f"step{len(steps)+1}"
            kind = str(s.get("kind", "COMPUTE")).upper()
            args = dict(s.get("args", {}))
            # Provide kind for fallback executor convenience
            args["_kind"] = kind
            steps.append(PlanStep(step_id=step_id, kind=kind, args=args))
        return Plan(steps=steps)

    async def select_tool(step: PlanStep, ctx: Dict[str, Any]) -> Dict[str, Any]:
        tool_id, tool_args = tool_resolver(step, ctx)
        if not tool_id:
            return {"status": "NOT_FOUND", "reason": "UNKNOWN_TOOL"}
        tool = {"tool_id": tool_id, "input_schema": step.args.get("_input_schema"), "output_schema": step.args.get("_output_schema")}
        # Input schema validation before we attempt execution
        err = _validate(tool.get("input_schema"), tool_args)
        if err:
            return {"status": "NOT_FOUND", "reason": f"INPUT_SCHEMA_VIOLATION: {err}"}
        return {"status": "FOUND", "tool": tool, "args": tool_args}

    async def execute(tool: Dict[str, Any], args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        correlation_id = ctx.get("correlation_id")
        tool_id = tool["tool_id"]
        span_id = new_span_id()
        emitter.emit(
            event_type="TOOL_CALL_START",
            payload={"tool_id": tool_id, "input_hash": hash_json(args), "step_id": ctx.get("step_index"), "args_preview": {k: args[k] for k in list(args)[:3]}},
            correlation_id=correlation_id,
            span_id=span_id,
        )
        # Retry with jitter
        attempt = 0
        last_err: Optional[str] = None
        backoff = 0
        while attempt <= max_retries:
            try:
                # Async timeout wrapper
                res = await asyncio.wait_for(
                    asyncio.to_thread(executor.run, tool_id, args, per_tool_timeout_s),
                    timeout=per_tool_timeout_s + 1.0,
                )
                # Output schema validation
                err = _validate(tool.get("output_schema"), res)
                if err:
                    raise ValueError(f"OUTPUT_SCHEMA_VIOLATION: {err}")
                evt = emitter.emit(
                    event_type="TOOL_CALL_OK",
                    payload={
                        "tool_id": tool_id,
                        "result": res,
                        "output_hash": hash_json(res),
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "backoff_ms": backoff,
                    },
                    correlation_id=correlation_id,
                    span_id=span_id,
                )
                try:
                    from .kg import store
                    for k, v in res.items():
                        if isinstance(v, (str, int, float)):
                            store.add_triple(tool_id, k, str(v), src=evt["event_id"])
                            break
                except Exception:
                    pass
                return {"status": "SUCCESS", "result": res}
            except asyncio.TimeoutError:
                last_err = f"TIMEOUT after {per_tool_timeout_s}s"
            except Exception as e:
                last_err = str(e)
            # Retry backoff
            attempt += 1
            if attempt <= max_retries:
                backoff = retry_base_ms * (2 ** (attempt - 1)) + random.randint(0, 50)
                await asyncio.sleep(backoff / 1000.0)

        emitter.emit(
            event_type="TOOL_CALL_ERR",
            payload={
                "tool_id": tool_id,
                "error": last_err or "unknown",
                "attempt": attempt,
                "max_retries": max_retries,
                "backoff_ms": backoff,
            },
            correlation_id=correlation_id,
            span_id=span_id,
        )
        return {"status": "ERROR", "error": last_err}

    async def process_result(ctx_dict: Dict[str, Any]) -> Dict[str, bool]:
        # Simple default: if FSM increased step_index >= len(steps) → complete.
        plan: Plan = ctx_dict.get("plan")  # type: ignore
        step_index: int = ctx_dict.get("step_index", 0)
        more = step_index < len(plan.steps)
        return {"more_steps_needed": more, "task_complete": not more}

    return {
        "decompose": decompose,
        "select_tool": select_tool,
        "execute": execute,
        "process_result": process_result,
    }
