from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict

from .events import new_id
from .tools.registry import register_tool


class State(str, Enum):
    AWAITING_INPUT = "AWAITING_INPUT"
    DECOMPOSE_TASK = "DECOMPOSE_TASK"
    SELECT_TOOL = "SELECT_TOOL"
    EXECUTE_TOOL = "EXECUTE_TOOL"
    PROCESS_TOOL_RESULT = "PROCESS_TOOL_RESULT"
    CREATING_DYNAMIC_TOOL = "CREATING_DYNAMIC_TOOL"
    HANDLING_ERROR = "HANDLING_ERROR"
    RESPONDING_SUCCESS = "RESPONDING_SUCCESS"


@dataclass
class FSMContext:
    raw_input: Any = None
    plan: Any = None
    current_step: int = 0
    results: list | None = None
    error: Any = None
    correlation_id: str = ""


class ExecutionFlow:
    def __init__(self, services: Dict[str, Callable], emitter):
        self.services = services
        self.emitter = emitter

    def _emit_transition(self, from_state: State, to_state: State, correlation_id: str) -> None:
        self.emitter.emit(
            event_type="STATE_TRANSITION",
            payload={"from": from_state.name, "to": to_state.name},
            correlation_id=correlation_id,
        )

    async def _create_dynamic_tool(self, step, ctx_dict, correlation_id: str):
        tool = {
            "tool_id": f"dynamic.{step.kind.lower()}.{step.step_id}",
            "version": "1",
            "input_schema": step.args.get("_input_schema"),
            "output_schema": step.args.get("_output_schema"),
            "binding": "python",
            "guard": "allow",
        }
        register_tool(tool)
        self.emitter.emit(
            event_type="TOOL_REGISTERED",
            payload={"tool_id": tool["tool_id"]},
            correlation_id=correlation_id,
        )
        try:
            await self.services["execute"](
                tool,
                {"_kind": "GENERATE", "text": "ping"},
                {**ctx_dict, "step_index": ctx_dict["current_step"]},
            )
        except Exception:
            pass
        return tool

    async def run(self, user_input: Dict[str, Any]) -> FSMContext:
        ctx = FSMContext(raw_input=user_input, results=[], correlation_id=new_id())
        ctx_dict = ctx.__dict__
        self._emit_transition(State.AWAITING_INPUT, State.DECOMPOSE_TASK, ctx.correlation_id)
        plan = await self.services["decompose"](user_input)
        ctx.plan = plan

        while ctx.current_step < len(plan.steps):
            self._emit_transition(
                State.DECOMPOSE_TASK if ctx.current_step == 0 else State.PROCESS_TOOL_RESULT,
                State.SELECT_TOOL,
                ctx.correlation_id,
            )
            step = plan.steps[ctx.current_step]
            sel = await self.services["select_tool"](step, ctx_dict)
            if sel["status"] == "FOUND":
                self._emit_transition(State.SELECT_TOOL, State.EXECUTE_TOOL, ctx.correlation_id)
                args = sel.get("args", step.args)
                res = await self.services["execute"](
                    sel["tool"],
                    args,
                    {**ctx_dict, "step_index": ctx.current_step},
                )
                if res["status"] == "SUCCESS":
                    ctx.results.append(res["result"])
                    ctx.current_step += 1
                    self._emit_transition(State.EXECUTE_TOOL, State.PROCESS_TOOL_RESULT, ctx.correlation_id)
                    pr = await self.services["process_result"](ctx_dict)
                    if pr.get("task_complete"):
                        self._emit_transition(
                            State.PROCESS_TOOL_RESULT, State.RESPONDING_SUCCESS, ctx.correlation_id
                        )
                        return ctx
                else:
                    ctx.error = res.get("error")
                    self._emit_transition(State.EXECUTE_TOOL, State.HANDLING_ERROR, ctx.correlation_id)
                    return ctx
            else:
                reason = sel.get("reason")
                if reason == "UNKNOWN_TOOL":
                    self._emit_transition(State.SELECT_TOOL, State.CREATING_DYNAMIC_TOOL, ctx.correlation_id)
                    tool = await self._create_dynamic_tool(step, ctx_dict, ctx.correlation_id)
                    self._emit_transition(State.CREATING_DYNAMIC_TOOL, State.EXECUTE_TOOL, ctx.correlation_id)
                    res = await self.services["execute"](
                        tool,
                        step.args,
                        {**ctx_dict, "step_index": ctx.current_step},
                    )
                    if res["status"] == "SUCCESS":
                        ctx.results.append(res["result"])
                        ctx.current_step += 1
                        self._emit_transition(State.EXECUTE_TOOL, State.PROCESS_TOOL_RESULT, ctx.correlation_id)
                        pr = await self.services["process_result"](ctx_dict)
                        if pr.get("task_complete"):
                            self._emit_transition(
                                State.PROCESS_TOOL_RESULT, State.RESPONDING_SUCCESS, ctx.correlation_id
                            )
                            return ctx
                    else:
                        ctx.error = res.get("error")
                        self._emit_transition(State.EXECUTE_TOOL, State.HANDLING_ERROR, ctx.correlation_id)
                        return ctx
                else:
                    ctx.error = reason
                    self._emit_transition(State.SELECT_TOOL, State.HANDLING_ERROR, ctx.correlation_id)
                    return ctx

        self._emit_transition(State.PROCESS_TOOL_RESULT, State.RESPONDING_SUCCESS, ctx.correlation_id)
        return ctx

