from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict

class State(str, Enum):
    AWAITING_INPUT = "awaiting_input"
    DECOMPOSE_TASK = "decompose_task"
    SELECT_TOOL = "select_tool"
    EXECUTE_TOOL = "execute_tool"
    PROCESS_TOOL_RESULT = "process_tool_result"
    CREATING_DYNAMIC_TOOL = "creating_dynamic_tool"
    HANDLING_ERROR = "handling_error"
    RESPONDING_SUCCESS = "responding_success"

@dataclass
class FSMContext:
    raw_input: Any = None
    plan: Any = None
    current_step: int = 0
    results: list = None
    error: Exception = None

class ExecutionFlow:
    def __init__(self, services: Dict[str, Callable], emit_event: Callable[[Dict], None]):
        self.services = services
        self.emit_event = emit_event
        self.state = State.AWAITING_INPUT
        self.ctx = FSMContext(results=[])

    def send(self, event_type: str, payload: Dict[str, Any]):
        if self.state == State.AWAITING_INPUT and event_type == "START":
            self.state = State.DECOMPOSE_TASK
            self.ctx.raw_input = payload
            self.emit_event({"event_type": "STATE_TRANSITION","from":"AWAITING_INPUT","to":"DECOMPOSE_TASK"})
            self.ctx.plan = self.services["decompose"](payload)
            self.state = State.SELECT_TOOL

        elif self.state == State.SELECT_TOOL:
            step = self.ctx.plan[self.ctx.current_step]
            sel = self.services["select_tool"](step)
            if sel["status"] == "FOUND":
                self.state = State.EXECUTE_TOOL
                res = self.services["execute"](sel["tool"], step, self.ctx)
                if res["status"] == "SUCCESS":
                    self.ctx.results.append(res["result"])
                    self.ctx.current_step += 1
                    more = self.services["process_result"](self.ctx)
                    if more.get("task_complete"):
                        self.state = State.RESPONDING_SUCCESS
                else:
                    self.state = State.HANDLING_ERROR
            else:
                self.state = State.CREATING_DYNAMIC_TOOL

