from __future__ import annotations

from typing import Any, Dict

import asyncio
from src.script_of_thought.parser import ScriptOfThoughtParser, ScriptStep, StepType
from src.computational_env.executor import ComputationalEnvironment

parser = ScriptOfThoughtParser()
_env: ComputationalEnvironment | None = None


def _get_env() -> ComputationalEnvironment:
    global _env
    if _env is None:
        _env = ComputationalEnvironment()
    return _env


def decompose(raw_input: str) -> list[ScriptStep]:
    script = parser.parse(raw_input)
    return script.steps


def select_tool(step: ScriptStep) -> Dict[str, Any]:
    if step.step_type in {StepType.COMPUTE, StepType.TOOL}:
        return {"status": "FOUND", "tool": "code"}
    return {"status": "FOUND", "tool": None}


def execute(tool: str | None, step: ScriptStep, ctx: Any) -> Dict[str, Any]:
    if tool == "code":
        env = _get_env()
        result = asyncio.run(env.execute_code(step.content))
        return {"status": "SUCCESS", "result": result}
    return {"status": "SUCCESS", "result": step.content}


def process_result(ctx) -> Dict[str, Any]:
    if ctx.current_step >= len(ctx.plan):
        return {"task_complete": True}
    return {"task_complete": False}
