import os
from typing import Any

from cortex.common.types import ToolSpec
from cortex.orchestrator.subagent import SubAgentRunner
from cortex.proxy.prompting import ReminderEngine, build_prompt_bundle


class DummyLLM:
    def __init__(self, script=None):
        self.script = script or []

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.script:
            return {"content": "Final summary: done [done]"}
        return self.script.pop(0)


def test_prompt_bundle_and_reminders():
    os.environ["CORTEX_PROMPT_REMINDERS"] = "1"
    tools = [
        ToolSpec(name="todo.write", description="...", args_schema={"type": "object"})
    ]
    history = [{"role": "user", "content": "please fix failing tests"}]
    rem = ReminderEngine()
    rem.last_tool_step_idx = -10
    rem_text = rem.maybe_inject(todos_empty=True, step_idx=6)
    bundle = build_prompt_bundle(
        history, tools, context_text="ctx", reminders_text=rem_text
    )
    assert "Task Management" in bundle.system
    assert bundle.reminders is not None


def test_subagent_isolation_and_summary():
    # Script causes one tool call then a done
    script = [
        {
            "tool_call": {
                "tool": "todo.write",
                "args": {"parent_title": "Fix", "subtasks": [{"title": "Reproduce"}]},
            }
        },
        {"content": "Final summary: Investigated and planned [done]"},
    ]
    runner = SubAgentRunner(DummyLLM(script=script))
    res = runner.run(
        goal_system_prompt="Subagent persona: requirements-analyzer",
        initial_user_message="Analyze login failure",
    )
    assert res["steps"] >= 1
    assert "summary" in res and "Investigated" in res["summary"]
    # Internal history is NOT exposed anywhere; we only see compact summary
