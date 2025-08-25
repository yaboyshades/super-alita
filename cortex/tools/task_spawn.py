from __future__ import annotations

from typing import Any

from cortex.common.types import ToolSpec
from cortex.orchestrator.subagent import SubAgentRunner

task_spawn_spec = ToolSpec(
    name="task.spawn",
    description="Launch a focused sub-agent with its own system prompt and an isolated conversation. Returns only a final summary.",
    args_schema={
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": "Short label for the sub-agent, e.g., 'requirements-analyzer'",
            },
            "system_prompt": {
                "type": "string",
                "description": "Detailed specification/instructions for the sub-agent.",
            },
            "goal_message": {
                "type": "string",
                "description": "Initial user message describing the task and inputs.",
            },
        },
        "required": ["system_prompt", "goal_message"],
    },
)


def task_spawn(args: dict[str, Any], llm_client) -> dict[str, Any]:
    runner = SubAgentRunner(llm_client=llm_client)
    res = runner.run(
        goal_system_prompt=args["system_prompt"],
        initial_user_message=args["goal_message"],
    )
    # Return only a compact summary to main agent history
    return {
        "agent": args.get("agent_name") or "subagent",
        "summary": res["summary"],
        "steps": res["steps"],
        "used_tool": res["used_tool"],
    }
