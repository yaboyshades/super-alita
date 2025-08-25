from __future__ import annotations

from typing import Any

from cortex.common.types import PromptBundle, ToolSpec
from cortex.config.flags import SUBAGENT
from cortex.proxy.prompting import build_prompt_bundle


class SubAgentRunner:
    """
    Launches a task-focused sub-agent with isolated history and tool policy.
    Returns a summarized result and discards internal history.
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def _filter_tools(self) -> list[ToolSpec]:
        # Import locally to avoid circular dependency
        from cortex.tools.todo_write import todo_write_spec

        specs = []
        # Hardcoded tool specs to avoid circular import
        available_specs = [todo_write_spec]
        for spec in available_specs:
            # Enforce safety fences
            if not SUBAGENT.allow_tools:
                continue
            # Could filter shell/net tools here if present
            specs.append(spec)
        return specs

    def run(self, goal_system_prompt: str, initial_user_message: str) -> dict[str, Any]:
        if not SUBAGENT.enable_subagents:
            return {"summary": "(subagents disabled)", "steps": 0}
        tools = self._filter_tools()
        history: list[dict[str, Any]] = [
            {"role": "user", "content": initial_user_message}
        ]
        context = None  # no inherited full history by design

        steps = 0
        used_tool_any = False
        while steps < SUBAGENT.max_steps:
            bundle: PromptBundle = build_prompt_bundle(
                history=history,
                tools=tools,
                context_text=context,
                reminders_text=None,
            )
            # Override system with sub-agent persona/spec (isolation)
            bundle_system = goal_system_prompt + "\n\n" + bundle.system
            payload = {
                "system": bundle_system,
                "tools": [spec.__dict__ for spec in tools],
                "messages": bundle.history
                + (
                    [{"role": "system", "content": bundle.reminders}]
                    if bundle.reminders
                    else []
                ),
            }
            resp = self.llm.complete(payload)  # expected to support tool calls or text
            steps += 1

            if isinstance(resp, dict) and resp.get("tool_call"):
                used_tool_any = True
                call = resp["tool_call"]
                tool_name = call["tool"]
                args = call.get("args", {})
                try:
                    if tool_name == "todo.write":
                        from cortex.tools.todo_write import todo_write

                        result = todo_write(args)
                    else:
                        result = {"ok": False, "error": f"unknown tool {tool_name}"}
                except Exception as e:
                    result = {"ok": False, "error": repr(e)}
                history.append({"role": "tool", "name": tool_name, "content": result})
                continue

            content = resp["content"] if isinstance(resp, dict) else str(resp)
            history.append({"role": "assistant", "content": content})
            # Simple stopping condition: model declares completion
            if "[done]" in content.lower() or "final summary:" in content.lower():
                break

        # Summarize (truncate to max chars)
        summary = ""
        for msg in history[-20:]:
            if msg["role"] in ("assistant", "tool", "user"):
                summary += f"{msg['role']}: {str(msg['content'])}\n"
        if len(summary) > SUBAGENT.summary_max_chars:
            summary = summary[: SUBAGENT.summary_max_chars - 3] + "..."
        # DO NOT return history – isolation contract
        return {"summary": summary.strip(), "steps": steps, "used_tool": used_tool_any}
