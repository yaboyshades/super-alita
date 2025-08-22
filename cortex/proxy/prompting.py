from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from cortex.common.types import PromptBundle, ToolSpec
from cortex.config.flags import PROMPT

BASE_SYSTEM_HEADER = """\
# Cortex Agent — Operating Principles

You are Cortex, a repository-aware, execution-validated assistant.
Follow the instructions below with high reliability.

{reinforce}
"""

TASK_MANAGEMENT = """\
## Task Management
- Maintain and update a clear TODO plan for complex tasks.
- When no plan exists and the task is multi-step, create one via the `todo.write` tool.
- Keep tasks specific, testable, and arranged with clear dependencies.
"""

TOOLS_POLICY = """\
## Tools Policy
- Prefer using tools to read files, run tests, and apply changes; do not guess contents.
- Always reflect the outcome of tool calls in your next step.
- If a tool result conflicts with prior assumptions, update your plan.
"""

SAFETY_POLICY = """\
## Safety
- Do not run shell commands unless explicitly allowed by the environment.
- Keep diffs and patches minimal and reversible.
"""

REMINDER_TAG = "system-reminder"


def build_system_prompt() -> str:
    reinforce = " ".join(f"**{kw}**." for kw in PROMPT.reinforce_keywords)
    sections = [
        BASE_SYSTEM_HEADER.format(reinforce=reinforce),
        TASK_MANAGEMENT,
        TOOLS_POLICY,
        SAFETY_POLICY,
    ]
    return "\n\n".join(s.strip() for s in sections)


def clamp(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


class ReminderEngine:
    """Injects just-in-time reminders based on agent state (history/tools/todos)."""

    def __init__(self):
        self.last_tool_step_idx = 0
        self.last_progress_time = datetime.utcnow()

    def observe_step(self, step_idx: int, used_tool: bool, made_progress: bool) -> None:
        if used_tool:
            self.last_tool_step_idx = step_idx
        if made_progress:
            self.last_progress_time = datetime.utcnow()

    def maybe_inject(self, todos_empty: bool, step_idx: int) -> str | None:
        if not PROMPT.enable_reminders:
            return None
        msgs: list[str] = []
        now = datetime.utcnow()
        # (1) Encourage plan if TODO empty
        if PROMPT.remind_if_todo_empty and todos_empty:
            msgs.append(
                "Your to-do list is currently empty. If you are working on a multi-step task, use the `todo.write` tool to create a plan with concrete subtasks."
            )
        # (2) Encourage tools if idle
        if step_idx - self.last_tool_step_idx >= PROMPT.remind_if_no_tools_used_steps:
            msgs.append(
                "You have not used any tools for several steps. Prefer reading files, running tests, or inspecting logs via tools instead of guessing."
            )
        # (3) Stalled time
        if (now - self.last_progress_time) >= timedelta(
            seconds=PROMPT.remind_if_stalled_secs
        ):
            msgs.append(
                "No visible progress recently. Summarize what you know and run a targeted probe (e.g., read a file or run a single test) to move forward."
            )
        if not msgs:
            return None
        body = "\n".join(f"- {m}" for m in msgs)
        return f"<{REMINDER_TAG}>\n{body}\n</{REMINDER_TAG}>"


def build_prompt_bundle(
    history: list[dict[str, Any]],
    tools: list[ToolSpec],
    context_text: str | None,
    reminders_text: str | None,
) -> PromptBundle:
    system = build_system_prompt()
    # filter+clamp
    hist = history[-100:]  # cheap cap by turns; upstream caller should token-trim
    ctx = clamp(context_text or "", PROMPT.max_context_chars)
    rem = clamp(reminders_text or "", 2000) if reminders_text else None
    return PromptBundle(
        system=system,
        tools=tools,
        history=hist,
        context=ctx or None,
        reminders=rem,
        metadata={"builder": "CortexPromptBuilder/v1"},
    )
