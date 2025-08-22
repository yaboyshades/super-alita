from __future__ import annotations

from typing import Any

from cortex.adapters.leanrag_adapter import build_situation_brief
from cortex.config.flags import FLAGS, PROMPT
from cortex.config.flags import LEANRAG as LR_FLAGS
from cortex.orchestrator.ladder_adapter import handle_user_event as ladder_handle
from cortex.proxy import ReminderEngine, build_prompt_bundle
from cortex.tools import ALL_TOOL_SPECS


def _complexity_score(query: str, context: str = "") -> float:
    """
    Estimate complexity score for routing decision.
    Higher score = more complex, should go to LADDER.
    """
    factors = []

    # Query length factor
    factors.append(min(len(query.split()) / 50.0, 0.3))

    # Keyword complexity factors
    complex_keywords = [
        "implement",
        "design",
        "architecture",
        "refactor",
        "optimize",
        "complex",
        "multiple",
        "integrate",
        "system",
        "workflow",
        "dependencies",
        "requirements",
        "analysis",
        "planning",
    ]

    simple_keywords = [
        "show",
        "list",
        "find",
        "what",
        "where",
        "when",
        "simple",
        "quick",
        "help",
        "status",
        "check",
    ]

    text = (query + " " + context).lower()

    complex_count = sum(1 for kw in complex_keywords if kw in text)
    simple_count = sum(1 for kw in simple_keywords if kw in text)

    factors.append(complex_count * 0.1)
    factors.append(-simple_count * 0.05)

    # Context complexity
    if context:
        factors.append(min(len(context.split()) / 100.0, 0.2))

    score = sum(factors)
    return min(score, 1.0)


async def route_user_event(kg, bandit, user_event, orchestrator=None) -> dict[str, Any]:
    """
    Route user events to appropriate handler based on complexity.
    Integrates LeanRAG situation briefs for complex requests.
    """
    query = str(user_event.get("query", ""))
    context = str(user_event.get("context", ""))

    # Calculate complexity score
    score = _complexity_score(query, context)

    if FLAGS.use_ladder_router and score >= 0.5:
        # Optionally enrich with LeanRAG brief for the planner prompt context
        if LR_FLAGS.enable:
            brief = build_situation_brief(kg.graph, query, flags=LR_FLAGS)
            if orchestrator and hasattr(orchestrator, "event_bus"):
                orchestrator.event_bus.emit_sync("context.leanrag.brief", payload=brief)
            # Attach to event context for downstream prompt builder
            if brief.get("enabled") and brief.get("brief"):
                user_event.context = (
                    (user_event.context or "") + "\n\nContext:\n" + brief["brief"]
                )

        # Prepare prompt bundle with tools and JIT reminders
        if PROMPT.enable_prompt_builder:
            rem_engine = getattr(orchestrator, "_rem_engine", None) or ReminderEngine()
            orchestrator._rem_engine = rem_engine
            todos_empty = (
                getattr(orchestrator, "todo_store", None) is not None
                and not orchestrator.todo_store.items
            )
            reminders = rem_engine.maybe_inject(
                todos_empty=todos_empty, step_idx=getattr(orchestrator, "step_idx", 0)
            )
            bundle = build_prompt_bundle(
                history=orchestrator.history,
                tools=ALL_TOOL_SPECS,
                context_text=user_event.context,
                reminders_text=reminders,
            )
            orchestrator.last_prompt_bundle = bundle

        # Complex: go through the LADDER pipeline
        return await ladder_handle(
            kg=kg, bandit=bandit, user_event=user_event, orchestrator=orchestrator
        )

    # Simple: return a lightweight plan skeleton (no side effects)
    return {
        "mode": "simple",
        "suggested_next_actions": [
            {"title": "Search codebase for keywords", "reason": "quick triage"},
            {"title": "Open related tests", "reason": "localize failure"},
        ],
        "complexity_score": score,
    }
