from __future__ import annotations
from typing import Any, Dict
from cortex.planner import LadderPlanner
from cortex.todo import InMemoryTodoStore


class _NoopEventBus:
    async def emit(self, kind: str, **kwargs) -> None:
        return None

    def emit_sync(self, kind: str, **kwargs) -> None:
        return None


class _DefaultOrchestrator:
    """Lightweight default orchestrator stub if none exists; replace with real impl."""

    def __init__(self, kg, bandit, event_bus=None):
        self.kg = kg
        self.bandit = bandit
        self.event_bus = event_bus or _NoopEventBus()

    async def execute_action(self, tool: str, todo, context: str, shadow: bool = True) -> str:
        # Replace with your tool runner; this is a placeholder
        return f"Simulated run of {tool} on {todo.title}"


async def handle_user_event(kg, bandit, user_event, orchestrator=None) -> Dict[str, Any]:
    """
    Drop-in function to route a user event into the LADDER planner.
    Returns minimal planning result (root id and child ids).
    """
    orch = orchestrator or _DefaultOrchestrator(kg=kg, bandit=bandit)
    store = InMemoryTodoStore()
    planner = LadderPlanner(kg=kg, bandit=bandit, store=store, orchestrator=orch)
    root = await planner.plan_from_user_event(user_event)
    return {"root_todo_id": root.id, "children": root.children_ids}
