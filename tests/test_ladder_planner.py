import asyncio
import pytest
from typing import Any, Dict

from cortex.planner import LadderPlanner
from cortex.todo import InMemoryTodoStore, TodoStatus


class FakeKG:
    def get_context_for_title(self, title: str) -> str:
        return f"context for {title}"

    def compute_energy_for_title(self, title: str) -> float:
        # simple stable pseudo-energy based on length
        return min(1.0, max(0.1, len(title) / 40.0))

    def write_decision(self, tool: str, node_id: str, reward: float) -> None:
        # could record to a list if needed
        pass

    def estimate_metric_delta(self, title: str) -> float:
        return 0.1  # pretend small positive impact


class FakeBandit:
    def __init__(self):
        self.updates = []

    def select_tool(self, context=None) -> str:
        return "codebase_search"

    def update(self, tool: str, reward: float) -> None:
        self.updates.append((tool, reward))


class FakeEventBus:
    def __init__(self):
        self.events = []

    async def emit(self, kind: str, **kwargs):
        self.events.append(("async", kind, kwargs))

    def emit_sync(self, kind: str, **kwargs):
        self.events.append(("sync", kind, kwargs))


class FakeOrchestrator:
    def __init__(self):
        self.event_bus = FakeEventBus()

    async def execute_action(self, tool: str, todo, context: str, shadow: bool = True) -> str:
        return f"Ran {tool} on {todo.title} (shadow={shadow})"


class UserEvent:
    def __init__(self, query: str, context: str = ""):
        self.payload = {"query": query, "context": context}


@pytest.mark.asyncio
async def test_ladder_planner_happy_path():
    kg = FakeKG()
    bandit = FakeBandit()
    store = InMemoryTodoStore()
    orch = FakeOrchestrator()

    planner = LadderPlanner(kg=kg, bandit=bandit, store=store, orchestrator=orch)
    evt = UserEvent(query="debug the API tests", context="previous conversation history...")
    root = await planner.plan_from_user_event(evt)

    # root created
    assert root.id
    assert root.title == "debug the API tests"
    # at least 3 children
    children = store.children_of(root.id)
    assert len(children) >= 3

    # dependencies form a DAG (no self-deps; trivial check)
    for c in children:
        assert c.id not in c.depends_on

    # execution marked done (shadow mode heuristic)
    done_count = sum(1 for c in children if c.status == TodoStatus.DONE)
    assert done_count >= 3

    # bandit got reward updates
    assert len(bandit.updates) >= 1

    # root completed
    refreshed_root = store.get(root.id)
    assert refreshed_root.status == TodoStatus.DONE

    # event bus saw key events
    kinds = [k for _, k, _ in orch.event_bus.events]
    assert "plan.decomposed" in kinds
    assert "plan.completed" in [k for _, k, _ in orch.event_bus.events]
