import asyncio

from src.reug_runtime.router import execute_turn
from tests.runtime.fakes import (
    FakeAbilityRegistry,
    FakeEventBus,
    FakeKG,
    FakeLLM,
)


def test_execute_turn_captures_knowledge_graph_artifacts() -> None:
    event_bus = FakeEventBus()
    registry = FakeAbilityRegistry()
    kg = FakeKG()
    model = FakeLLM()

    async def _run_turn() -> list[dict[str, object]]:
        collected: list[dict[str, object]] = []
        async for event in execute_turn(
            "hello knowledge graph",
            "kg-session",
            event_bus,
            registry,
            kg,
            model,
        ):
            collected.append(event)
        return collected

    events = asyncio.run(_run_turn())

    # Router still concludes turn successfully.
    assert any(evt["type"] == "TaskSucceeded" for evt in events)

    # Knowledge graph context is retrieved and surfaced via telemetry.
    ctx_events = [
        evt for evt in event_bus.events if evt["type"] == "KnowledgeContextRetrieved"
    ]
    assert ctx_events, "expected a KnowledgeContextRetrieved event"
    assert ctx_events[0]["session_id"] == "kg-session"

    # Final answer is stored as an atom and connected back to the goal.
    atom_events = [
        evt for evt in event_bus.events if evt["type"] == "KnowledgeAtomCreated"
    ]
    assert atom_events and atom_events[0]["atom_type"] == "final_answer"
    assert any(bond["type"] == "ANSWERED" for bond in kg.bonds)
