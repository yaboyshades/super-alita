import pytest

from src.intelligence import IntelligenceConsolidator
from src.memory import ACEvolver, LearningMemoryStack
from src.reug_runtime.loop import initialize_learning_stack


class _StubEventBus:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def emit(self, event: dict[str, Any]) -> dict[str, Any]:
        self.events.append(event)
        return event


@pytest.mark.asyncio
async def test_initialize_learning_stack_wires_components() -> None:
    bus = _StubEventBus()
    consolidator = initialize_learning_stack(bus)
    assert isinstance(consolidator, IntelligenceConsolidator)
    assert isinstance(consolidator._ace_evolver, ACEvolver)
    assert isinstance(consolidator._memory_stack, LearningMemoryStack)

    outcome = await consolidator.consolidate_interaction(
        session_id="session-123",
        interaction_outcome={
            "success": True,
            "validation": {"score": 0.9},
            "patterns": [{"type": "alignment", "weight": 1.0}],
        },
    )
    assert outcome.validation_feedback["approved"] is True
    assert outcome.evolved_context["status"] == "approved"
    event_types = {event.get("type") for event in bus.events}
    assert "intelligence_evolved" in event_types
    assert "MemoryContextStored" in event_types
    assert consolidator._memory_stack.snapshot(), "expected memory stack to persist contexts"
