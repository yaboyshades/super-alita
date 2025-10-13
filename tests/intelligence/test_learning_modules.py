import asyncio
from typing import Any

import pytest

from src.intelligence import IntelligenceConsolidator, RealtimeLearningEngine


class _FakeEventBus:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def emit(self, event: dict[str, Any]) -> dict[str, Any]:
        self.events.append(event)
        return event

    async def subscribe(self, event_type: str, handler):  # pragma: no cover - optional
        self._handler = handler


class _FakeProcessor:
    async def extract_patterns(self, session_id: str, outcome: dict[str, Any]):
        return [{"session": session_id, "outcome": outcome.get("success") }]


class _FakeACE:
    async def evolve_from_patterns(self, patterns, validation_feedback):
        return {"patterns": patterns, "validation": validation_feedback}


class _FakeValidator:
    async def validate_outcome(self, outcome):
        return {"review": outcome.get("success")}


class _FakeMemory:
    async def store_context(self, context):
        return {"stored": True, "context": context}


@pytest.mark.asyncio
async def test_consolidator_emits_learning_event():
    bus = _FakeEventBus()
    consolidator = IntelligenceConsolidator(
        ace_evolver=_FakeACE(),
        event_processor=_FakeProcessor(),
        validator=_FakeValidator(),
        memory_stack=_FakeMemory(),
        event_bus=bus,
    )
    outcome = await consolidator.consolidate_interaction(
        session_id="s-1",
        interaction_outcome={"success": True, "validation": {"foo": "bar"}},
    )
    assert outcome.patterns[0]["session"] == "s-1"
    assert outcome.evolved_context["validation"]["review"] is True
    assert bus.events[-1]["type"] == "intelligence_evolved"


@pytest.mark.asyncio
async def test_realtime_learning_engine_broadcasts_patterns():
    bus = _FakeEventBus()
    engine = RealtimeLearningEngine(bus)
    received: list[dict[str, Any]] = []

    async def _handler(payload: dict[str, Any]) -> None:
        received.append(payload)

    engine.register_subscriber("agent", _handler)
    await engine.start_collective_learning()
    await engine.process_learning_event({"type": "agent_reasoning", "agent_id": "alpha"})
    assert received and received[0]["value"] == "alpha"
