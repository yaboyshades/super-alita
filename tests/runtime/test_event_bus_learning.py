import asyncio

import pytest

from src.reug_runtime.event_bus import ProductionEventBus


class _LearningEngine:
    def __init__(self) -> None:
        self.events = []

    async def process_learning_event(self, event):
        self.events.append(event)


@pytest.mark.asyncio
async def test_event_bus_routes_learning_events():
    bus = ProductionEventBus(log_dir=None)
    engine = _LearningEngine()
    bus.attach_learning_engine(engine)
    await bus.publish({"type": "AbilitySucceeded"})
    await asyncio.sleep(0.05)
    assert engine.events and engine.events[0]["type"] == "AbilitySucceeded"
