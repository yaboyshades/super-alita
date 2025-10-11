from __future__ import annotations

import asyncio

import pytest

from src.memory.event_carried_state import (
    EventCarriedState,
    StateChange,
)
from src.reug_runtime.event_bus import InMemoryPubSubEventBus


@pytest.mark.asyncio
async def test_propagate_agent_state_serializes_change() -> None:
    bus = InMemoryPubSubEventBus(log_dir=None)
    ecs = EventCarriedState(bus, source="memory.tests")

    change = StateChange(
        agent_id="agent-1",
        state_key="mode",
        value="planning",
        previous_value="idle",
        metadata={"confidence": 0.9},
    )

    payload = await ecs.propagate_agent_state(change)

    assert payload["event_type"] == "AgentStateEvent"
    assert payload["priority"] == "medium"
    assert payload["source"] == "memory.tests"
    assert payload["agent_id"] == "agent-1"
    assert payload["change"]["state_key"] == "mode"
    assert payload["change"]["value"] == "planning"
    assert payload["change"]["previous_value"] == "idle"
    assert payload["change"]["metadata"] == {"confidence": 0.9}


@pytest.mark.asyncio
async def test_propagated_events_update_cache_and_notify_subscribers() -> None:
    bus = InMemoryPubSubEventBus(log_dir=None)
    ecs = EventCarriedState(bus)

    observed: list[dict[str, object]] = []

    async def handler(event: dict[str, object]) -> None:
        observed.append(event)

    await bus.subscribe("AgentStateEvent", handler)

    change = StateChange(agent_id="agent-7", state_key="status", value={"step": 1})

    await ecs.propagate_agent_state(change)
    await asyncio.sleep(0)  # allow publish to dispatch asynchronously

    assert len(observed) == 1
    event = observed[0]
    assert event["agent_id"] == "agent-7"
    assert bus.get_agent_state("agent-7")["status"]["value"] == {"step": 1}
