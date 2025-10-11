"""Regression tests for the A2A agent communication protocol."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.agents.communication import A2AProtocol


class FakeEventBus:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.priorities: list[str | None] = []

    async def publish(self, event: dict[str, object], priority: str | None = None) -> dict[str, object]:
        self.events.append(event)
        self.priorities.append(priority)
        return event


class EmitOnlyBus:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def emit(self, event: dict[str, object]) -> dict[str, object]:
        self.events.append(event)
        return event


@pytest.mark.asyncio
async def test_agent_to_agent_emits_expected_envelope() -> None:
    bus = FakeEventBus()
    protocol = A2AProtocol(bus)

    payload = {"content": "status update"}
    event = await protocol.agent_to_agent(
        "agent-alpha",
        "agent-beta",
        "status",
        payload,
        priority="high",
        correlation_id="corr-123",
    )

    assert bus.priorities == ["high"]
    assert len(bus.events) == 1
    emitted = bus.events[0]
    assert emitted["event_type"] == "agent_message"
    assert emitted["protocol_version"] == "a2a-1.0"
    assert emitted["sender_id"] == "agent-alpha"
    assert emitted["recipient_id"] == "agent-beta"
    assert emitted["message_type"] == "status"
    assert emitted["payload"] == payload
    assert emitted["priority"] == "high"
    assert emitted["correlation_id"] == "corr-123"

    security = emitted["security_context"]
    assert security["issuer"] == "agent-alpha"
    assert security["audience"] == "agent-beta"
    assert security["correlation_id"] == "corr-123"
    assert "nonce" in security
    # Should be ISO formatted timestamp
    datetime.fromisoformat(security["issued_at"])


@pytest.mark.asyncio
async def test_agent_to_agent_falls_back_to_emit_and_includes_claims() -> None:
    bus = EmitOnlyBus()
    protocol = A2AProtocol(bus)
    metadata = {"claims": {"scope": "internal"}}

    event = await protocol.agent_to_agent(
        "agent-gamma",
        "agent-delta",
        "notify",
        {"detail": "ready"},
        metadata=metadata,
    )

    assert bus.events and bus.events[0] == event
    emitted = bus.events[0]
    assert emitted["priority"] == "medium"
    security = emitted["security_context"]
    assert security["claims"] == metadata["claims"]
