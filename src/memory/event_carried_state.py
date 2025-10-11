"""Utilities for propagating agent state via the runtime event bus.

This module provides a lightweight façade that accepts structured
``StateChange`` updates, wraps them in ``AgentStateEvent`` payloads and
publishes them through the configured event bus. The events use a
standard shape so downstream subscribers can synchronise their local
registries when agent state changes occur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class StateChange:
    """Represents a change to an agent-scoped state attribute."""

    agent_id: str
    state_key: str
    value: Any
    previous_value: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    change_id: str = field(default_factory=lambda: str(uuid4()))

    def to_payload(self) -> dict[str, Any]:
        """Serialise the state change for inclusion in an event payload."""

        payload: dict[str, Any] = {
            "change_id": self.change_id,
            "agent_id": self.agent_id,
            "state_key": self.state_key,
            "value": self.value,
        }
        if self.previous_value is not None:
            payload["previous_value"] = self.previous_value
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class AgentStateEvent:
    """Envelope for propagating ``StateChange`` payloads via the event bus."""

    change: StateChange
    source: str
    correlation_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    event_type: str = "AgentStateEvent"
    kind: str = "agent_state"
    priority: str = "medium"

    def to_payload(self) -> dict[str, Any]:
        """Convert the event to a serialisable payload."""

        payload: dict[str, Any] = {
            "type": self.event_type,
            "event_type": self.event_type,
            "kind": self.kind,
            "priority": self.priority,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.change.agent_id,
            "correlation_id": self.correlation_id,
            "change": self.change.to_payload(),
        }
        return {k: v for k, v in payload.items() if v is not None}


class EventCarriedState:
    """Publish state changes as events to keep distributed caches in sync."""

    def __init__(self, event_bus: Any, source: str = "memory.event_carried_state"):
        self._event_bus = event_bus
        self._source = source

    async def propagate_agent_state(self, change: StateChange) -> dict[str, Any]:
        """Publish the change via the event bus with medium priority."""

        event = AgentStateEvent(change=change, source=self._source)
        payload = event.to_payload()

        publisher = getattr(self._event_bus, "publish", None)
        if callable(publisher):
            try:
                await publisher(payload, priority=event.priority)
            except TypeError:
                await publisher(payload)
        else:  # pragma: no cover - best effort fallback
            await self._event_bus.emit(payload)

        return payload


__all__ = ["StateChange", "AgentStateEvent", "EventCarriedState"]
