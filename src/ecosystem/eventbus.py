# src/ecosystem/eventbus.py
"""
Defines the event bus protocol for system-wide, decoupled communication.
Events allow for observability, metrics collection, and triggering of
asynchronous workflows without tightly coupling components.
"""
import json
from datetime import UTC, datetime
from typing import Any, Protocol


class IEventBus(Protocol):
    """Interface for an event bus client."""

    async def emit(self, topic: str, payload: dict[str, Any]) -> None: ...


class NoopEventBus(IEventBus):
    """A no-operation event bus that does nothing. Safe for default use."""

    async def emit(self, topic: str, payload: dict[str, Any]) -> None:
        pass  # Does nothing


class StdoutEventBus(IEventBus):
    """An event bus that prints all events to standard output as JSON lines."""

    def __init__(self):
        self.events: list[dict[str, Any]] = []

    async def emit(self, topic: str, payload: dict[str, Any]) -> None:
        """Emits an event by printing it to stdout."""
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "topic": topic,
            "payload": payload,
        }
        # Store for potential inspection in tests or simple scenarios
        self.events.append(event)
        print(json.dumps(event))
