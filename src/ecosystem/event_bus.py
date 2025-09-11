"""
A simple, in-memory, asynchronous event bus implementation for the
ecosystem module. Provides publish/subscribe semantics with fire-and-
forget dispatch to decouple publishers from subscribers.

Note: This event bus is intentionally lightweight and scoped to the
ecosystem package. For broader system integrations, prefer the
`src/core` EventBus implementations.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

# Import interface from orchestrator to avoid duplicating protocols.
# This import placement avoids circular dependencies since the orchestrator
# does not import this concrete bus.
from .master_orchestrator import IEventBus  # type: ignore


class InMemoryEventBus(IEventBus):
    """In-memory pub/sub bus using async handler dispatch.

    - Handlers are stored per concrete event type (class-based).
    - `publish` schedules handlers via `asyncio.create_task` (fire-and-forget).
    - Suitable for tests and local development.
    """

    def __init__(self) -> None:
        self._handlers: defaultdict[
            type, list[Callable[[Any], Awaitable[None]]]
        ] = defaultdict(list)

    def subscribe(self, event_type: type, handler: Callable[[Any], Awaitable[None]]) -> None:
        """Subscribes an async handler to a specific event type."""
        self._handlers[event_type].append(handler)

    def publish(self, event: Any) -> None:
        """Publishes an event to all handlers subscribed to its type."""
        event_type: type = type(event)
        for handler in self._handlers.get(event_type, []):
            asyncio.create_task(handler(event))

