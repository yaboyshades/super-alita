"""
Simple in-memory event bus for testing and demos.
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from .events import BaseEvent

logger = logging.getLogger(__name__)


class InMemoryEventBus:
    """
    Simple in-memory event bus for testing and demos.

    This provides a lightweight EventBus implementation that doesn't
    require Redis/Memurai for basic functionality.
    """

    def __init__(self):
        self._handlers: dict[str, list[Callable]] = {}
        self._is_running = False

    async def start(self) -> None:
        """Start the event bus."""
        self._is_running = True
        logger.info("✅ InMemoryEventBus started")

    async def stop(self) -> None:
        """Stop the event bus."""
        self._is_running = False
        logger.info("✅ InMemoryEventBus stopped")

    async def subscribe(
        self,
        event_type: str,
        callback: Callable[[BaseEvent], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(callback)
        logger.debug(f"✅ Subscribed to {event_type}")

    async def emit(
        self, event_type: str, source_plugin: str = "unknown", **kwargs: Any
    ) -> None:
        """Emit an event."""
        if not self._is_running:
            return

        # Create event
        event_data = {
            "event_type": event_type,
            "source_plugin": source_plugin,
            **kwargs,
        }

        # Create BaseEvent-like object
        event = BaseEvent(**event_data)

        # Call handlers
        handlers = self._handlers.get(event_type, [])
        if handlers:
            logger.debug(
                f"📤 Emitting {event_type} to {len(handlers)} handlers"
            )
            # Run handlers concurrently
            await asyncio.gather(
                *[handler(event) for handler in handlers],
                return_exceptions=True,
            )

    @property
    def is_running(self) -> bool:
        """Check if the event bus is running."""
        return self._is_running
