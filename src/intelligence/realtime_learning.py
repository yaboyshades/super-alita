"""Real-time learning integration across collaborating agents.

The module operationalises ideas from joint attention and shared blackboard
systems (Lesser & Corkill, 1983) by streaming agent reasoning traces through a
lightweight publish/subscribe fabric.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

LearningHandler = Callable[[dict[str, Any]], Awaitable[None]]


class EventBusSubscriber(Protocol):
    """Protocol capturing the subset of the event bus used for subscriptions."""

    async def subscribe(
        self, event_type: str, handler: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Register a handler for the supplied event type."""


class RealtimeLearningEngine:
    """Facilitate on-the-fly knowledge sharing between concurrent agents."""

    def __init__(self, event_bus: EventBusSubscriber | None) -> None:
        self._event_bus = event_bus
        self._subscribers: dict[str, list[LearningHandler]] = defaultdict(list)
        self._active = False
        self._lock = asyncio.Lock()

    async def start_collective_learning(self) -> None:
        """Activate subscriptions so that reasoning traces trigger callbacks."""

        async with self._lock:
            if self._active:
                return
            self._active = True
        if not self._event_bus:
            logger.debug("Realtime learning engine started without event bus")
            return
        # We subscribe to coarse topics; ProductionEventBus will fan in events.
        for topic in ("agent_stream", "validation_stream", "memory_stream"):
            try:
                await self._event_bus.subscribe(topic, self._dispatch_learning_event)
            except Exception:  # pragma: no cover - defensive in case of mismatched APIs
                logger.debug("Event bus does not support topic '%s'", topic)

    def register_subscriber(self, channel: str, handler: LearningHandler) -> None:
        """Register downstream consumers for learning signals."""

        if not asyncio.iscoroutinefunction(handler):
            raise TypeError("handler must be an async callable")
        self._subscribers[channel].append(handler)

    async def process_learning_event(self, event: dict[str, Any]) -> None:
        """Entry point used by the event bus to deliver learning opportunities."""

        event_type = str(event.get("type", "")).lower()
        if not event_type:
            return
        if event_type.startswith("agent"):
            await self._handle_agent_event(event)
        elif event_type.startswith("validation"):
            await self._handle_validation_event(event)
        elif event_type.startswith("memory") or event_type.endswith("memory"):
            await self._handle_memory_event(event)

    async def _dispatch_learning_event(self, event: dict[str, Any]) -> None:
        """Fan-out helper used when subscribing directly to the event bus."""

        await self.process_learning_event(event)

    async def _handle_agent_event(self, event: dict[str, Any]) -> None:
        pattern = await self._extract_pattern(event, key="agent_id")
        await self._broadcast_learning("agent", pattern)

    async def _handle_validation_event(self, event: dict[str, Any]) -> None:
        pattern = await self._extract_pattern(event, key="validation")
        await self._broadcast_learning("validation", pattern)

    async def _handle_memory_event(self, event: dict[str, Any]) -> None:
        pattern = await self._extract_pattern(event, key="memory")
        await self._broadcast_learning("memory", pattern)

    async def _extract_pattern(self, event: dict[str, Any], key: str) -> dict[str, Any]:
        """Derive a lightweight learning pattern from an incoming event."""

        details = event.get(key)
        if isinstance(details, dict):
            return details
        if isinstance(details, list):
            return {"items": details}
        return {"value": details, "event": event.get("type")}

    async def _broadcast_learning(self, channel: str, payload: dict[str, Any]) -> None:
        """Send derived learning payload to registered subscribers."""

        subscribers = list(self._subscribers.get(channel, []))
        for handler in subscribers:
            try:
                await handler(payload)
            except Exception:
                logger.exception("Learning subscriber failed", extra={"channel": channel})
