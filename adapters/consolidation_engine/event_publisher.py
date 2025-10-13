"""Event bus adapter for consolidation telemetry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from domain.consolidation_engine.models import ConsolidationEvent
from domain.consolidation_engine.service import ConsolidationEventPublisher


@runtime_checkable
class EventBusLike(Protocol):
    """Subset of the runtime event bus API consumed by the adapter."""

    async def publish(self, event: Mapping[str, Any]) -> Mapping[str, Any]:
        """Publish an event and return the enriched payload."""


class EventBusPublisherAdapter(ConsolidationEventPublisher):
    """Adapter bridging domain events onto the runtime bus."""

    def __init__(self, event_bus: EventBusLike) -> None:
        self._event_bus = event_bus

    async def publish(self, event: ConsolidationEvent) -> None:
        payload = {
            "event_type": event.event_type,
            "topic": event.topic,
            "payload": event.payload.model_dump(),
        }
        await self._event_bus.publish(payload)
