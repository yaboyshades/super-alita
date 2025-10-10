"""Redis-backed EventBus abstractions (placeholder)."""

from __future__ import annotations

from typing import Any

from .events import UnifiedEvent


class RedisEventBus:
    """Contract for the unified EventBus; methods are TBD during GREEN phase."""

    def __init__(self, *, channel: str, redis: Any) -> None:
        self.channel = channel
        self._redis = redis

    async def publish(self, event: UnifiedEvent) -> None:
        raise NotImplementedError(
            "RedisEventBus.publish pending implementation"
        )

    async def subscribe(self, *, timeout: float | None = None) -> UnifiedEvent:
        raise NotImplementedError(
            "RedisEventBus.subscribe pending implementation"
        )

    async def emit_metrics(self, metrics: dict[str, Any]) -> None:
        """Hook for metrics collection; implemented in GREEN phase."""
        raise NotImplementedError(
            "RedisEventBus.emit_metrics pending implementation"
        )

    @classmethod
    def from_url(cls, url: str, *, channel: str) -> RedisEventBus:
        raise NotImplementedError(
            "RedisEventBus.from_url pending implementation"
        )
