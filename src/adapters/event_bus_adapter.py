"""Event bus adapter providing pluggable backend modes.

Modes (EVENT_BUS_MODE env var):
  local  - existing FileEventBus / in-memory semantics
  redis  - Redis pub/sub cluster (future; optional dependency)
  kafka  - Kafka/Pulsar style (placeholder stub)
  hybrid - Try distributed first, fallback to local

The adapter exposes a minimal subset publish/subscribe used by current
code. Additional methods can be forwarded transparently as needed.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

try:  # optional redis import
    from redis.asyncio import Redis  # type: ignore
except Exception:  # pragma: no cover
    Redis = None  # type: ignore

from reug_runtime.event_bus import BaseEventBus, FileEventBus  # type: ignore

logger = logging.getLogger(__name__)


class EventBusMode:
    LOCAL = "local"
    REDIS = "redis"
    KAFKA = "kafka"
    HYBRID = "hybrid"


class EventBusAdapter(BaseEventBus):  # type: ignore[misc]
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.mode = os.getenv("EVENT_BUS_MODE", EventBusMode.LOCAL).lower()
        self.local_bus = FileEventBus(self.config)
        self._redis: Redis | None = None
        self._pubsub = None

    async def initialize(self) -> None:  # type: ignore[override]
        await self.local_bus.initialize()
        if self.mode in (EventBusMode.REDIS, EventBusMode.HYBRID):
            if Redis is None:
                logger.warning("Redis not installed; falling back to local event bus")
                return
            try:
                host = self.config.get("redis_host", "localhost")
                port = int(self.config.get("redis_port", 6379))
                self._redis = Redis(host=host, port=port, decode_responses=True)
                await self._redis.ping()
                self._pubsub = self._redis.pubsub()
                logger.info("Redis event bus connected")
            except Exception as e:  # pragma: no cover
                logger.warning(f"Redis connection failed: {e}; using local fallback")
                self._redis = None

    async def publish(self, event_type: str, **kwargs: Any) -> str:  # type: ignore[override]
        # Always publish locally (ensures existing behavior)
        event_id = await self.local_bus.publish(event_type, **kwargs)
        # Try distributed
        if self._redis is not None:
            try:
                await self._redis.publish(f"sa:{event_type}", str(kwargs))
            except Exception as e:  # pragma: no cover
                logger.debug(f"Redis publish failed: {e}")
        return event_id

    async def subscribe(self, event_type: str, callback: Callable[[dict[str, Any]], Awaitable[None]]):  # type: ignore[override]
        await self.local_bus.subscribe(event_type, callback)
        if self._redis is not None:
            try:
                await self._pubsub.subscribe(f"sa:{event_type}")  # type: ignore[arg-type]
            except Exception as e:  # pragma: no cover
                logger.debug(f"Redis subscribe failed: {e}")

    async def close(self) -> None:  # optional cleanup
        try:
            if self._pubsub is not None:
                await self._pubsub.close()
            if self._redis is not None:
                await self._redis.close()
        except Exception:  # pragma: no cover
            pass
        await getattr(self.local_bus, "close", lambda: None)()
