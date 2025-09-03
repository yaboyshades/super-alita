#!/usr/bin/env python3
"""Redis Event Bus Adapter - simplified version."""

import json
import logging
import os
from typing import Any, Optional

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

from src.core.event_bus import EventBus

logger = logging.getLogger(__name__)


class RedisEventBus(EventBus):
    """Redis-backed distributed event bus with graceful fallback."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.redis_client = None
        self._connected = False

        # Configuration
        self.redis_url = (
            self.config.get("redis_url") or
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )

    async def initialize(self) -> bool:
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using fallback")
            return True

        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            self._connected = True
            logger.info(f"Redis event bus connected: {self.redis_url}")
            return True
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            return True  # Fallback mode

    async def emit(self, event: dict[str, Any]) -> bool:
        """Emit event to Redis or fallback."""
        if self._connected and self.redis_client:
            try:
                channel = f"events:{event.get('type', 'unknown')}"
                message = json.dumps(event)
                await self.redis_client.publish(channel, message)
                return True
            except Exception as e:
                logger.error(f"Redis publish failed: {e}")

        # Fallback to parent implementation
        return await super().emit(event)

    async def disconnect(self) -> None:
        """Clean up Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
        self._connected = False


class EventBusAdapter:
    """Factory for creating event bus."""

    @staticmethod
    def create(config: Optional[dict[str, Any]] = None) -> EventBus:
        """Create event bus based on configuration."""
        config = config or {}
        use_redis = config.get("use_redis", True)

        if use_redis and REDIS_AVAILABLE:
            return RedisEventBus(config)

        # Fall back to in-memory
        from src.core.event_bus import InMemoryEventBus
        return InMemoryEventBus()
