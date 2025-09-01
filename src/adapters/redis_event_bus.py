#!/usr/bin/env python3
"""
Redis Event Bus Adapter for Super Alita

Provides distributed event pub/sub capabilities using Redis as the message
broker. Falls back gracefully to in-memory event bus when Redis is unavailable.
"""

import asyncio
import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

try:
    import redis.asyncio as redis
    from redis.asyncio import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore
    ConnectionPool = None  # type: ignore
    REDIS_AVAILABLE = False

from src.core.event_bus import EventBus, EventHandler

logger = logging.getLogger(__name__)


class RedisEventBus(EventBus):
    """Redis-backed distributed event bus with graceful fallback."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize Redis event bus with connection pooling.

        Args:
            config: Redis configuration with keys:
                - redis_url: Redis connection URL (default: redis://localhost:6379)
                - redis_host: Redis host (default: localhost)
                - redis_port: Redis port (default: 6379)
                - redis_db: Redis database number (default: 0)
                - redis_password: Redis password (optional)
                - max_connections: Connection pool size (default: 10)
                - retry_on_failure: Retry on connection failures (default: True)
                - fallback_to_memory: Use in-memory when unavailable (default: True)
        """
        super().__init__()
        self.config = config or {}
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self._subscribers: dict[str, list[EventHandler]] = {}
        self._background_tasks: list[asyncio.Task] = []
        self._connected = False
        self._fallback_handlers: dict[str, list[EventHandler]] = {}

        # Configuration
        self.redis_url = self.config.get("redis_url") or os.getenv("REDIS_URL")
        self.redis_host = self.config.get(
            "redis_host", os.getenv("REDIS_HOST", "localhost")
        )
        self.redis_port = int(
            self.config.get("redis_port", os.getenv("REDIS_PORT", 6379))
        )
        self.redis_db = int(
            self.config.get("redis_db", os.getenv("REDIS_DB", 0))
        )
        self.redis_password = self.config.get(
            "redis_password"
        ) or os.getenv("REDIS_PASSWORD")
        self.max_connections = int(self.config.get("max_connections", 10))
        self.retry_on_failure = self.config.get("retry_on_failure", True)
        self.fallback_to_memory = self.config.get("fallback_to_memory", True)

        # Event channel prefix for namespacing
        self.channel_prefix = self.config.get("channel_prefix", "super_alita:")

    async def initialize(self) -> bool:
        """Initialize Redis connection and start background listeners."""
        if not REDIS_AVAILABLE:
            logger.warning(
                "Redis not available, falling back to in-memory event bus"
            )
            return self.fallback_to_memory

        try:
            # Create connection pool
            if self.redis_url:
                pool = ConnectionPool.from_url(
                    self.redis_url,
                    max_connections=self.max_connections,
                    retry_on_failure=self.retry_on_failure
                )
            else:
                pool = ConnectionPool(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=self.redis_db,
                    password=self.redis_password,
                    max_connections=self.max_connections,
                    retry_on_failure=self.retry_on_failure
                )

            self.redis_client = redis.Redis(connection_pool=pool)

            # Test connection
            await self.redis_client.ping()
            self._connected = True

            # Initialize pub/sub
            self.pubsub = self.redis_client.pubsub()

            logger.info(
                f"Redis event bus connected to {self.redis_host}:{self.redis_port}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            if self.fallback_to_memory:
                logger.info("Falling back to in-memory event bus")
                return True
            return False

    async def subscribe(self, event_type: str, handler: EventHandler) -> bool:
        """Subscribe to events of a specific type."""
        if not self._connected and not self.fallback_to_memory:
            return False

        # Add to local subscribers registry
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

        if self._connected and self.pubsub:
            try:
                channel = f"{self.channel_prefix}{event_type}"
                await self.pubsub.subscribe(channel)

                # Start listener task if not already running
                if not any(task.get_name() == f"redis_listener_{event_type}"
                          for task in self._background_tasks if not task.done()):
                    task = asyncio.create_task(
                        self._listen_to_channel(channel),
                        name=f"redis_listener_{event_type}"
                    )
                    self._background_tasks.append(task)

                logger.debug(f"Subscribed to Redis channel: {channel}")
                return True

            except Exception as e:
                logger.error(
                    f"Failed to subscribe to Redis channel {event_type}: {e}"
                )
                if self.fallback_to_memory:
                    # Store in fallback handlers
                    if event_type not in self._fallback_handlers:
                        self._fallback_handlers[event_type] = []
                    self._fallback_handlers[event_type].append(handler)
                    return True
                return False
        else:
            # In-memory fallback mode
            if event_type not in self._fallback_handlers:
                self._fallback_handlers[event_type] = []
            self._fallback_handlers[event_type].append(handler)
            return True

    async def publish(self, event: dict[str, Any]) -> bool:
        """Publish an event to the distributed event bus."""
        event_type = event.get("type", "unknown")

        if not self._connected and not self.fallback_to_memory:
            return False

        # Add event metadata
        if "id" not in event:
            event["id"] = str(uuid4())
        if "timestamp" not in event:
            import datetime
            event["timestamp"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()

        if self._connected and self.redis_client:
            try:
                channel = f"{self.channel_prefix}{event_type}"
                message = json.dumps(event)
                await self.redis_client.publish(channel, message)
                logger.debug(
                    f"Published event to Redis channel {channel}: {event.get('id')}"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to publish to Redis: {e}")
                if self.fallback_to_memory:
                    return await self._publish_to_memory(event)
                return False
        else:
            # In-memory fallback
            return await self._publish_to_memory(event)

    async def emit(self, event: dict[str, Any]) -> bool:
        """Alias for publish() to maintain EventBus interface."""
        return await self.publish(event)

    async def _listen_to_channel(self, channel: str) -> None:
        """Background task to listen for messages on a Redis channel."""
        if not self.pubsub:
            return

        try:
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    try:
                        event = json.loads(message["data"])
                        event_type = event.get("type", "unknown")

                        # Deliver to local subscribers
                        if event_type in self._subscribers:
                            for handler in self._subscribers[event_type]:
                                try:
                                    if asyncio.iscoroutinefunction(handler):
                                        await handler(event)
                                    else:
                                        handler(event)
                                except Exception as e:
                                    logger.error(f"Error in event handler for {event_type}: {e}")

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode event message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")

        except asyncio.CancelledError:
            logger.debug(f"Redis listener for {channel} cancelled")
        except Exception as e:
            logger.error(f"Redis listener error for {channel}: {e}")

    async def _publish_to_memory(self, event: Dict[str, Any]) -> bool:
        """Fallback to in-memory event delivery."""
        event_type = event.get("type", "unknown")

        # Deliver to fallback handlers
        handlers = self._fallback_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in fallback handler for {event_type}: {e}")

        return len(handlers) > 0

    async def disconnect(self) -> None:
        """Clean up Redis connections and background tasks."""
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Close Redis connections
        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()

        if self.redis_client:
            await self.redis_client.close()

        self._connected = False
        logger.info("Redis event bus disconnected")

    async def health_check(self) -> Dict[str, Any]:
        """Check Redis connection health."""
        if not self._connected:
            return {
                "status": "disconnected",
                "redis_available": REDIS_AVAILABLE,
                "fallback_mode": self.fallback_to_memory
            }

        if self.redis_client:
            try:
                await self.redis_client.ping()
                return {
                    "status": "connected",
                    "redis_available": True,
                    "subscribers": len(self._subscribers),
                    "fallback_handlers": len(self._fallback_handlers)
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "fallback_mode": self.fallback_to_memory
                }

        return {
            "status": "fallback",
            "fallback_handlers": len(self._fallback_handlers)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            "connected": self._connected,
            "redis_available": REDIS_AVAILABLE,
            "subscribers": {event_type: len(handlers)
                          for event_type, handlers in self._subscribers.items()},
            "fallback_handlers": {event_type: len(handlers)
                                for event_type, handlers in self._fallback_handlers.items()},
            "background_tasks": len([t for t in self._background_tasks if not t.done()]),
            "config": {
                "host": self.redis_host,
                "port": self.redis_port,
                "db": self.redis_db,
                "max_connections": self.max_connections
            }
        }


class EventBusAdapter:
    """Factory for creating appropriate event bus implementation."""

    @staticmethod
    def create(config: Optional[Dict[str, Any]] = None) -> EventBus:
        """Create event bus based on configuration and availability."""
        config = config or {}

        # Check if Redis is preferred and available
        use_redis = config.get("use_redis", True)
        if use_redis and REDIS_AVAILABLE:
            return RedisEventBus(config)

        # Fall back to in-memory implementation
        logger.info("Using in-memory event bus")
        from src.core.event_bus import InMemoryEventBus
        return InMemoryEventBus()
