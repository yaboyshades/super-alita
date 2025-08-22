#!/usr/bin/env python3
"""
Redis-based EventBus for inter-process communication
"""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional

import redis.asyncio as redis

from src.core.event_serializer import EventSerializer, EventSerializerError
from src.core.events import BaseEvent

logger = logging.getLogger(__name__)


class RedisEventBus:
    """
    Redis-based EventBus for distributed Super Alita components.

    Enables communication between the main agent and chat interface
    running in separate processes.
    """

    _instance: Optional["RedisEventBus"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RedisEventBus, cls).__new__(cls)
        return cls._instance

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._redis_host = host
        self._redis_port = port
        self._redis_db = db
        self._redis: redis.Redis | None = None
        self._pubsub: redis.client.PubSub | None = None
        self._serializer = EventSerializer()
        self._listeners: dict[str, list[Callable]] = defaultdict(list)
        self._listener_task: asyncio.Task | None = None
        self._running = False
        self._initialized = True

        logger.info(f"RedisEventBus initialized (host={host}, port={port}, db={db})")

    async def connect(self) -> None:
        """Connect to Redis server."""
        try:
            self._redis = redis.Redis(
                host=self._redis_host,
                port=self._redis_port,
                db=self._redis_db,
                decode_responses=False,
            )

            # Test connection
            await self._redis.ping()
            logger.info("Successfully connected to Redis server")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def start(self) -> None:
        """Start the event bus listener."""
        if not self._redis:
            await self.connect()

        if not self._running:
            self._running = True
            self._listener_task = asyncio.create_task(self._listener_loop())
            logger.info("RedisEventBus started")

    async def stop(self) -> None:
        """Stop the event bus and close connections."""
        self._running = False

        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._pubsub:
            await self._pubsub.close()

        if self._redis:
            await self._redis.close()

        logger.info("RedisEventBus stopped")

    async def emit(
        self, event_type: str, source_plugin: str = "unknown", **kwargs
    ) -> None:
        """
        Emit an event to Redis.

        Args:
            event_type: Type of event
            source_plugin: Plugin that emitted the event
            **kwargs: Event data
        """
        try:
            # Create event object
            event_data = {
                "event_type": event_type,
                "source_plugin": source_plugin,
                "timestamp": datetime.now(),
                **kwargs,
            }

            # Create BaseEvent object
            event = BaseEvent(**event_data)

            # Serialize and publish
            serialized_data = self._serializer.serialize(event)
            channel = f"alita:{event_type}"

            await self._redis.publish(channel, serialized_data)
            logger.debug(f"Published event '{event_type}' to Redis channel '{channel}'")

        except Exception as e:
            logger.error(f"Failed to emit event '{event_type}': {e}")

    async def subscribe(self, event_type: str, handler: Callable) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Type of event to subscribe to
            handler: Function to call when event is received
        """
        self._listeners[event_type].append(handler)
        logger.info(
            f"Subscribed to event type '{event_type}' (handler: {handler.__name__})"
        )

        # If we're already running, subscribe to the Redis channel
        if self._running and self._pubsub:
            channel = f"alita:{event_type}"
            await self._pubsub.subscribe(channel)
            logger.debug(f"Subscribed to Redis channel '{channel}'")

    async def _listener_loop(self) -> None:
        """Main listener loop for Redis pub/sub."""
        try:
            self._pubsub = self._redis.pubsub()

            # Subscribe to all channels we have listeners for
            for event_type in self._listeners:
                channel = f"alita:{event_type}"
                await self._pubsub.subscribe(channel)
                logger.debug(f"Subscribed to Redis channel '{channel}'")

            logger.info("Redis listener loop started")

            while self._running:
                try:
                    # Get message with timeout
                    message = await self._pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=1.0
                    )

                    if message and message["type"] == "message":
                        await self._handle_message(message)

                except TimeoutError:
                    continue  # Normal timeout, keep listening
                except Exception as e:
                    logger.error(f"Error in listener loop: {e}")
                    await asyncio.sleep(1)  # Brief pause before retrying

        except Exception as e:
            logger.error(f"Fatal error in listener loop: {e}")
        finally:
            if self._pubsub:
                await self._pubsub.close()

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle incoming Redis message."""
        try:
            channel = message["channel"].decode("utf-8")
            data = message["data"]

            # Extract event type from channel name
            if channel.startswith("alita:"):
                event_type = channel[6:]  # Remove 'alita:' prefix
            else:
                event_type = channel

            # Deserialize event
            try:
                event = self._serializer.deserialize(data)
            except EventSerializerError as e:
                logger.error(
                    f"Failed to deserialize event from channel '{channel}': {e}"
                )
                return

            # Call all handlers for this event type
            if event_type in self._listeners:
                for handler in self._listeners[event_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            # Run sync handler in executor
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, handler, event)
                    except Exception as e:
                        logger.error(f"Error in event handler {handler.__name__}: {e}")

        except Exception as e:
            logger.error(f"Error handling Redis message: {e}")


# Global instance
_global_redis_bus: RedisEventBus | None = None


async def get_global_redis_bus() -> RedisEventBus:
    """Get the global Redis EventBus instance."""
    global _global_redis_bus
    if _global_redis_bus is None:
        _global_redis_bus = RedisEventBus()
        await _global_redis_bus.connect()
    return _global_redis_bus
