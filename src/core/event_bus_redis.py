#!/usr/bin/env python3
"""
Neural Event Bus for Super Alita Agent (Redis-Powered)
Provides async pub-sub communication between plugins with semantic routing
across multiple processes using Redis as the neural backbone.

This preserves all neural/semantic intelligence while enabling inter-process communication.
The "neural" aspects (embeddings, semantic filtering) are preserved in the messages
and subscriber intelligence, while Redis provides the distributed nervous system.
"""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import redis.asyncio as redis

from src.core.event_serializer import EventSerializer, EventSerializerError
from src.core.events import BaseEvent, create_event

logger = logging.getLogger(__name__)


class EventSubscription:
    """
    Neural subscription to an event type with semantic filtering capabilities.

    This preserves all the neural intelligence from the original implementation
    while working with Redis transport.
    """

    def __init__(
        self,
        event_type: str,
        handler: Callable,
        subscriber_id: str = None,
        semantic_filter: np.ndarray | None = None,
        semantic_threshold: float = 0.7,
        filter_func: Callable | None = None,
    ):
        self.event_type = event_type
        self.handler = handler
        self.subscriber_id = subscriber_id or f"sub_{id(handler)}"
        self.semantic_filter = semantic_filter
        self.semantic_threshold = semantic_threshold
        self.filter_func = filter_func
        self.is_active = True
        self.call_count = 0
        self.last_called = None

    async def matches_event(self, event: BaseEvent) -> bool:
        """
        Neural event matching with semantic intelligence.

        This is where the "neural" magic happens - semantic similarity
        calculations and intelligent filtering.
        """

        # Check event type
        if event.event_type != self.event_type:
            return False

        # Apply custom filter
        if self.filter_func:
            if asyncio.iscoroutinefunction(self.filter_func):
                if not await self.filter_func(event):
                    return False
            elif not self.filter_func(event):
                return False

        # Apply semantic filter (THE NEURAL PART!)
        if self.semantic_filter is not None and event.embedding is not None:
            # Convert embedding to numpy array if needed
            if isinstance(event.embedding, list):
                event_embedding = np.array(event.embedding)
            else:
                event_embedding = event.embedding

            # Calculate semantic similarity (cosine similarity)
            similarity = np.dot(self.semantic_filter, event_embedding)
            if similarity < self.semantic_threshold:
                return False

        return True

    async def handle_event(self, event: BaseEvent) -> None:
        """Handle the event with this neural subscription."""

        try:
            self.call_count += 1
            self.last_called = datetime.now(UTC)

            if asyncio.iscoroutinefunction(self.handler):
                await self.handler(event)
            else:
                # Run sync handler in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.handler, event)

        except Exception as e:
            logger.error(f"Error in neural event handler {self.subscriber_id}: {e}")

    def deactivate(self) -> None:
        """Deactivate this neural subscription."""
        self.is_active = False


class RedisEventBus:
    """
    Neural Event Bus with Redis-powered distributed nervous system.

    This maintains ALL neural/semantic capabilities from the original implementation
    while adding inter-process communication through Redis. The "neural" intelligence
    is preserved in:
    1. Event embeddings (carried in messages)
    2. Semantic filtering (in subscriptions)
    3. Intelligent routing (based on meaning, not just type)

    Redis serves as the distributed nervous system that connects all brain components.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RedisEventBus, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        max_workers: int = 4,
    ):
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Redis connection for distributed nervous system
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._redis: redis.Redis | None = None
        self._pubsub: redis.client.PubSub | None = None
        self._serializer = EventSerializer()

        # Neural subscriptions (preserves all semantic intelligence)
        self._subscriptions: dict[str, list[EventSubscription]] = defaultdict(list)

        # Background neural processing
        self._listener_task: asyncio.Task | None = None
        self._running = False
        self._max_workers = max_workers

        # Neural statistics
        self._stats = {
            "events_emitted": 0,
            "events_processed": 0,
            "neural_matches": 0,  # Events that matched semantic filters
            "errors": 0,
            "active_subscriptions": 0,
        }

        self._initialized = True
        logger.info(
            f"Neural EventBus initialized with Redis backbone (host={redis_host}, port={redis_port})"
        )

    async def connect(self) -> None:
        """Connect to the Redis nervous system."""
        try:
            self._redis = redis.Redis(
                host=self._redis_host,
                port=self._redis_port,
                decode_responses=False,  # We handle binary neural data ourselves
            )

            # Test the neural connection
            await self._redis.ping()
            logger.info("Neural EventBus connected to Redis nervous system")

        except Exception as e:
            logger.error(f"Failed to connect to Redis nervous system: {e}")
            raise

    async def start(self) -> None:
        """Start the neural event processing system."""
        if self._running:
            return

        if not self._redis:
            await self.connect()

        self._running = True
        self._listener_task = asyncio.create_task(self._neural_listener_loop())
        logger.info("Neural EventBus started - distributed brain is active")

    async def stop(self) -> None:
        """Stop the neural event bus and cleanup connections."""
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

        logger.info("Neural EventBus stopped")

    async def emit(
        self,
        event_type: str,
        source_plugin: str = "unknown",
        embedding: list[float] | None = None,
        **kwargs,
    ) -> None:
        """
        Emit a neural event with semantic embedding to the distributed brain.

        Args:
            event_type: Type of event to emit
            source_plugin: Plugin that emitted the event
            embedding: NEURAL EMBEDDING for semantic routing
            **kwargs: Event-specific data
        """

        try:
            if not self._redis:
                await self.connect()

            # Create neural event with embedding
            event = create_event(
                event_type=event_type,
                source_plugin=source_plugin,
                embedding=embedding,  # THIS IS THE NEURAL PART!
                **kwargs,
            )

            # Serialize neural event for transmission
            serialized_data = self._serializer.serialize(event)

            # Broadcast through the distributed nervous system
            await self._redis.publish(event_type, serialized_data)

            self._stats["events_emitted"] += 1
            logger.debug(
                f"Neural event '{event_type}' transmitted through Redis nervous system"
            )

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Failed to emit neural event '{event_type}': {e}")
            raise

    async def emit_event(self, event: BaseEvent) -> None:
        """Emit a pre-created neural event."""
        try:
            if not self._redis:
                await self.connect()

            # Serialize neural event
            serialized_data = self._serializer.serialize(event)

            # Broadcast through nervous system
            await self._redis.publish(event.event_type, serialized_data)

            self._stats["events_emitted"] += 1
            logger.debug(f"Pre-created neural event '{event.event_type}' transmitted")

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Failed to emit neural event: {e}")
            raise

    async def subscribe(
        self,
        event_type: str,
        handler: Callable,
        subscriber_id: str = None,
        semantic_filter: np.ndarray | None = None,
        semantic_threshold: float = 0.7,
        filter_func: Callable | None = None,
    ) -> str:
        """
        Subscribe to neural events with semantic intelligence.

        Args:
            event_type: Type of event to subscribe to
            handler: Async function to handle the event
            subscriber_id: Optional unique identifier for the subscription
            semantic_filter: NEURAL EMBEDDING for semantic filtering
            semantic_threshold: Minimum similarity for neural matching
            filter_func: Optional custom neural filter function

        Returns:
            Subscription ID for later unsubscribing
        """

        # Create neural subscription with semantic intelligence
        subscription = EventSubscription(
            event_type=event_type,
            handler=handler,
            subscriber_id=subscriber_id,
            semantic_filter=semantic_filter,  # NEURAL INTELLIGENCE PRESERVED!
            semantic_threshold=semantic_threshold,
            filter_func=filter_func,
        )

        self._subscriptions[event_type].append(subscription)
        self._stats["active_subscriptions"] += 1

        # Subscribe to Redis channel if we're running
        if self._running and self._pubsub:
            await self._pubsub.subscribe(event_type)

        neural_info = (
            f" (semantic filter: {'YES' if semantic_filter is not None else 'NO'})"
        )
        logger.info(
            f"Neural subscription: {subscription.subscriber_id} -> {event_type}{neural_info}"
        )
        return subscription.subscriber_id

    async def _neural_listener_loop(self) -> None:
        """
        Neural event processing loop - the brain of the distributed system.

        This receives neural events from Redis and applies semantic intelligence
        to route them to the right subscribers.
        """
        try:
            self._pubsub = self._redis.pubsub()

            # Subscribe to all neural channels we have handlers for
            if self._subscriptions:
                await self._pubsub.subscribe(*self._subscriptions.keys())
                logger.info(
                    f"Neural brain subscribed to channels: {list(self._subscriptions.keys())}"
                )

            logger.info("Neural listener loop activated - brain is processing")

            while self._running:
                try:
                    # Listen for neural signals from the distributed nervous system
                    message = await self._pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=1.0
                    )

                    if message and message["type"] == "message":
                        await self._process_neural_message(message)

                except TimeoutError:
                    # Normal timeout, continue neural processing
                    continue
                except Exception as e:
                    logger.error(f"Error in neural listener loop: {e}")
                    self._stats["errors"] += 1
                    await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Fatal error in neural listener loop: {e}")
            self._stats["errors"] += 1
        finally:
            logger.info("Neural listener loop stopped")

    async def _process_neural_message(self, message: dict[str, Any]) -> None:
        """
        Process a neural message with semantic intelligence.

        This is where the distributed brain applies neural processing
        to incoming events.
        """
        try:
            message["channel"].decode("utf-8")
            data = message["data"]

            # Deserialize neural event (preserves embeddings!)
            event = self._serializer.deserialize(data)
            self._stats["events_processed"] += 1

            # Apply neural intelligence to find matching subscriptions
            matching_subs = []
            for subscription_list in self._subscriptions.values():
                for sub in subscription_list:
                    if sub.is_active and await sub.matches_event(event):
                        matching_subs.append(sub)
                        if sub.semantic_filter is not None:
                            self._stats["neural_matches"] += 1

            # Send neural event to all matching subscribers
            if matching_subs:
                await asyncio.gather(
                    *[sub.handle_event(event) for sub in matching_subs],
                    return_exceptions=True,
                )

        except EventSerializerError as e:
            logger.error(f"Failed to deserialize neural message: {e}")
            self._stats["errors"] += 1
        except Exception as e:
            logger.error(f"Error processing neural message: {e}")
            self._stats["errors"] += 1

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from neural events.

        Args:
            subscription_id: ID returned from subscribe()

        Returns:
            True if neural subscription was found and removed
        """

        for event_type, subscription_list in self._subscriptions.items():
            for i, sub in enumerate(subscription_list):
                if sub.subscriber_id == subscription_id:
                    subscription_list.pop(i)
                    self._stats["active_subscriptions"] -= 1
                    logger.info(f"Neural unsubscribed: {subscription_id}")
                    return True

        return False

    async def get_stats(self) -> dict[str, Any]:
        """Get neural event bus statistics."""
        return {
            **self._stats,
            "connected": self._redis is not None,
            "running": self._running,
            "neural_subscriptions": sum(
                len([s for s in subs if s.semantic_filter is not None])
                for subs in self._subscriptions.values()
            ),
            "total_subscriptions": sum(
                len(subs) for subs in self._subscriptions.values()
            ),
            "event_types": list(self._subscriptions.keys()),
        }


# For backwards compatibility, let's create an EventBus alias
EventBus = RedisEventBus


# Global neural event bus singleton
_global_bus: RedisEventBus | None = None


async def get_global_bus() -> RedisEventBus:
    """Get the global Neural EventBus singleton."""
    global _global_bus
    if _global_bus is None:
        _global_bus = RedisEventBus()
        await _global_bus.connect()
        await _global_bus.start()
    return _global_bus


async def emit_global(event_type: str, **kwargs) -> None:
    """Emit a neural event using the global event bus."""
    bus = await get_global_bus()
    await bus.emit(event_type, **kwargs)


async def subscribe_global(event_type: str, handler: Callable, **kwargs) -> str:
    """Subscribe to events using the global neural event bus."""
    bus = await get_global_bus()
    return await bus.subscribe(event_type, handler, **kwargs)
