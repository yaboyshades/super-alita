"""
Async Protobuf event bus for Super Alita agent communication.
Provides semantic routing and version-aware event handling.
"""

import asyncio
import contextlib
import json
import uuid
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import numpy as np

from .events import BaseEvent, create_event, serialize_event


class EventSubscription:
    """Represents a subscription to events."""

    def __init__(
        self,
        event_type: str,
        handler: Callable,
        filter_func: Callable | None = None,
        semantic_filter: np.ndarray | None = None,
        semantic_threshold: float = 0.7,
    ):
        self.id = str(uuid.uuid4())
        self.event_type = event_type
        self.handler = handler
        self.filter_func = filter_func
        self.semantic_filter = semantic_filter
        self.semantic_threshold = semantic_threshold
        self.created_at = datetime.now(UTC)
        self.call_count = 0
        self.last_called = None
        self.is_active = True

    async def should_handle(self, event: BaseEvent) -> bool:
        """Check if this subscription should handle the event."""

        if not self.is_active:
            return False

        # Check event type
        if self.event_type != "*" and event.event_type != self.event_type:
            return False

        # Apply custom filter
        if self.filter_func:
            if asyncio.iscoroutinefunction(self.filter_func):
                if not await self.filter_func(event):
                    return False
            elif not self.filter_func(event):
                return False

        # Apply semantic filter
        if self.semantic_filter is not None and event.embedding is not None:
            similarity = np.dot(self.semantic_filter, event.embedding)
            if similarity < self.semantic_threshold:
                return False

        return True

    async def handle_event(self, event: BaseEvent) -> None:
        """Handle the event with this subscription."""

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
            # Log error but don't break event handling
            print(f"Error in event handler {self.id}: {e}")

    def deactivate(self) -> None:
        """Deactivate this subscription."""
        self.is_active = False


class EventBus:
    """
    Async event bus with semantic routing and versioning support.

    Provides decoupled communication between agent plugins with
    support for semantic filtering and event versioning.
    """

    def __init__(self, max_workers: int = 4):
        self._subscriptions: dict[str, list[EventSubscription]] = defaultdict(list)
        self._global_subscriptions: list[EventSubscription] = []
        self._event_history: list[BaseEvent] = []
        self._max_history = 1000
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._stats = {
            "events_emitted": 0,
            "events_handled": 0,
            "subscriptions_created": 0,
            "errors": 0,
        }
        self._lock = asyncio.Lock()
        self._running = False
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        """Check if the event bus is currently running."""
        return self._running

    async def start(self) -> None:
        """Start the event bus processing."""
        if not self._running:
            self._running = True
            self._processing_task = asyncio.create_task(self._process_events())

    async def stop(self) -> None:
        """Stop the event bus processing."""
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._processing_task

        self._executor.shutdown(wait=True)

    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                # Wait for event with timeout to allow checking _running
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._handle_event_internal(event)
                self._event_queue.task_done()
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._stats["errors"] += 1
                print(f"Error processing event: {e}")

    async def emit(
        self,
        event_type: str,
        source_plugin: str = "unknown",
        embedding: list[float] | None = None,
        **kwargs,
    ) -> None:
        """
        Emit an event to the bus.

        Args:
            event_type: Type of event to emit
            source_plugin: Plugin that emitted the event
            embedding: Optional semantic embedding for routing
            **kwargs: Event-specific data
        """

        # Create event
        event = create_event(
            event_type=event_type,
            source_plugin=source_plugin,
            embedding=embedding,
            **kwargs,
        )

        # Add to queue for processing
        await self._event_queue.put(event)
        self._stats["events_emitted"] += 1

    async def emit_event(self, event: BaseEvent) -> None:
        """Emit a pre-created event."""
        await self._event_queue.put(event)
        self._stats["events_emitted"] += 1

    async def _handle_event_internal(self, event: BaseEvent) -> None:
        """Internal event handling."""

        # Add to history
        async with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

        # Handle with specific subscriptions
        event_subscriptions = self._subscriptions.get(event.event_type, [])
        all_subscriptions = event_subscriptions + self._global_subscriptions

        # Filter and handle subscriptions concurrently
        tasks = []
        for subscription in all_subscriptions:
            if await subscription.should_handle(event):
                tasks.append(subscription.handle_event(event))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self._stats["events_handled"] += len(tasks)

    async def subscribe(
        self,
        event_type: str,
        handler: Callable,
        filter_func: Callable | None = None,
        semantic_filter: list[float] | None = None,
        semantic_threshold: float = 0.7,
    ) -> str:
        """
        Subscribe to events.

        Args:
            event_type: Type of events to subscribe to, or "*" for all
            handler: Async or sync function to handle events
            filter_func: Optional filter function for additional filtering
            semantic_filter: Optional embedding for semantic filtering
            semantic_threshold: Similarity threshold for semantic filtering

        Returns:
            Subscription ID for later unsubscribing
        """

        # Convert semantic filter to numpy array
        semantic_array = None
        if semantic_filter:
            semantic_array = np.array(semantic_filter)
            # Normalize for cosine similarity
            semantic_array = semantic_array / np.linalg.norm(semantic_array)

        subscription = EventSubscription(
            event_type=event_type,
            handler=handler,
            filter_func=filter_func,
            semantic_filter=semantic_array,
            semantic_threshold=semantic_threshold,
        )

        if event_type == "*":
            self._global_subscriptions.append(subscription)
        else:
            self._subscriptions[event_type].append(subscription)

        self._stats["subscriptions_created"] += 1
        return subscription.id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: ID returned from subscribe()

        Returns:
            True if subscription was found and removed
        """

        # Check global subscriptions
        for i, sub in enumerate(self._global_subscriptions):
            if sub.id == subscription_id:
                sub.deactivate()
                del self._global_subscriptions[i]
                return True

        # Check event-specific subscriptions
        for event_type, subscriptions in self._subscriptions.items():
            for i, sub in enumerate(subscriptions):
                if sub.id == subscription_id:
                    sub.deactivate()
                    del subscriptions[i]
                    return True

        return False

    async def subscribe_semantic(
        self,
        semantic_query: list[float],
        handler: Callable,
        threshold: float = 0.7,
        event_type: str = "*",
    ) -> str:
        """
        Subscribe to events based on semantic similarity.

        Args:
            semantic_query: Embedding vector for semantic matching
            handler: Event handler function
            threshold: Similarity threshold (0-1)
            event_type: Event type filter, or "*" for all

        Returns:
            Subscription ID
        """

        return await self.subscribe(
            event_type=event_type,
            handler=handler,
            semantic_filter=semantic_query,
            semantic_threshold=threshold,
        )

    async def get_event_history(
        self,
        event_type: str | None = None,
        limit: int = 100,
        source_plugin: str | None = None,
    ) -> list[BaseEvent]:
        """
        Get event history with optional filtering.

        Args:
            event_type: Filter by event type
            limit: Maximum number of events to return
            source_plugin: Filter by source plugin

        Returns:
            List of events in reverse chronological order
        """

        async with self._lock:
            events = list(reversed(self._event_history))

        # Apply filters
        filtered_events = []
        for event in events:
            if event_type and event.event_type != event_type:
                continue
            if source_plugin and event.source_plugin != source_plugin:
                continue

            filtered_events.append(event)
            if len(filtered_events) >= limit:
                break

        return filtered_events

    async def search_events_semantic(
        self, query_embedding: list[float], threshold: float = 0.7, limit: int = 50
    ) -> list[BaseEvent]:
        """
        Search events by semantic similarity.

        Args:
            query_embedding: Query vector
            threshold: Similarity threshold
            limit: Maximum results

        Returns:
            List of similar events
        """

        query_array = np.array(query_embedding)
        query_array = query_array / np.linalg.norm(query_array)

        similar_events = []

        async with self._lock:
            for event in reversed(self._event_history):
                if event.embedding:
                    event_embedding = np.array(event.embedding)
                    similarity = np.dot(query_array, event_embedding)

                    if similarity >= threshold:
                        similar_events.append((event, similarity))

        # Sort by similarity and return top results
        similar_events.sort(key=lambda x: x[1], reverse=True)
        return [event for event, _ in similar_events[:limit]]

    async def get_stats(self) -> dict[str, Any]:
        """Get event bus statistics."""

        active_subscriptions = sum(
            len(subs) for subs in self._subscriptions.values()
        ) + len(self._global_subscriptions)

        return {
            **self._stats,
            "active_subscriptions": active_subscriptions,
            "event_history_size": len(self._event_history),
            "queue_size": self._event_queue.qsize(),
            "is_running": self._running,
        }

    async def clear_history(self) -> None:
        """Clear event history."""
        async with self._lock:
            self._event_history.clear()

    async def export_events(
        self,
        filepath: str,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> None:
        """
        Export events to JSON file.

        Args:
            filepath: Output file path
            event_type: Filter by event type
            limit: Maximum number of events
        """

        events = await self.get_event_history(
            event_type=event_type, limit=limit or len(self._event_history)
        )

        export_data = {
            "events": [serialize_event(event) for event in events],
            "export_timestamp": datetime.now(UTC).isoformat(),
            "total_events": len(events),
            "stats": await self.get_stats(),
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)


# Global event bus instance
_global_bus: EventBus | None = None


async def get_global_bus() -> EventBus:
    """Get the global event bus instance."""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
        await _global_bus.start()
    return _global_bus


async def emit_global(event_type: str, **kwargs) -> None:
    """Convenience function to emit to global bus."""
    bus = await get_global_bus()
    await bus.emit(event_type, **kwargs)


async def subscribe_global(event_type: str, handler: Callable, **kwargs) -> str:
    """Convenience function to subscribe to global bus."""
    bus = await get_global_bus()
    return await bus.subscribe(event_type, handler, **kwargs)
