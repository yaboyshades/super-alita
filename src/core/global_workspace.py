# Version: 3.0.0
# Description: The consciousness layer implementing Global Workspace Theory.

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AttentionLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class WorkspaceEvent:
    timestamp: float
    data: Any
    source: str
    attention_level: AttentionLevel = AttentionLevel.MEDIUM
    broadcast: bool = True
    subscribers_notified: set[str] | None = None

    def __post_init__(self):
        if self.subscribers_notified is None:
            self.subscribers_notified = set()


class GlobalWorkspace:
    """
    Consciousness layer implementing Global Workspace Theory.

    Manages event broadcasting and attention focus for the agent,
    enabling consciousness-inspired coordination between modules.
    """

    def __init__(self, max_events: int = 1000):
        """
        Initialize the Global Workspace.

        Args:
            max_events: Maximum number of events to keep in memory
        """
        self.max_events = max_events
        self._events: list[WorkspaceEvent] = []
        self._subscribers: dict[str, Callable] = {}
        self._attention_focus: set[str] = set()
        logger.info("Global Workspace initialized with max_events=%d", max_events)

    def subscribe(self, subscriber_id: str, callback: Callable):
        """
        Subscribe to workspace events.

        Args:
            subscriber_id: Unique identifier for the subscriber
            callback: Async function to call when events occur
        """
        self._subscribers[subscriber_id] = callback
        logger.debug("Subscriber '%s' registered", subscriber_id)

    async def update(
        self,
        data: Any,
        source: str = "unknown",
        attention_level: AttentionLevel = AttentionLevel.MEDIUM,
        broadcast: bool = True,
    ) -> None:
        """
        Update workspace with new information and broadcast if enabled.

        Args:
            data: The information to add to the workspace
            source: Identifier of the source module
            attention_level: Priority level for attention management
            broadcast: Whether to broadcast this event to subscribers
        """
        # Create workspace event
        event = WorkspaceEvent(
            timestamp=time.time(),
            data=data,
            source=source,
            attention_level=attention_level,
            broadcast=broadcast,
        )

        # Add to events list
        self._events.append(event)

        # Maintain event history limit
        if len(self._events) > self.max_events:
            self._events.pop(0)

        logger.debug(
            "Workspace updated: source=%s, attention=%s, broadcast=%s",
            source,
            attention_level.value,
            broadcast,
        )

        # Broadcast to subscribers if enabled
        if broadcast:
            await self._broadcast_event(event)

    async def _broadcast_event(self, event: WorkspaceEvent) -> None:
        """
        Broadcast event to relevant subscribers based on attention.

        Args:
            event: The workspace event to broadcast
        """
        # Determine which subscribers should be notified
        if self._attention_focus:
            # Focus mode: only notify subscribers in attention focus
            target_subscribers = {
                subscriber_id: callback
                for subscriber_id, callback in self._subscribers.items()
                if subscriber_id in self._attention_focus
            }
        else:
            # Normal mode: notify all subscribers
            target_subscribers = self._subscribers.copy()

        # Broadcast to each target subscriber
        notification_tasks = []
        for subscriber_id, callback in target_subscribers.items():
            try:
                # Create async task for parallel notification
                task = asyncio.create_task(
                    self._safe_notify_subscriber(subscriber_id, callback, event)
                )
                notification_tasks.append(task)
                event.subscribers_notified.add(subscriber_id)

            except Exception as e:
                logger.error(
                    "Failed to create notification task for subscriber '%s': %s",
                    subscriber_id,
                    e,
                )

        # Wait for all notifications to complete
        if notification_tasks:
            await asyncio.gather(*notification_tasks, return_exceptions=True)

        logger.debug(
            "Event broadcast complete: %d subscribers notified",
            len(event.subscribers_notified),
        )

    async def _safe_notify_subscriber(
        self, subscriber_id: str, callback: Callable, event: WorkspaceEvent
    ) -> None:
        """
        Safely notify a single subscriber with error handling.

        Args:
            subscriber_id: ID of the subscriber to notify
            callback: Callback function to invoke
            event: Event to send to the subscriber
        """
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                # Handle synchronous callbacks by running in executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, callback, event)

        except Exception as e:
            logger.error(
                "Error notifying subscriber '%s': %s", subscriber_id, e, exc_info=True
            )

    def set_attention_focus(self, subscriber_ids: set[str]) -> None:
        """
        Focus attention on specific subscribers.

        Args:
            subscriber_ids: Set of subscriber IDs to focus attention on
        """
        self._attention_focus = subscriber_ids.copy()
        logger.info(
            "Attention focus set to %d subscribers: %s",
            len(subscriber_ids),
            list(subscriber_ids),
        )

    def clear_attention_focus(self) -> None:
        """Clear attention focus, returning to normal broadcast mode."""
        self._attention_focus.clear()
        logger.info("Attention focus cleared - returning to normal broadcast mode")

    def get_recent_events(
        self,
        count: int = 10,
        limit: int | None = None,
        attention_level: AttentionLevel | None = None,
    ) -> list[WorkspaceEvent]:
        """
        Get recent events from the workspace.

        Args:
            count: Maximum number of events to return (deprecated parameter name)
            limit: Maximum number of events to return (preferred parameter name)
            attention_level: Filter by attention level (optional)

        Returns:
            List of recent WorkspaceEvent objects
        """
        # Support both 'count' and 'limit' for compatibility
        event_limit = limit if limit is not None else count

        events = self._events

        # Filter by attention level if specified
        if attention_level:
            events = [e for e in events if e.attention_level == attention_level]

        # Return most recent events
        return events[-event_limit:] if events else []

    def get_workspace_stats(self) -> dict[str, Any]:
        """
        Get statistics about the workspace state.

        Returns:
            Dictionary containing workspace statistics
        """
        return {
            "total_events": len(self._events),
            "max_events": self.max_events,
            "subscribers_count": len(self._subscribers),
            "attention_focus_active": bool(self._attention_focus),
            "attention_focus_size": len(self._attention_focus),
            "recent_events_by_level": {
                level.value: len(
                    [e for e in self._events[-100:] if e.attention_level == level]
                )
                for level in AttentionLevel
            },
        }
