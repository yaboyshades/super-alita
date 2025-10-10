"""Event Sourcing + CQRS infrastructure using Redis Streams.

Provides immutable event storage, replay capabilities, and state projection.
Every component operation becomes an auditable event with perfect replay/debugging.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

try:
    # Use redis.asyncio for Python 3.11+ compatibility
    import redis.asyncio as aioredis
except ImportError:
    try:
        import aioredis  # Fallback to legacy aioredis
    except ImportError:
        aioredis = None  # type: ignore[assignment]

from src.contracts import UnifiedEvent


class EventStore:
    """Redis Streams-based event store with append and replay capabilities.

    Events are immutable and stored in Redis Streams for audit trails,
    debugging, and state reconstruction through event replay.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize event store.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis: Any | None = None
        self._connected = False

    async def connect(self) -> None:
        """Establish Redis connection."""
        if aioredis is None:
            raise RuntimeError(
                "aioredis not installed. Install with: pip install aioredis"
            )
        if not self._connected:
            self.redis = await aioredis.from_url(
                self.redis_url, encoding="utf-8", decode_responses=True
            )
            self._connected = True

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._connected and self.redis:
            await self.redis.close()
            self._connected = False

    async def append(
        self, stream: str, events: list[UnifiedEvent]
    ) -> list[str]:
        """Append events to a stream.

        Args:
            stream: Stream name
            events: Events to append

        Returns:
            List of Redis message IDs for appended events
        """
        if not self._connected:
            await self.connect()

        message_ids = []
        for evt in events:
            # Serialize event to flat dict for Redis
            event_dict = evt.model_dump()
            # Convert nested dicts to JSON strings
            event_dict["payload"] = json.dumps(event_dict["payload"])

            msg_id = await self.redis.xadd(f"stream:{stream}", event_dict)  # type: ignore[union-attr]
            message_ids.append(msg_id)

        return message_ids

    async def replay(
        self, stream: str, from_id: str = "0", count: int = 100
    ) -> AsyncIterator[UnifiedEvent]:
        """Replay events from a stream.

        Args:
            stream: Stream name
            from_id: Starting message ID (default: beginning)
            count: Max events per read batch

        Yields:
            UnifiedEvent instances from the stream
        """
        if not self._connected:
            await self.connect()

        current_id = from_id
        while True:
            # Read batch of messages
            messages = await self.redis.xread(  # type: ignore[union-attr]
                {f"stream:{stream}": current_id}, count=count
            )

            if not messages:
                break

            for _stream_name, stream_messages in messages:
                for msg_id, msg_data in stream_messages:
                    # Deserialize event
                    event_dict = dict(msg_data)
                    event_dict["payload"] = json.loads(event_dict["payload"])

                    yield UnifiedEvent.model_validate(event_dict)

                    # Update current ID for next iteration
                    current_id = msg_id

            # If we got fewer messages than requested, we've reached the end
            if len(stream_messages) < count:
                break

    async def get_stream_info(self, stream: str) -> dict[str, Any]:
        """Get metadata about a stream.

        Args:
            stream: Stream name

        Returns:
            Stream info dict with length, first/last IDs, etc.
        """
        if not self._connected:
            await self.connect()

        info = await self.redis.xinfo_stream(f"stream:{stream}")  # type: ignore[union-attr]
        return info


class ComponentProjection:
    """State projection from event stream.

    Rebuilds component state by replaying events and applying them
    sequentially. Supports CQRS read model construction.
    """

    def __init__(self, event_store: EventStore):
        """Initialize projection.

        Args:
            event_store: EventStore instance for event replay
        """
        self.store = event_store
        self.state: dict[str, Any] = {}
        self.last_event_id: str | None = None

    async def rebuild_from_events(self, stream: str) -> None:
        """Rebuild state from event stream.

        Args:
            stream: Stream name to replay
        """
        self.state.clear()
        self.last_event_id = None

        async for evt in self.store.replay(stream):
            await self.apply_event(evt)
            self.last_event_id = evt.event_id

    async def apply_event(self, evt: UnifiedEvent) -> None:
        """Apply a single event to update state.

        Override this method in subclasses to implement domain-specific
        event handling logic.

        Args:
            evt: Event to apply
        """
        # Default implementation: store events by type
        event_type = evt.event_type
        if event_type not in self.state:
            self.state[event_type] = []

        self.state[event_type].append(
            {
                "event_id": evt.event_id,
                "source": evt.source,
                "target": evt.target,
                "payload": evt.payload,
                "ts": evt.ts,
                "corr_id": evt.corr_id,
            }
        )

    async def get_state_snapshot(self) -> dict[str, Any]:
        """Get current state snapshot.

        Returns:
            Current projection state
        """
        return {
            "state": self.state.copy(),
            "last_event_id": self.last_event_id,
            "event_count": sum(len(events) for events in self.state.values()),
        }


class EventSourcedAggregate:
    """Base class for event-sourced aggregates.

    Aggregates apply business logic and emit events. State is derived
    entirely from event history.
    """

    def __init__(self, aggregate_id: str, event_store: EventStore):
        """Initialize aggregate.

        Args:
            aggregate_id: Unique aggregate identifier
            event_store: EventStore for persisting events
        """
        self.aggregate_id = aggregate_id
        self.event_store = event_store
        self.uncommitted_events: list[UnifiedEvent] = []
        self.version = 0

    async def load_from_history(self) -> None:
        """Load aggregate state from event history."""
        stream = f"aggregate:{self.aggregate_id}"

        async for evt in self.event_store.replay(stream):
            await self._apply(evt)
            self.version += 1

    async def _apply(self, evt: UnifiedEvent) -> None:
        """Apply event to aggregate state.

        Override in subclasses to implement domain logic.

        Args:
            evt: Event to apply
        """
        pass

    def _emit(self, evt: UnifiedEvent) -> None:
        """Emit an event (stage for commit).

        Args:
            evt: Event to emit
        """
        self.uncommitted_events.append(evt)

    async def commit(self) -> None:
        """Persist uncommitted events to event store."""
        if not self.uncommitted_events:
            return

        stream = f"aggregate:{self.aggregate_id}"
        await self.event_store.append(stream, self.uncommitted_events)

        # Apply events to local state
        for evt in self.uncommitted_events:
            await self._apply(evt)
            self.version += 1

        self.uncommitted_events.clear()
