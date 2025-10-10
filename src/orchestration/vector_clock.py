"""Vector Clocks for distributed temporal consistency.

Implements vector clock mechanism for:
- Causal ordering of distributed events
- Happens-before relationship detection
- Concurrent event identification
- Distributed system synchronization
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CausalRelation(str, Enum):
    """Possible causal relationships between events."""

    HAPPENS_BEFORE = "happens_before"  # A -> B
    HAPPENS_AFTER = "happens_after"  # A <- B
    CONCURRENT = "concurrent"  # A || B
    IDENTICAL = "identical"  # A == B


@dataclass
class VectorClock:
    """Vector clock for distributed event ordering.

    Each component maintains a logical clock. When events occur,
    the local clock increments. When events are sent/received,
    clocks are merged to maintain causal consistency.
    """

    clocks: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure clocks use defaultdict."""
        if not isinstance(self.clocks, defaultdict):
            self.clocks = defaultdict(int, self.clocks)

    def increment(self, component: str) -> VectorClock:
        """Increment local clock for a component.

        Args:
            component: Component name

        Returns:
            New VectorClock with incremented value
        """
        new_clocks = dict(self.clocks)
        new_clocks[component] = new_clocks.get(component, 0) + 1
        return VectorClock(clocks=new_clocks)

    def merge(self, other: VectorClock) -> VectorClock:
        """Merge with another vector clock (take maximum of each component).

        Args:
            other: Other vector clock

        Returns:
            New VectorClock with merged values
        """
        all_components = set(self.clocks.keys()) | set(other.clocks.keys())

        merged = {
            component: max(
                self.clocks.get(component, 0), other.clocks.get(component, 0)
            )
            for component in all_components
        }

        return VectorClock(clocks=merged)

    def compare(self, other: VectorClock) -> CausalRelation:
        """Compare with another vector clock to determine causal relationship.

        Args:
            other: Other vector clock

        Returns:
            CausalRelation (HAPPENS_BEFORE, HAPPENS_AFTER, CONCURRENT, IDENTICAL)
        """
        all_components = set(self.clocks.keys()) | set(other.clocks.keys())

        # Check if identical
        if all(
            self.clocks.get(c, 0) == other.clocks.get(c, 0)
            for c in all_components
        ):
            return CausalRelation.IDENTICAL

        # Check if self <= other (happens before)
        self_leq_other = all(
            self.clocks.get(c, 0) <= other.clocks.get(c, 0)
            for c in all_components
        )

        # Check if other <= self (happens after)
        other_leq_self = all(
            other.clocks.get(c, 0) <= self.clocks.get(c, 0)
            for c in all_components
        )

        if self_leq_other and not other_leq_self:
            return CausalRelation.HAPPENS_BEFORE
        elif other_leq_self and not self_leq_other:
            return CausalRelation.HAPPENS_AFTER
        else:
            return CausalRelation.CONCURRENT

    def to_dict(self) -> dict[str, int]:
        """Export to dictionary."""
        return dict(self.clocks)

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> VectorClock:
        """Import from dictionary."""
        return cls(clocks=data)

    def __str__(self) -> str:
        """String representation."""
        items = sorted(self.clocks.items())
        return "{" + ", ".join(f"{k}:{v}" for k, v in items) + "}"

    def __repr__(self) -> str:
        """Repr."""
        return f"VectorClock({self})"


@dataclass
class TimestampedEvent:
    """Event with vector clock timestamp."""

    event_id: str
    component: str
    event_type: str
    payload: dict[str, Any]
    timestamp: VectorClock

    def happens_before(self, other: TimestampedEvent) -> bool:
        """Check if this event happens before another.

        Args:
            other: Other event

        Returns:
            True if this event causally precedes the other
        """
        relation = self.timestamp.compare(other.timestamp)
        return relation == CausalRelation.HAPPENS_BEFORE

    def concurrent_with(self, other: TimestampedEvent) -> bool:
        """Check if this event is concurrent with another.

        Args:
            other: Other event

        Returns:
            True if events are concurrent (no causal relationship)
        """
        relation = self.timestamp.compare(other.timestamp)
        return relation == CausalRelation.CONCURRENT


class VectorClockManager:
    """Manages vector clocks for distributed components.

    Handles clock increments, merges, and causal relationship queries.
    Useful for:
    - Distributed event ordering
    - Conflict detection (concurrent writes)
    - Causal consistency validation
    """

    def __init__(self):
        """Initialize manager."""
        self.component_clocks: dict[str, VectorClock] = {}
        self.event_history: list[TimestampedEvent] = []

    def register_component(self, component: str) -> None:
        """Register a component with initial clock.

        Args:
            component: Component name
        """
        if component not in self.component_clocks:
            self.component_clocks[component] = VectorClock()

    def local_event(
        self,
        component: str,
        event_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> TimestampedEvent:
        """Record a local event (no message sent).

        Args:
            component: Component generating event
            event_id: Event identifier
            event_type: Event type
            payload: Event payload

        Returns:
            TimestampedEvent with updated clock
        """
        if component not in self.component_clocks:
            self.register_component(component)

        # Increment local clock
        self.component_clocks[component] = self.component_clocks[
            component
        ].increment(component)

        event = TimestampedEvent(
            event_id=event_id,
            component=component,
            event_type=event_type,
            payload=payload or {},
            timestamp=self.component_clocks[component],
        )

        self.event_history.append(event)
        return event

    def send_event(
        self,
        sender: str,
        receiver: str,
        event_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> TimestampedEvent:
        """Record event sent from one component to another.

        Args:
            sender: Sending component
            receiver: Receiving component
            event_id: Event identifier
            event_type: Event type
            payload: Event payload

        Returns:
            TimestampedEvent with sender's updated clock
        """
        if sender not in self.component_clocks:
            self.register_component(sender)
        if receiver not in self.component_clocks:
            self.register_component(receiver)

        # Increment sender's clock
        self.component_clocks[sender] = self.component_clocks[
            sender
        ].increment(sender)

        event = TimestampedEvent(
            event_id=event_id,
            component=sender,
            event_type=event_type,
            payload=payload or {},
            timestamp=self.component_clocks[sender],
        )

        self.event_history.append(event)
        return event

    def receive_event(
        self, receiver: str, received_event: TimestampedEvent
    ) -> VectorClock:
        """Record event received by a component.

        Merges received clock with local clock and increments.

        Args:
            receiver: Receiving component
            received_event: Event being received

        Returns:
            Updated VectorClock for receiver
        """
        if receiver not in self.component_clocks:
            self.register_component(receiver)

        # Merge with received clock
        self.component_clocks[receiver] = self.component_clocks[
            receiver
        ].merge(received_event.timestamp)

        # Increment local clock
        self.component_clocks[receiver] = self.component_clocks[
            receiver
        ].increment(receiver)

        return self.component_clocks[receiver]

    def get_causal_history(
        self, event: TimestampedEvent
    ) -> list[TimestampedEvent]:
        """Get all events that causally precede this event.

        Args:
            event: Target event

        Returns:
            List of events that happened before this one
        """
        return [
            e
            for e in self.event_history
            if e.event_id != event.event_id
            and e.timestamp.compare(event.timestamp)
            == CausalRelation.HAPPENS_BEFORE
        ]

    def get_concurrent_events(
        self, event: TimestampedEvent
    ) -> list[TimestampedEvent]:
        """Get all events concurrent with this event.

        Args:
            event: Target event

        Returns:
            List of concurrent events
        """
        return [
            e
            for e in self.event_history
            if e.event_id != event.event_id
            and e.timestamp.compare(event.timestamp)
            == CausalRelation.CONCURRENT
        ]

    def detect_conflicts(
        self, event_type: str | None = None
    ) -> list[tuple[TimestampedEvent, TimestampedEvent]]:
        """Detect concurrent events of the same type (potential conflicts).

        Args:
            event_type: Filter by event type (all if None)

        Returns:
            List of (event_a, event_b) pairs that are concurrent
        """
        events = (
            [e for e in self.event_history if e.event_type == event_type]
            if event_type
            else self.event_history
        )

        conflicts = []
        for i, event_a in enumerate(events):
            for event_b in events[i + 1 :]:
                if event_a.concurrent_with(event_b):
                    conflicts.append((event_a, event_b))

        return conflicts

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        return {
            "components": len(self.component_clocks),
            "total_events": len(self.event_history),
            "component_clocks": {
                comp: str(clock)
                for comp, clock in self.component_clocks.items()
            },
        }


# Example usage
def example_vector_clocks() -> None:
    """Example demonstrating vector clock usage."""
    manager = VectorClockManager()

    # Register components
    manager.register_component("service_a")
    manager.register_component("service_b")
    manager.register_component("service_c")

    # Local events
    e1 = manager.local_event("service_a", "e1", "user_login")
    print(f"Event e1: {e1.timestamp}")

    e2 = manager.local_event("service_b", "e2", "db_query")
    print(f"Event e2: {e2.timestamp}")

    # Send event from A to B
    e3 = manager.send_event("service_a", "service_b", "e3", "api_call")
    print(f"Event e3 (A->B): {e3.timestamp}")

    # B receives event from A
    manager.receive_event("service_b", e3)
    e4 = manager.local_event("service_b", "e4", "db_write")
    print(f"Event e4 (after receive): {e4.timestamp}")

    # Concurrent event on C
    e5 = manager.local_event("service_c", "e5", "db_write")
    print(f"Event e5 (concurrent): {e5.timestamp}")

    # Check causal relationships
    print(f"\ne1 -> e3: {e1.happens_before(e3)}")
    print(f"e3 -> e4: {e3.happens_before(e4)}")
    print(f"e4 || e5: {e4.concurrent_with(e5)}")

    # Causal history of e4
    history = manager.get_causal_history(e4)
    print(f"\nCausal history of e4: {[e.event_id for e in history]}")

    # Concurrent events with e4
    concurrent = manager.get_concurrent_events(e4)
    print(f"Concurrent with e4: {[e.event_id for e in concurrent]}")

    # Detect conflicts
    conflicts = manager.detect_conflicts("db_write")
    print(
        f"\nConflicts (db_write): {[(a.event_id, b.event_id) for a, b in conflicts]}"
    )

    # Stats
    print("\nStats:")
    print(json.dumps(manager.get_stats(), indent=2))


if __name__ == "__main__":
    example_vector_clocks()
