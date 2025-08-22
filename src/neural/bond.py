#!/usr/bin/env python3
"""
Bond relationships between Neural Atoms
"""

import json
import uuid
from dataclasses import asdict, dataclass
from typing import Any

try:
    from prometheus_client import Counter

    # Prometheus metrics
    bond_events = Counter("neural_bond_events_total", "Bond events", ["type"])
    prometheus_available = True
except ImportError:
    prometheus_available = False

    # Mock counter for graceful degradation
    class MockCounter:
        def labels(self, **kwargs: Any) -> "MockCounter":
            return self

        def inc(self) -> None:
            pass

    bond_events = MockCounter()

# Use same namespace as atoms for consistency
ATOM_NS = uuid.UUID("d6e2a8b1-4c7f-4e0a-8b9c-1d2e3f4a5b6c")


@dataclass
class Bond:
    """
    Relationship between Neural Atoms

    A Bond represents a typed relationship between two atoms with:
    - Deterministic UUIDv5 ID for reproducibility
    - Energy weight for relationship strength
    - Type classification for relationship semantics
    """

    source_id: str
    target_id: str
    bond_type: str
    energy: float = 0.5
    bond_id: str | None = None

    def __post_init__(self):
        if not self.bond_id:
            seed = f"{self.source_id}|{self.target_id}|{self.bond_type}"
            self.bond_id = str(uuid.uuid5(ATOM_NS, seed))

        # Update metrics if available
        if prometheus_available:
            bond_events.labels(type=self.bond_type).inc()

    def to_dict(self) -> dict[str, Any]:
        """Convert bond to dictionary for serialization"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert bond to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Bond":
        """Create bond from dictionary"""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "Bond":
        """Create bond from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        return f"Bond({self.bond_type}, {self.source_id} -> {self.target_id}, {self.bond_id})"

    def __repr__(self) -> str:
        return f"Bond(bond_type='{self.bond_type}', source_id='{self.source_id}', target_id='{self.target_id}', bond_id='{self.bond_id}')"
