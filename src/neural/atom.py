#!/usr/bin/env python3
"""
Neural Atom implementation with deterministic UUIDv5 IDs
"""

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from typing import Any

try:
    from prometheus_client import Counter

    # Prometheus metrics
    atom_events = Counter("neural_atom_events_total", "Neural Atom events", ["type"])
    prometheus_available = True
except ImportError:
    prometheus_available = False

    # Mock counter for graceful degradation
    class MockCounter:
        def labels(self, **kwargs: Any) -> "MockCounter":
            return self

        def inc(self) -> None:
            pass

    atom_events = MockCounter()

# Deterministic UUID namespace for Super-Alita Neural Atoms
ATOM_NS = uuid.UUID("d6e2a8b1-4c7f-4e0a-8b9c-1d2e3f4a5b6c")


@dataclass
class Atom:
    """
    Neural Atom - deterministic knowledge unit

    A Neural Atom represents a discrete piece of knowledge with:
    - Deterministic UUIDv5 ID for reproducibility
    - Content-based hashing for large content
    - Metadata for context and relationships
    - Type classification for processing
    """

    atom_type: str
    title: str
    content: str
    meta: dict[str, Any]
    atom_id: str | None = None

    def __post_init__(self):
        if not self.atom_id:
            # Use content hashing for large content to keep seed manageable
            content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
            seed = f"{self.atom_type}|{self.title}|{content_hash}"
            self.atom_id = str(uuid.uuid5(ATOM_NS, seed))

        # Update metrics if available
        if prometheus_available:
            atom_events.labels(type=self.atom_type).inc()

    def to_dict(self) -> dict[str, Any]:
        """Convert atom to dictionary for serialization"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert atom to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Atom":
        """Create atom from dictionary"""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "Atom":
        """Create atom from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        return f"Atom({self.atom_type}, {self.title[:50]}..., {self.atom_id})"

    def __repr__(self) -> str:
        return f"Atom(atom_type='{self.atom_type}', title='{self.title}', atom_id='{self.atom_id}')"
