"""Knowledge atom models for the neural federation layer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(slots=True)
class KnowledgeAtom:
    """Minimal representation of a knowledge atom destined for neural storage."""

    atom_id: str
    text: str
    tags: Sequence[str] = field(default_factory=tuple)
    embedding: Sequence[float] | None = None
    metadata: dict[str, str | float | int] = field(default_factory=dict)

    def has_embedding(self) -> bool:
        return self.embedding is not None and len(self.embedding) > 0
