"""Observation utilities for CodeAct."""

from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Any

from src.core.structures import generate_atom_id


@dataclass(slots=True)
class Observation:
    """Normalized output of sandbox execution."""

    stdout: str = ""
    stderr: str = ""
    error: str | None = None

    def to_atom(self, namespace: uuid.UUID) -> dict[str, Any]:
        """Convert this observation into an atom dictionary."""
        content = self.stdout or self.error or ""
        atom_id = generate_atom_id("OBSERVATION", "", content, namespace)
        return {"atom_id": atom_id, "atom_type": "OBSERVATION", "content": content}
