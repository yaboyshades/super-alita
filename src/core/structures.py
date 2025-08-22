"""Core data structures for Super Alita.

This module defines immutable :class:`Atom` and :class:`Bond` records and a
helper for deterministic identifier generation.  The implementation is minimal
and only includes the features required by the diagnostics script.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any


def _normalize(text: str) -> str:
    """Normalize content for ID generation.

    The content is lower‑cased and consecutive whitespace is collapsed.  If the
    normalized text exceeds 256 characters, a SHA256 digest is used instead to
    keep the UUID seed bounded.

    Args:
        text: Raw content string.

    Returns:
        str: Normalized content or ``sha256:<digest>`` when truncated.
    """

    collapsed = " ".join(text.lower().split())
    if len(collapsed) > 256:
        digest = hashlib.sha256(collapsed.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"
    return collapsed


def generate_atom_id(
    atom_type: str, title: str, content: str, namespace: uuid.UUID
) -> str:
    """Generate a deterministic UUIDv5 for an atom.

    Args:
        atom_type: The type/category of the atom (e.g. ``"NOTE"``).
        title: Optional title for the atom.
        content: Atom body text.
        namespace: UUID namespace providing deterministic seeding.

    Returns:
        str: Deterministic UUID string.
    """

    seed = f"{atom_type}|{title.strip().lower()}|{_normalize(content)}"
    return str(uuid.uuid5(namespace, seed))


@dataclass(frozen=True)
class Atom:
    """Immutable knowledge record."""

    atom_id: str
    atom_type: str
    content: str
    title: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Bond:
    """Relation between two atoms."""

    source_id: str
    target_id: str
    bond_type: str
    meta: dict[str, Any] = field(default_factory=dict)
