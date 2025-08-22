"""Demo atomizer tool."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Any

from core.structures import Atom, Bond, generate_atom_id

from mcp.registry import ToolRegistry

NS = uuid.UUID("12345678-1234-5678-1234-567812345678")


async def atomizer(
    text: str, max_notes: int = 5, context_tags: list[str] | None = None
) -> dict[str, Any]:
    """Split text into NOTE atoms linked sequentially.

    Args:
        text: Input text to segment.
        max_notes: Maximum number of atoms to produce.
        context_tags: Optional provenance tags.

    Returns:
        Dict[str, Any]: ``{"atoms": [...], "bonds": [...]}`` payload.
    """

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()][:max_notes]
    atoms: list[dict[str, Any]] = []
    bonds: list[dict[str, Any]] = []

    for idx, sentence in enumerate(sentences):
        atom_id = generate_atom_id("NOTE", "", sentence, NS)
        atom = Atom(atom_id=atom_id, atom_type="NOTE", content=sentence, meta={})
        prov = {
            "source_type": "tool",
            "source_id": "atomizer/0.1.0",
            "activity_type": "invoke",
            "timestamp": datetime.now(UTC).isoformat(),
            "context_tags": context_tags or [],
        }
        atom.meta["provenance"] = prov
        atoms.append(atom.__dict__)

        if idx > 0:
            bond = Bond(
                source_id=atoms[idx - 1]["atom_id"],
                target_id=atom_id,
                bond_type="RELATES_TO",
                meta={"provenance": {"parent_atom_ids": [atoms[idx - 1]["atom_id"]]}},
            )
            bonds.append(
                {
                    "source_id": bond.source_id,
                    "target_id": bond.target_id,
                    "bond_type": bond.bond_type,
                    "meta": bond.meta,
                }
            )

    return {"atoms": atoms, "bonds": bonds}


def register_into_registry(registry: ToolRegistry) -> None:
    """Register the atomizer tool into a :class:`ToolRegistry`.

    Args:
        registry: Target registry instance.
    """

    registry.register("atomizer", atomizer)
