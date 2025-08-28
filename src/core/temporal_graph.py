from __future__ import annotations

"""Minimal temporal graph and neural atom structures used by cortex plugins."""

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NeuralAtom:
    uuid: str
    content: str
    atom_type: str
    metadata: dict[str, Any]
    _outgoing: list[str] = field(default_factory=list)

    def bonds_out(self) -> list[str]:
        return list(self._outgoing)

    def add_bond(self, target_uuid: str) -> None:
        self._outgoing.append(target_uuid)


class TemporalGraph:
    """Very small in-memory graph for tests."""

    def __init__(self) -> None:
        self.atoms: dict[str, NeuralAtom] = {}

    def create_atom(
        self, content: str, atom_type: str, metadata: dict[str, Any]
    ) -> NeuralAtom:
        atom_uuid = str(uuid.uuid4())
        atom = NeuralAtom(
            uuid=atom_uuid, content=content, atom_type=atom_type, metadata=metadata
        )
        self.atoms[atom_uuid] = atom
        return atom

    def create_bond(
        self, source_uuid: str, target_uuid: str, metadata: dict[str, Any] | None = None
    ) -> None:
        if source_uuid in self.atoms and target_uuid in self.atoms:
            self.atoms[source_uuid].add_bond(target_uuid)
