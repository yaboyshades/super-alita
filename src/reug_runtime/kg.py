"""Minimal Knowledge Graph interface and development adapters.

This module defines a ``BaseKG`` protocol with CRUD operations and two
simple adapters:

* ``InMemoryKG`` – volatile store useful for tests and local runs.
* ``FileKG`` – persists atoms and bonds to a JSON file.

An external driver can be supplied via the ``REUG_KG_DRIVER`` environment
variable using the ``module:Class`` pattern. If unspecified, the factory
falls back to ``FileKG`` when ``REUG_KG_FILE`` is set, otherwise
``InMemoryKG``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
import json
import os
from pathlib import Path
from typing import Any, Protocol


class BaseKG(Protocol):
    """Operations the runtime expects from a knowledge graph backend."""

    async def retrieve_relevant_context(self, user_message: str) -> str:
        ...

    async def get_goal_for_session(self, session_id: str) -> dict[str, Any]:
        ...

    async def create_atom(self, atom_type: str, content: Any) -> dict[str, Any]:
        ...

    async def get_atom(self, atom_id: str) -> dict[str, Any] | None:
        ...

    async def update_atom(self, atom_id: str, content: Any) -> None:
        ...

    async def delete_atom(self, atom_id: str) -> None:
        ...

    async def create_bond(
        self, bond_type: str, source_atom_id: str, target_atom_id: str
    ) -> None:
        ...

    async def get_bonds(self, atom_id: str) -> list[dict[str, Any]]:
        ...

    async def delete_bond(
        self, bond_type: str, source_atom_id: str, target_atom_id: str
    ) -> None:
        ...


@dataclass
class InMemoryKG(BaseKG):
    """Simple in-memory knowledge graph used for development and tests."""

    atoms: dict[str, dict[str, Any]] = field(default_factory=dict)
    bonds: list[dict[str, Any]] = field(default_factory=list)

    async def retrieve_relevant_context(self, user_message: str) -> str:
        return f"(kg_ctx for: {user_message[:32]}...)"

    async def get_goal_for_session(self, session_id: str) -> dict[str, Any]:
        return {
            "id": f"goal_{session_id}",
            "description": f"Assist session {session_id}",
        }

    async def create_atom(self, atom_type: str, content: Any) -> dict[str, Any]:
        atom_id = f"atom_{len(self.atoms)}"
        atom = {"id": atom_id, "type": atom_type, "content": content}
        self.atoms[atom_id] = atom
        return atom

    async def get_atom(self, atom_id: str) -> dict[str, Any] | None:
        return self.atoms.get(atom_id)

    async def update_atom(self, atom_id: str, content: Any) -> None:
        if atom_id in self.atoms:
            self.atoms[atom_id]["content"] = content

    async def delete_atom(self, atom_id: str) -> None:
        self.atoms.pop(atom_id, None)
        self.bonds = [
            b for b in self.bonds if b["src"] != atom_id and b["tgt"] != atom_id
        ]

    async def create_bond(
        self, bond_type: str, source_atom_id: str, target_atom_id: str
    ) -> None:
        self.bonds.append(
            {"type": bond_type, "src": source_atom_id, "tgt": target_atom_id}
        )

    async def get_bonds(self, atom_id: str) -> list[dict[str, Any]]:
        return [
            b
            for b in self.bonds
            if b["src"] == atom_id or b["tgt"] == atom_id
        ]

    async def delete_bond(
        self, bond_type: str, source_atom_id: str, target_atom_id: str
    ) -> None:
        self.bonds = [
            b
            for b in self.bonds
            if not (
                b["type"] == bond_type
                and b["src"] == source_atom_id
                and b["tgt"] == target_atom_id
            )
        ]


class FileKG(InMemoryKG):
    """Knowledge graph that persists data to a JSON file."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        if path.exists():
            try:
                data = json.loads(path.read_text("utf-8"))
                for atom in data.get("atoms", []):
                    self.atoms[atom["id"]] = atom
                self.bonds.extend(data.get("bonds", []))
            except Exception:
                pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"atoms": list(self.atoms.values()), "bonds": self.bonds}
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    async def create_atom(self, atom_type: str, content: Any) -> dict[str, Any]:
        atom = await super().create_atom(atom_type, content)
        self._save()
        return atom

    async def update_atom(self, atom_id: str, content: Any) -> None:
        await super().update_atom(atom_id, content)
        self._save()

    async def delete_atom(self, atom_id: str) -> None:
        await super().delete_atom(atom_id)
        self._save()

    async def create_bond(
        self, bond_type: str, source_atom_id: str, target_atom_id: str
    ) -> None:
        await super().create_bond(bond_type, source_atom_id, target_atom_id)
        self._save()

    async def delete_bond(
        self, bond_type: str, source_atom_id: str, target_atom_id: str
    ) -> None:
        await super().delete_bond(bond_type, source_atom_id, target_atom_id)
        self._save()


def create_kg_from_env() -> BaseKG:
    """Factory that constructs a knowledge graph backend based on environment."""

    driver = os.getenv("REUG_KG_DRIVER")
    if driver:
        module, _, cls_name = driver.partition(":")
        mod = import_module(module)
        cls = getattr(mod, cls_name)
        return cls()  # type: ignore[return-value]

    file_path = os.getenv("REUG_KG_FILE")
    if file_path:
        return FileKG(Path(file_path))

    return InMemoryKG()
