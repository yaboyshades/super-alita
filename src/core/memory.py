"""In-memory graph store for atoms and bonds."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

from .structures import Atom, Bond


class Memory:
    """Simple lineage-aware store for atoms and bonds."""

    def __init__(self) -> None:
        self.atoms: dict[str, Atom] = {}
        self.bonds: dict[tuple[str, str, str], Bond] = {}
        self._children: dict[str, list[str]] = defaultdict(list)
        self._parents: dict[str, list[str]] = defaultdict(list)

    def store_atoms_bonds(
        self, atoms: Iterable[Atom], bonds: Iterable[Bond]
    ) -> dict[str, list[str]]:
        """Store atoms and bonds with deduplication.

        Args:
            atoms: Iterable of atoms to persist.
            bonds: Iterable of bonds to persist.

        Returns:
            Dict[str, List[str]]: Lists of stored atom IDs and bond keys.
        """

        stored_atoms: list[str] = []
        for atom in atoms:
            self.atoms[atom.atom_id] = atom
            stored_atoms.append(atom.atom_id)

        stored_bonds: list[str] = []
        for bond in bonds:
            key = (bond.source_id, bond.target_id, bond.bond_type)
            if key not in self.bonds:
                self.bonds[key] = bond
                self._children[bond.source_id].append(bond.target_id)
                self._parents[bond.target_id].append(bond.source_id)
                stored_bonds.append(str(key))

        return {"stored_atoms": stored_atoms, "stored_bonds": stored_bonds}

    def get_children(self, atom_id: str) -> list[str]:
        """Return immediate children of an atom."""

        return list(self._children.get(atom_id, []))

    def get_parents(self, atom_id: str) -> list[str]:
        """Return immediate parents of an atom."""

        return list(self._parents.get(atom_id, []))

    def bfs(
        self, start_ids: Iterable[str], direction: str = "forward", max_hops: int = 2
    ) -> dict[str, int]:
        """Breadth-first search over the bond graph.

        Args:
            start_ids: Iterable of starting atom IDs.
            direction: ``"forward"``, ``"backward"``, or ``"both"``.
            max_hops: Maximum traversal depth.

        Returns:
            Dict[str, int]: Mapping of visited atom IDs to hop distance.
        """

        visited: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque((sid, 0) for sid in start_ids)

        while queue:
            current, dist = queue.popleft()
            if current in visited or dist > max_hops:
                continue
            visited[current] = dist

            if direction in ("forward", "both"):
                for child in self._children.get(current, []):
                    queue.append((child, dist + 1))
            if direction in ("backward", "both"):
                for parent in self._parents.get(current, []):
                    queue.append((parent, dist + 1))

        return visited
