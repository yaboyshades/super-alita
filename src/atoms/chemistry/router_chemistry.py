from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...core.neural_atom import NeuralAtom, NeuralStore

from .primitives import bond_strength


class AtomRegistry:
    """
    Simple atom registry interface that wraps NeuralStore for chemistry operations.
    Provides compatibility layer for the chemistry system.
    """

    def __init__(self, neural_store: NeuralStore | None = None):
        self._store = neural_store

    def list_ids(self) -> list[str]:
        """List all atom IDs in the registry."""
        if self._store:
            return list(self._store._atoms.keys())
        return []

    def get(self, atom_id: str) -> NeuralAtom | None:
        """Get an atom by ID."""
        if self._store:
            return self._store.get(atom_id)
        return None


class RouterConfig:
    """Configuration for evolutionary router."""

    def __init__(self, top_k: int = 5, fitness_weight: float = 0.3):
        self.top_k = top_k
        self.fitness_weight = fitness_weight


class EvolutionaryRouter:
    """
    Base evolutionary router that can be enhanced with chemistry.
    Provides base ranking functionality.
    """

    def __init__(self, registry: AtomRegistry, config: RouterConfig):
        self.registry = registry
        self.config = config

    def rank(
        self, task: dict[str, Any], candidate_filter: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """
        Base ranking method using simple capability matching.

        Args:
            task: Task description dictionary
            candidate_filter: Optional list of atom IDs to consider

        Returns:
            List of (atom_id, score) tuples sorted by score descending
        """
        task_description = task.get("description", "")
        candidates = candidate_filter or self.registry.list_ids()

        scored_atoms = []
        for atom_id in candidates:
            atom = self.registry.get(atom_id)
            if atom and hasattr(atom, "can_handle"):
                try:
                    score = atom.can_handle(task_description)
                    if score > 0:
                        scored_atoms.append((atom_id, score))
                except Exception:
                    # If can_handle fails, give it a low score
                    scored_atoms.append((atom_id, 0.1))

        # Sort by score descending
        scored_atoms.sort(key=lambda x: x[1], reverse=True)
        return scored_atoms[: self.config.top_k]


class MolecularRouter(EvolutionaryRouter):
    """
    Chemistry-flavored router: combines base evolutionary score with
    context-free 'bondability' to other high-fitness atoms (promotes stable molecules).
    """

    def __init__(self, registry: AtomRegistry, **kwargs):
        cfg = kwargs.pop("config", None) or RouterConfig()
        super().__init__(registry=registry, config=cfg)

    def rank(
        self, task: dict[str, Any], candidate_filter: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """
        Enhanced ranking that considers both base capability and chemical bonding potential.
        """
        base = super().rank(task, candidate_filter)  # [(atom_id, score)]
        if not base:
            return base

        # compute a global "bond factor" as average bond strength to top-K peers
        top_ids = [aid for aid, _ in base[: min(5, len(base))]]
        top_atoms = [
            self.registry.get(aid) for aid in top_ids if self.registry.get(aid)
        ]

        boosted = []
        for aid, base_score in base:
            atom = self.registry.get(aid)
            if not atom or not top_atoms:
                boosted.append((aid, base_score))
                continue

            bonds = []
            for peer in top_atoms:
                # Check that both atoms are valid and have metadata
                if (
                    peer
                    and atom
                    and hasattr(peer, "metadata")
                    and hasattr(atom, "metadata")
                    and peer.metadata
                    and atom.metadata
                ):
                    peer_name = getattr(peer.metadata, "name", str(peer))
                    atom_name = getattr(atom.metadata, "name", str(atom))

                    if peer_name != atom_name:
                        try:
                            bond_str = bond_strength(atom, peer)
                            bonds.append(bond_str)
                        except Exception:
                            # If bond calculation fails, assume weak bond
                            bonds.append(0.1)

            bond_factor = sum(bonds) / len(bonds) if bonds else 0.0

            # blend: prefer atoms that both score well AND bond well with peers
            final = float(0.8 * base_score + 0.2 * bond_factor)
            boosted.append((aid, final))

        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted
