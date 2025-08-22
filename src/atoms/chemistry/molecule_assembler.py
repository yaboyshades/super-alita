from __future__ import annotations

import hashlib
import logging
from itertools import combinations
from typing import TYPE_CHECKING, Any

from .primitives import bond_strength  # MOVED to top-level import

logger = logging.getLogger(__name__)  # NEW

# ------------------------------------------------------------------
# Robust, type-checker-friendly imports
# ------------------------------------------------------------------
if TYPE_CHECKING:
    from src.core.atom_registry import AtomRegistry
    from src.core.neural_atom import NeuralAtom, NeuralAtomMetadata
else:  # Runtime fallback stubs (keep mypy silent)
    NeuralAtom = Any  # type: ignore[assignment]
    NeuralAtomMetadata = Any  # type: ignore[assignment]
    AtomRegistry = Any  # type: ignore[assignment]


class CompositePlanStep:
    """A step in a composite skill plan."""

    def __init__(self, atom_id: str, inputs: dict[str, Any]):
        self.atom_id = atom_id
        self.inputs = inputs


def make_composite_skill(
    skill_id: str,
    version: str,
    title: str,
    sub_steps: list[CompositePlanStep],
    description: str = "",
    tags: list[str] | None = None,
) -> NeuralAtom:
    """
    Create a composite skill atom from a set of sub-steps.

    This creates a new Neural Atom that represents a workflow composed of other atoms.
    """
    metadata = NeuralAtomMetadata(
        name=skill_id,
        description=description or f"Composite skill: {title}",
        capabilities=tags or ["composite", "auto-assembled"],
        version=version,
    )

    class CompositeSkillAtom(NeuralAtom):
        def __init__(
            self, metadata: NeuralAtomMetadata, steps: list[CompositePlanStep]
        ) -> None:
            super().__init__(metadata)
            self.key = metadata.name  # For NeuralStore compatibility
            self.steps = steps

        async def execute(self, input_data: Any | None = None) -> Any:
            """Execute all sub-steps in sequence."""
            results = []
            current_input = input_data

            for step in self.steps:
                # This would need to be enhanced to actually execute the sub-atoms
                # For now, just return the plan structure
                step_result = {
                    "atom_id": step.atom_id,
                    "inputs": step.inputs,
                    "input_received": current_input,
                }
                results.append(step_result)
                current_input = step_result  # Chain outputs to inputs

            return {
                "composite_skill": skill_id,
                "sub_step_results": results,
                "success": True,
            }

        def get_embedding(self) -> list[float]:
            """Generate embedding based on sub-skills."""
            # Simple approach: average embeddings of constituent atoms would go here
            # For now, return a basic embedding
            hash_obj = hashlib.md5(skill_id.encode())
            hash_bytes = hash_obj.digest()
            return [float(b) / 255.0 for b in hash_bytes[:128]]

        def can_handle(self, task_description: str) -> float:
            """Assess if this composite can handle a task."""
            # Check if any constituent atoms might handle it
            if "composite" in task_description.lower():
                return 0.8
            if any(step.atom_id in task_description.lower() for step in self.steps):
                return 0.7
            return 0.2  # Default low capability

    return CompositeSkillAtom(metadata, sub_steps)


def assemble_molecule(
    registry: AtomRegistry,
    min_atoms: int = 2,
    bond_threshold: float = 0.75,
    title: str = "Auto-Bonded Molecule",
    prefix: str = "molecule.auto",
) -> str | None:
    """
    Scan registry, find a tightly bonded clique of atoms (size >= min_atoms),
    and register a composite skill out of them.
    """
    # ------------------------------------------------------------------
    # Filter registry atoms without creating a list[NeuralAtom | None] type
    # ------------------------------------------------------------------
    all_ids = registry.list_ids()
    atoms: list[NeuralAtom] = []
    for aid in all_ids:
        atom = registry.get(aid)
        if atom is not None:
            atoms.append(atom)

    if len(atoms) < min_atoms:
        return None

    best_combo = None
    best_score = 0.0

    # Search for the best combination of atoms
    for k in range(min_atoms, min(min_atoms + 3, len(atoms) + 1)):
        for combo in combinations(atoms, k):
            try:
                bonds = [bond_strength(a, b) for a, b in combinations(combo, 2)]
                if not bonds:
                    continue
                avg_bond = sum(bonds) / len(bonds)
                if avg_bond > best_score and avg_bond >= bond_threshold:
                    best_score = avg_bond
                    best_combo = combo
            except (ValueError, TypeError):
                # Skip combinations that fail bond calculation
                continue

    if not best_combo:
        return None

    # Create composite skill from the best combination
    steps = []
    for atom in best_combo:
        if hasattr(atom, "metadata"):
            atom_id = getattr(atom.metadata, "name", str(atom))
        else:
            atom_id = str(atom)
        steps.append(CompositePlanStep(atom_id=atom_id, inputs={}))

    # Generate unique skill ID
    atom_names = []
    for atom in best_combo:
        if hasattr(atom, "metadata") and atom.metadata:
            name = getattr(atom.metadata, "name", "unknown")
            # Clean the name for use in ID
            clean_name = name.replace("atom.", "").replace("_", "-")
            atom_names.append(clean_name)
        else:
            atom_names.append("unknown")

    skill_id = f"{prefix}.{'.'.join(atom_names)}"

    # Create the composite skill
    skill = make_composite_skill(
        skill_id=skill_id,
        version="v1",
        title=title,
        sub_steps=steps,
        description=f"Auto-assembled molecule (avg_bond={best_score:.2f})",
        tags=["chemistry", "auto", "molecule"],
    )

    # Register the skill in the store if possible
    try:
        if hasattr(registry, "_store") and registry._store:
            registry._store.register(skill)
    except (AttributeError, ValueError, RuntimeError) as e:  # NARROWED
        # Log and continue - avoid crashing the assembler while keeping diagnostics
        logger.warning(
            "Unable to register composite skill '%s' in AtomRegistry: %s",
            skill_id,
            e,
        )

    # Return the skill identifier
    skill_key = getattr(skill, "key", skill_id)
    if not skill_key and hasattr(skill, "metadata"):
        skill_key = getattr(skill.metadata, "name", skill_id)

    return skill_key or skill_id


def _validate_molecule_structure(self, structure: dict[str, Any]) -> bool:
    """Validate the molecular structure for chemical feasibility."""
    try:
        # Check for required fields
        if not all(key in structure for key in ["atoms", "bonds"]):
            return False

        # Validate atoms
        atoms = structure.get("atoms", [])
        if not isinstance(atoms, list) or len(atoms) == 0:
            return False

        # Validate bonds
        bonds = structure.get("bonds", [])
        if not isinstance(bonds, list):
            return False

        # Check bond consistency
        atom_count = len(atoms)
        for bond in bonds:
            if not isinstance(bond, dict):
                return False
            atom1, atom2 = bond.get("atom1", -1), bond.get("atom2", -1)
            if not (0 <= atom1 < atom_count and 0 <= atom2 < atom_count):
                return False

        return True

    except Exception as e:
        logger.error(f"Structure validation error: {e}")
        return False
