from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from src.core.atom_registry import AtomRegistry
    from src.core.neural_atom import NeuralAtom, NeuralAtomMetadata
else:  # Runtime fallbacks
    NeuralAtom = Any  # type: ignore
    NeuralAtomMetadata = Any  # type: ignore
    AtomRegistry = Any  # type: ignore


class CompositePlanStep:
    """Single step definition within a composite skill."""

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
    """Create a composite skill wrapper aggregating sequential steps."""
    metadata = NeuralAtomMetadata(  # type: ignore[call-arg]
        name=skill_id,
        description=description or f"Composite skill: {title}",
        capabilities=tags or ["composite", "auto-assembled"],
        version=version,
    )

    class _CompositeSkill(NeuralAtom):  # type: ignore[misc]
        def __init__(self, meta: NeuralAtomMetadata, steps: list[CompositePlanStep]):  # type: ignore[override]
            super().__init__(
                key=meta.name,
                value={"steps": [s.atom_id for s in steps]},
                metadata=meta,
                birth_event=f"composite_skill:{meta.name}",
            )
            self.steps = steps

        async def execute(self, input_data: Any | None = None) -> Any:
            chain: list[dict[str, Any]] = []
            current = input_data
            for step in self.steps:
                out = {
                    "atom_id": step.atom_id,
                    "inputs": step.inputs,
                    "input_received": current,
                }
                chain.append(out)
                current = out
            return {
                "composite_skill": skill_id,
                "sub_step_results": chain,
                "success": True,
            }

        def can_handle(self, task_description: str) -> float:
            lower = task_description.lower()
            if "composite" in lower:
                return 0.8
            if any(step.atom_id in lower for step in self.steps):
                return 0.7
            return 0.2

        def get_embedding(self) -> list[float]:  # deterministic stub
            return [0.01 * (i + 1) for i in range(16)]

    return _CompositeSkill(metadata, sub_steps)


def assemble_molecule(
    registry: AtomRegistry,
    min_atoms: int = 2,
    title: str = "Auto-Bonded Molecule",
    prefix: str = "molecule.auto",
) -> str | None:
    """Assemble a composite skill from the first N registered atoms."""
    try:
        ids = list(registry.list_ids())  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return None
    if len(ids) < min_atoms:
        return None
    chosen = ids[:min_atoms]
    steps = [CompositePlanStep(atom_id=str(a), inputs={}) for a in chosen]
    skill_id = f"{prefix}.{'.'.join(chosen)}"
    skill = make_composite_skill(
        skill_id=skill_id,
        version="v1",
        title=title,
        sub_steps=steps,
        description="Auto-assembled (simplified)",
        tags=["chemistry", "auto", "molecule"],
    )
    try:  # best-effort registration
        if hasattr(registry, "_store") and registry._store:
            registry._store.register(skill)  # type: ignore[call-arg]
    except Exception:  # pragma: no cover
        pass
    return getattr(skill, "key", skill_id) or skill_id


def _validate_molecule_structure(structure: dict[str, Any]) -> bool:
    """Simple structural validation utility used by tests."""
    atoms = structure.get("atoms")
    bonds = structure.get("bonds")
    if not isinstance(atoms, list) or not atoms:
        return False
    if not isinstance(bonds, list):
        return False
    n = len(atoms)
    for b in bonds:
        if not isinstance(b, dict):
            return False
        a1 = b.get("atom1")
        a2 = b.get("atom2")
        if not isinstance(a1, int) or not isinstance(a2, int):
            return False
        if a1 < 0 or a1 >= n or a2 < 0 or a2 >= n:
            return False
    return True
