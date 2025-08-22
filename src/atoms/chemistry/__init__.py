# Chemistry Layer for Super Alita Neural Atoms
# Provides chemical bonding metaphors for atom composition and energy-based routing

from .molecule_assembler import assemble_molecule
from .primitives import (
    CatalystQuark,
    EnergyState,
    OrbitalContext,
    bond_strength,
    io_overlap,
)
from .router_chemistry import MolecularRouter

__all__ = [
    "CatalystQuark",
    "EnergyState",
    "MolecularRouter",
    "OrbitalContext",
    "assemble_molecule",
    "bond_strength",
    "io_overlap",
]
