#!/usr/bin/env python3
"""
Chemistry Layer Demo for Super Alita

This demo shows how the chemistry layer enhances atom selection and composition.
It creates "water" molecules from hydrogen and oxygen atoms as a conceptual example.
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockAtom:
    """Mock atom for demonstration purposes."""

    def __init__(
        self,
        name: str,
        capabilities: list,
        latency_ms: float = 100,
        success_rate: float = 0.9,
    ):
        self.metadata = MockMetadata(name, capabilities)
        self.average_latency_ms = latency_ms

    def can_handle(self, task_description: str) -> float:
        """Return confidence score for handling a task."""
        task_lower = task_description.lower()

        # Simple keyword matching
        for capability in self.metadata.capabilities:
            if capability.lower() in task_lower:
                return 0.8

        return 0.1


class MockMetadata:
    """Mock metadata for demonstration."""

    def __init__(self, name: str, capabilities: list):
        self.name = name
        self.capabilities = capabilities
        self.success_rate = 0.9


class MockRegistry:
    """Mock registry for demonstration."""

    def __init__(self):
        self.atoms = {}

    def add_atom(self, atom):
        self.atoms[atom.metadata.name] = atom

    def list_ids(self):
        return list(self.atoms.keys())

    def get(self, atom_id: str):
        return self.atoms.get(atom_id)


async def demo_chemistry_water_molecule():
    """
    Demo: Create a "water" molecule from hydrogen and oxygen atoms.
    """
    print("🧪 Chemistry Layer Demo: Creating Water Molecule")
    print("=" * 50)

    # Create mock atoms representing hydrogen and oxygen
    hydrogen = MockAtom(
        name="hydrogen_atom",
        capabilities=["reactive", "bonding", "light"],
        latency_ms=50,
        success_rate=0.95,
    )

    oxygen = MockAtom(
        name="oxygen_atom",
        capabilities=["reactive", "bonding", "oxidation"],
        latency_ms=80,
        success_rate=0.92,
    )

    carbon = MockAtom(
        name="carbon_atom",
        capabilities=["stable", "structural"],
        latency_ms=200,  # Slower, less reactive
        success_rate=0.85,
    )

    # Create registry and add atoms
    registry = MockRegistry()
    registry.add_atom(hydrogen)
    registry.add_atom(oxygen)
    registry.add_atom(carbon)

    print(f"📦 Created {len(registry.list_ids())} atoms: {registry.list_ids()}")

    # Import chemistry components
    try:
        from .molecule_assembler import assemble_molecule
        from .primitives import bond_strength
        from .router_chemistry import AtomRegistry, MolecularRouter

        # Wrap mock registry in chemistry registry
        chem_registry = AtomRegistry()
        chem_registry._atoms = registry.atoms  # Direct assignment for demo
        chem_registry.list_ids = registry.list_ids
        chem_registry.get = registry.get

        print("\n🔬 Calculating bond strengths:")

        # Calculate bond strengths between atoms
        h_o_bond = bond_strength(hydrogen, oxygen)
        h_c_bond = bond_strength(hydrogen, carbon)
        o_c_bond = bond_strength(oxygen, carbon)

        print(f"  H-O bond strength: {h_o_bond:.3f}")
        print(f"  H-C bond strength: {h_c_bond:.3f}")
        print(f"  O-C bond strength: {o_c_bond:.3f}")

        # Create molecular router
        router = MolecularRouter(chem_registry)

        # Test routing for different tasks
        print("\n🎯 Testing chemistry-enhanced routing:")

        tasks = [
            {"description": "perform reactive bonding operation"},
            {"description": "create stable structural framework"},
            {"description": "oxidation reaction needed"},
        ]

        for task in tasks:
            ranked = router.rank(task)
            if ranked:
                best_atom, score = ranked[0]
                print(f"  Task: '{task['description']}'")
                print(f"  → Selected: {best_atom} (score: {score:.3f})")
            else:
                print(f"  Task: '{task['description']}' → No suitable atom found")

        # Auto-assemble molecules
        print("\n⚗️ Auto-assembling molecules:")

        molecule_id = assemble_molecule(
            registry=chem_registry,
            min_atoms=2,
            bond_threshold=0.5,  # Lower threshold for demo
            title="Water Molecule",
            prefix="demo.water",
        )

        if molecule_id:
            print(f"  ✅ Created molecule: {molecule_id}")
        else:
            print("  ❌ No suitable molecular combinations found")

        print("\n🎉 Chemistry layer demo completed successfully!")

    except ImportError as e:
        print(f"❌ Chemistry layer import failed: {e}")
        print("Make sure the chemistry modules are properly installed.")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


async def demo_planner_integration():
    """
    Demo: Show how chemistry integrates with planning.
    """
    print("\n🤖 Planner Integration Demo")
    print("=" * 30)

    try:
        # This would normally integrate with the real planner
        print("📋 Planning workflow:")
        print("  1. User requests a task")
        print("  2. Chemistry router analyzes atom compatibility")
        print("  3. Best-bonded atoms selected for execution")
        print("  4. Molecules auto-assembled for future use")
        print("  5. Task executed with optimized atom selection")

        print("\n💡 Chemistry benefits:")
        print("  ✓ Better tool selection based on performance affinity")
        print("  ✓ Auto-discovery of composite skills")
        print("  ✓ Emergent capability creation")
        print("  ✓ Self-improving atom ecosystem")

    except Exception as e:
        print(f"❌ Planner integration demo failed: {e}")


if __name__ == "__main__":
    print("🧬 Super Alita Chemistry Layer Demo")
    print("Demonstrating chemical bonding for Neural Atoms")
    print()

    # Run demos
    asyncio.run(demo_chemistry_water_molecule())
    asyncio.run(demo_planner_integration())

    print("\n🎯 Next Steps:")
    print("1. Integrate MolecularRouter into existing PlannerPlugin")
    print("2. Add chemistry metrics to telemetry system")
    print("3. Enable auto-molecule discovery in production")
    print("4. Add orbital/catalyst context for safety policies")
