"""
Chemistry-Enhanced Planning Integration

This module provides integration patches for the existing Super Alita planner
to use the chemistry layer for enhanced atom selection and routing.

Drop-in enhancement that preserves existing functionality while adding
chemical bonding awareness.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ChemistryEnhancedPlanner:
    """
    Drop-in enhancement for the existing PlannerPlugin that adds chemistry-aware routing.

    This class can be used to enhance the existing planner without breaking changes.
    """

    def __init__(self, original_planner, neural_store):
        self.original_planner = original_planner
        self.neural_store = neural_store
        self._molecular_router = None
        self._initialize_chemistry()

    def _initialize_chemistry(self):
        """Initialize the chemistry layer components."""
        try:
            from ..chemistry import MolecularRouter
            from ..chemistry.router_chemistry import AtomRegistry

            # Create chemistry-aware router
            registry = AtomRegistry(self.neural_store)
            self._molecular_router = MolecularRouter(registry)

            logger.info("Chemistry-enhanced planning initialized")

        except ImportError as e:
            logger.warning(f"Chemistry layer not available: {e}")
            self._molecular_router = None

    async def enhanced_tool_selection(
        self, task: dict[str, Any], fallback_to_original: bool = True
    ) -> str | None:
        """
        Select tools using chemistry-aware routing with fallback to original behavior.

        Args:
            task: Task description dictionary
            fallback_to_original: Whether to fallback to original selection if chemistry fails

        Returns:
            Selected tool name or None
        """
        if not self._molecular_router:
            if fallback_to_original and hasattr(self.original_planner, "_select_tool"):
                return await self.original_planner._select_tool(task)
            return None

        try:
            # Use chemistry-enhanced routing
            ranked_atoms = self._molecular_router.rank(task)

            if ranked_atoms:
                best_atom_id, score = ranked_atoms[0]
                logger.info(
                    f"Chemistry router selected: {best_atom_id} (score: {score:.3f})"
                )

                # Map atom ID to tool name for compatibility
                tool_name = self._atom_id_to_tool_name(best_atom_id)
                return tool_name

            # Fallback to original if no chemistry match
            if fallback_to_original and hasattr(self.original_planner, "_select_tool"):
                logger.info("No chemistry match, falling back to original selection")
                return await self.original_planner._select_tool(task)

        except Exception as e:
            logger.error(f"Chemistry-enhanced selection failed: {e}")
            if fallback_to_original and hasattr(self.original_planner, "_select_tool"):
                return await self.original_planner._select_tool(task)

        return None

    def _atom_id_to_tool_name(self, atom_id: str) -> str:
        """
        Map atom ID to tool name for compatibility with existing tool execution.

        This handles the translation between chemistry atom IDs and the tool names
        expected by the existing execution system.
        """
        # Handle common atom ID patterns
        if "web" in atom_id.lower() or "search" in atom_id.lower():
            return "web_agent"
        if "memory" in atom_id.lower():
            return "memory_manager"
        if "calculator" in atom_id.lower() or "math" in atom_id.lower():
            # For dynamically created calculation tools
            return atom_id
        # Default: assume atom_id is the tool name
        return atom_id

    async def auto_discover_molecules(self) -> list[str]:
        """
        Auto-discover and register molecular combinations from existing atoms.

        Returns:
            List of newly created molecule IDs
        """
        if not self._molecular_router:
            return []

        try:
            from ..chemistry import assemble_molecule

            # Try to find molecules with different bond thresholds
            discovered_molecules = []

            for threshold in [0.8, 0.7, 0.6]:  # High to low bond requirements
                molecule_id = assemble_molecule(
                    registry=self._molecular_router.registry,
                    min_atoms=2,
                    bond_threshold=threshold,
                    title=f"Auto-Discovered Molecule (bond≥{threshold})",
                    prefix=f"molecule.auto.t{int(threshold * 10)}",
                )

                if molecule_id:
                    discovered_molecules.append(molecule_id)
                    logger.info(f"Discovered molecule: {molecule_id}")

                    # Only create one molecule per threshold for now
                    if len(discovered_molecules) >= 3:
                        break

            return discovered_molecules

        except Exception as e:
            logger.error(f"Molecule auto-discovery failed: {e}")
            return []


def create_chemistry_enhanced_planner(original_planner, neural_store):
    """
    Factory function to create a chemistry-enhanced planner.

    This provides a clean interface for integrating chemistry into existing systems.
    """
    return ChemistryEnhancedPlanner(original_planner, neural_store)


# Example integration for existing PlannerPlugin
async def patch_planner_with_chemistry(planner_plugin):
    """
    Example function showing how to patch an existing planner with chemistry.

    This demonstrates the drop-in integration approach.
    """
    if not hasattr(planner_plugin, "store") or not planner_plugin.store:
        logger.warning("Cannot patch planner: no neural store available")
        return

    # Create chemistry-enhanced wrapper
    enhanced_planner = create_chemistry_enhanced_planner(
        planner_plugin, planner_plugin.store
    )

    # Auto-discover initial molecules
    molecules = await enhanced_planner.auto_discover_molecules()
    if molecules:
        logger.info(f"Auto-discovered {len(molecules)} molecular skills: {molecules}")

    # Store the enhanced planner as an attribute
    planner_plugin._chemistry_enhanced = enhanced_planner

    # Optionally replace the tool selection method
    original_emit_tool_call = planner_plugin._emit_tool_call

    async def enhanced_emit_tool_call(
        tool_name: str, params: dict, session_id: str, conversation_id: str
    ):
        """Enhanced tool call that uses chemistry routing when available."""

        # If the tool name suggests we should use chemistry routing
        if tool_name == "auto_select" or not tool_name:
            task = {
                "description": params.get("query", params.get("description", "")),
                "parameters": params,
            }

            selected_tool = await enhanced_planner.enhanced_tool_selection(task)
            if selected_tool:
                tool_name = selected_tool

        # Call original method
        return await original_emit_tool_call(
            tool_name, params, session_id, conversation_id
        )

    # Replace the method (optional - only if you want automatic chemistry routing)
    # planner_plugin._emit_tool_call = enhanced_emit_tool_call

    logger.info("Planner enhanced with chemistry layer")


# Example usage in a demo
async def demo_chemistry_integration():
    """
    Demo function showing chemistry layer integration.
    """
    from ...core.neural_atom import NeuralAtomMetadata, NeuralStore

    # Create a neural store with some sample atoms
    store = NeuralStore()

    # Add some sample atoms (this would normally be done by the system)
    from ...core.neural_atom import TextualMemoryAtom

    # Sample atoms for demo
    NeuralAtomMetadata(
        name="web_search_atom",
        description="Search the web for information",
        capabilities=["search", "web", "information"],
    )

    NeuralAtomMetadata(
        name="calculator_atom",
        description="Perform mathematical calculations",
        capabilities=["math", "calculation", "numbers"],
    )

    memory_atom = TextualMemoryAtom(
        metadata=NeuralAtomMetadata(
            name="sample_memory",
            description="Sample memory content",
            capabilities=["memory", "storage"],
        ),
        content="Sample memory data for chemistry testing",
    )

    # Register atoms in store
    store.register(memory_atom)

    # Create chemistry-enhanced planner
    enhanced_planner = ChemistryEnhancedPlanner(None, store)

    # Auto-discover molecules
    molecules = await enhanced_planner.auto_discover_molecules()
    print(f"Demo discovered molecules: {molecules}")

    # Test enhanced tool selection
    test_task = {
        "description": "search for information about neural networks",
        "type": "search",
    }

    selected_tool = await enhanced_planner.enhanced_tool_selection(test_task)
    print(f"Demo selected tool: {selected_tool}")


if __name__ == "__main__":
    # Run demo if executed directly
    asyncio.run(demo_chemistry_integration())
