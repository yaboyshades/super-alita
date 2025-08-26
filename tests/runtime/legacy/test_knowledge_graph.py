"""
Test the Knowledge Graph implementation
"""
import pytest
pytest.skip("legacy test", allow_module_level=True)

import asyncio
import tempfile
from pathlib import Path

from core.knowledge import KnowledgeStore, AtomType, BondType, KnowledgeGraphPlugin
from core.knowledge.handlers import KnowledgeGraphEventHandlers
from core.telemetry.simple_event_bus import SimpleEventBus
from core.cortex import create_cortex_runtime
from core.events import create_event

async def test_knowledge_store():
    """Test basic knowledge store functionality"""
    print("Testing Knowledge Store...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    
    try:
        with KnowledgeStore(db_path) as store:
            # Test atom creation
            print("  Creating atoms...")
            concept_atom = store.create_atom(
                atom_type=AtomType.CONCEPT,
                content={"name": "Machine Learning", "description": "AI technique"},
                metadata={"source": "test"}
            )
            
            entity_atom = store.create_atom(
                atom_type=AtomType.ENTITY,
                content={"name": "Python", "type": "programming_language"},
                metadata={"source": "test"}
            )
            
            print(f"    Created concept atom: {concept_atom.atom_id}")
            print(f"    Created entity atom: {entity_atom.atom_id}")
            
            # Test idempotency
            duplicate_atom = store.create_atom(
                atom_type=AtomType.CONCEPT,
                content={"name": "Machine Learning", "description": "AI technique"},
                metadata={"source": "test_duplicate"}
            )
            
            assert concept_atom.atom_id == duplicate_atom.atom_id
            print("    ✓ Idempotency test passed")
            
            # Test bond creation
            print("  Creating bonds...")
            bond = store.create_bond(
                from_atom_id=entity_atom.atom_id,
                to_atom_id=concept_atom.atom_id,
                bond_type=BondType.RELATES_TO,
                strength=0.9,
                metadata={"relationship": "used_for"}
            )
            
            print(f"    Created bond: {bond.bond_id}")
            
            # Test retrieval
            print("  Testing retrieval...")
            retrieved_atom = store.get_atom(concept_atom.atom_id)
            assert retrieved_atom is not None
            assert retrieved_atom.content["name"] == "Machine Learning"
            
            # Test bonds
            outgoing_bonds = store.get_bonds_from_atom(entity_atom.atom_id)
            assert len(outgoing_bonds) == 1
            assert outgoing_bonds[0].to_atom_id == concept_atom.atom_id
            
            # Test search
            search_results = store.search_atoms_by_content("Machine")
            assert len(search_results) >= 1
            
            # Test statistics
            stats = store.get_statistics()
            print(f"    Statistics: {stats}")
            assert stats["total_atoms"] >= 2
            assert stats["total_bonds"] >= 1
            
            print("  ✓ Knowledge store tests passed")
    
    finally:
        # Clean up
        if db_path.exists():
            db_path.unlink()

async def test_event_handlers():
    """Test knowledge graph event handlers"""
    print("Testing Event Handlers...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    
    try:
        with KnowledgeStore(db_path) as store:
            handlers = KnowledgeGraphEventHandlers(store)
            
            # Test cortex cycle event
            print("  Testing Cortex cycle event...")
            cortex_event = create_event(
                "cortex_cycle_complete",
                source_plugin="cortex_runtime",
                metadata={
                    "cycle_id": "test_cycle_123",
                    "success": True,
                    "duration_ms": 150.5,
                    "error": None
                }
            )
            
            await handlers.handle_cortex_cycle_event(cortex_event)
            
            # Check that atoms were created
            cortex_atoms = store.get_atoms_by_type(AtomType.CORTEX_RESULT)
            event_atoms = store.get_atoms_by_type(AtomType.EVENT)
            
            assert len(cortex_atoms) >= 1
            assert len(event_atoms) >= 1
            
            print(f"    Created {len(cortex_atoms)} cortex atoms and {len(event_atoms)} event atoms")
            
            # Test concept creation
            print("  Testing manual concept creation...")
            concept_id = handlers.create_concept_atom(
                "Neural Networks",
                "Deep learning architecture",
                {"layers": 3, "activation": "relu"}
            )
            
            concept_atom = store.get_atom(concept_id)
            assert concept_atom is not None
            assert concept_atom.content["name"] == "Neural Networks"
            
            print(f"    Created concept: {concept_id}")
            
            print("  ✓ Event handlers tests passed")
    
    finally:
        # Clean up
        if db_path.exists():
            db_path.unlink()

async def test_plugin_integration():
    """Test knowledge graph plugin integration"""
    print("Testing Plugin Integration...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    
    try:
        # Create plugin and event bus
        plugin = KnowledgeGraphPlugin(db_path)
        event_bus = SimpleEventBus()
        
        # Setup plugin
        await plugin.setup(event_bus=event_bus)
        
        # Create Cortex runtime and connect everything
        runtime = create_cortex_runtime()
        await runtime.setup(event_bus=event_bus)
        
        print("  Running Cortex cycles to generate events...")
        
        # Run some test cycles
        for i in range(2):
            context = runtime.create_context(f"test_session_{i}", "test_user")
            result = await runtime.process_cycle(f"Test query {i}", context)
            await asyncio.sleep(0.01)  # Small delay
        
        await runtime.shutdown()
        
        # Check knowledge graph statistics
        stats = plugin.get_statistics()
        print(f"  Knowledge graph statistics: {stats}")
        
        assert stats["total_atoms"] >= 2  # Should have created atoms from events
        
        # Test search
        search_results = plugin.search_atoms("cortex")
        print(f"  Found {len(search_results)} atoms matching 'cortex'")
        
        # Test manual concept creation
        concept_id = plugin.create_manual_concept(
            "Test Concept",
            "A concept created for testing",
            {"test": True}
        )
        
        print(f"  Created manual concept: {concept_id}")
        
        # Test relationships
        entity_id = plugin.create_manual_entity(
            "Test Entity",
            "test_type",
            {"value": 42}
        )
        
        bond_id = plugin.link_concepts(
            concept_id,
            entity_id,
            "relates_to",
            strength=0.8
        )
        
        print(f"  Created relationship: {bond_id}")
        
        # Test related atoms
        related = plugin.get_related_atoms(concept_id)
        assert len(related) >= 1
        print(f"  Found {len(related)} related atoms")
        
        await plugin.shutdown()
        print("  ✓ Plugin integration tests passed")
    
    finally:
        # Clean up
        if db_path.exists():
            db_path.unlink()

async def test_full_integration():
    """Test full integration with telemetry and knowledge graph"""
    print("Testing Full Integration...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    
    try:
        # Create all components
        from core.telemetry.collector import TelemetryCollector
        
        knowledge_plugin = KnowledgeGraphPlugin(db_path)
        telemetry_collector = TelemetryCollector()
        event_bus = SimpleEventBus()
        runtime = create_cortex_runtime()
        
        # Connect everything
        await knowledge_plugin.setup(event_bus=event_bus)  # This subscribes to events automatically
        await event_bus.subscribe("*", telemetry_collector.collect_event)
        await runtime.setup(event_bus=event_bus)
        
        print("  Running integrated test cycle...")
        
        # Run a cognitive cycle
        context = runtime.create_context("integration_test", "test_user")
        result = await runtime.process_cycle("Integrate knowledge graph with telemetry", context)
        
        await asyncio.sleep(0.1)  # Let events process
        
        # Check results
        kg_stats = knowledge_plugin.get_statistics()
        telemetry_metrics = telemetry_collector.get_metrics()
        
        print(f"  Knowledge graph: {kg_stats['total_atoms']} atoms, {kg_stats['total_bonds']} bonds")
        print(f"  Telemetry: {telemetry_metrics.total_events} events collected")
        
        assert kg_stats["total_atoms"] >= 1
        assert telemetry_metrics.total_events >= 1
        
        # Export graph data
        graph_data = knowledge_plugin.export_graph_data()
        assert "atoms" in graph_data
        assert "bonds" in graph_data
        assert "statistics" in graph_data
        
        print(f"  Exported graph with {len(graph_data['bonds'])} bonds")
        
        await runtime.shutdown()
        await knowledge_plugin.shutdown()
        
        print("  ✓ Full integration tests passed")
    
    finally:
        # Clean up
        if db_path.exists():
            db_path.unlink()

async def main():
    """Run all knowledge graph tests"""
    print("🔗 Running Knowledge Graph Tests")
    
    await test_knowledge_store()
    await test_event_handlers()
    await test_plugin_integration()
    await test_full_integration()
    
    print("✅ All knowledge graph tests passed!")

if __name__ == "__main__":
    asyncio.run(main())