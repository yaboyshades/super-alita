"""
Test the neural atom system functionality.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta

from src.core.neural_atom import NeuralAtom, NeuralStore, create_skill_atom, create_memory_atom


@pytest.mark.asyncio
async def test_neural_atom_creation():
    """Test basic neural atom creation."""
    
    atom = NeuralAtom(
        key="test_atom",
        default_value="test_value",
        vector=np.array([1.0, 2.0, 3.0]),
        parent_keys=["parent1", "parent2"],
        birth_event="test_creation"
    )
    
    assert atom.key == "test_atom"
    assert atom.default_value == "test_value"
    assert np.array_equal(atom.vector, np.array([1.0, 2.0, 3.0]))
    assert atom.parent_keys == ["parent1", "parent2"]
    assert atom.birth_event == "test_creation"
    assert atom.access_count == 0


def test_atom_access_tracking():
    """Test atom access tracking."""
    
    atom = NeuralAtom("test", "value")
    initial_time = atom.last_accessed
    
    # Update access
    atom.update_access()
    
    assert atom.access_count == 1
    assert atom.last_accessed > initial_time
    assert atom.get_age_seconds() >= 0


@pytest.mark.asyncio
async def test_neural_store_basic_operations():
    """Test basic neural store operations."""
    
    store = NeuralStore()
    
    # Register an atom
    atom = NeuralAtom("test_key", "test_value")
    await store.register_atom(atom)
    
    # Test retrieval
    value = await store.get("test_key")
    assert value == "test_value"
    
    # Test setting
    await store.set("test_key", "new_value")
    value = await store.get("test_key")
    assert value == "new_value"
    
    # Test atom listing
    atoms = await store.list_atoms()
    assert "test_key" in atoms


@pytest.mark.asyncio
async def test_neural_store_subscriptions():
    """Test neural store subscription system."""
    
    store = NeuralStore()
    atom = NeuralAtom("sub_test", "initial")
    await store.register_atom(atom)
    
    # Track subscription calls
    callback_data = []
    
    async def test_callback(old_value, new_value):
        callback_data.append((old_value, new_value))
    
    # Subscribe to changes
    sub_id = await store.subscribe("sub_test", test_callback)
    
    # Change value
    await store.set("sub_test", "changed")
    
    # Wait for async callback
    await asyncio.sleep(0.01)
    
    # Check callback was called
    assert len(callback_data) == 1
    assert callback_data[0] == ("initial", "changed")
    
    # Unsubscribe
    success = await store.unsubscribe("sub_test", sub_id)
    assert success


@pytest.mark.asyncio
async def test_computed_atoms():
    """Test computed atom functionality."""
    
    store = NeuralStore()
    
    # Register base atoms
    await store.register_atom(NeuralAtom("a", 10))
    await store.register_atom(NeuralAtom("b", 20))
    
    # Create computed atom
    def compute_sum(deps):
        return deps.get("a", 0) + deps.get("b", 0)
    
    await store.create_computed_atom("sum", compute_sum, ["a", "b"])
    
    # Test computed value
    value = await store.get("sum")
    assert value == 30
    
    # Update dependency and test recomputation
    await store.set("a", 15)
    await asyncio.sleep(0.01)  # Allow recomputation
    
    value = await store.get("sum")
    assert value == 35


@pytest.mark.asyncio
async def test_genealogy_tracking():
    """Test genealogy tracking in neural store."""
    
    store = NeuralStore()
    
    # Create parent atoms
    parent1 = NeuralAtom("parent1", "value1")
    parent2 = NeuralAtom("parent2", "value2")
    
    await store.register_atom(parent1)
    await store.register_atom(parent2)
    
    # Create child atom with lineage
    child = NeuralAtom(
        "child",
        "child_value",
        parent_keys=["parent1", "parent2"],
        birth_event="combination"
    )
    
    await store.register_with_lineage(
        child,
        parents=[parent1, parent2],
        birth_event="combination",
        lineage_metadata={"operation": "merge"}
    )
    
    # Test genealogy queries
    children = await store.get_children("parent1")
    assert "child" in children
    
    descendants = await store.get_descendants("parent1")
    assert "child" in descendants


@pytest.mark.asyncio
async def test_store_pruning():
    """Test neural store pruning functionality."""
    
    store = NeuralStore()
    
    # Create atoms with different ages and access patterns
    old_atom = NeuralAtom("old", "value")
    old_atom.created_at = datetime.utcnow() - timedelta(days=7)
    old_atom.access_count = 0
    
    recent_atom = NeuralAtom("recent", "value")
    recent_atom.access_count = 10
    
    await store.register_atom(old_atom)
    await store.register_atom(recent_atom)
    
    # Prune unfit atoms
    pruned = await store.prune_unfit(fitness_threshold=0.3)
    
    # Old, unused atom should be pruned
    assert "old" in pruned
    assert "recent" not in pruned


@pytest.mark.asyncio 
async def test_skill_atom_creation():
    """Test skill atom creation helper."""
    
    store = NeuralStore()
    
    skill_obj = {"code": "def test(): pass", "description": "Test skill"}
    embedding = np.array([1.0, 0.0, 0.0])
    
    atom = await create_skill_atom(
        store=store,
        skill_name="test_skill",
        skill_obj=skill_obj,
        parent_skills=["base_skill"],
        embedding=embedding,
        metadata={"type": "test"}
    )
    
    assert atom.key == "skill:test_skill"
    assert atom.default_value == skill_obj
    assert np.array_equal(atom.vector, embedding)
    assert atom.parent_keys == ["base_skill"]


@pytest.mark.asyncio
async def test_memory_atom_creation():
    """Test memory atom creation helper."""
    
    store = NeuralStore()
    
    content = "This is a test memory"
    embedding = np.array([0.5, 0.5, 0.0])
    
    atom = await create_memory_atom(
        store=store,
        memory_key="test_memory",
        content=content,
        memory_type="episodic",
        embedding=embedding,
        metadata={"importance": 0.8}
    )
    
    assert atom.key == "memory:episodic:test_memory"
    assert atom.default_value == content
    assert atom.lineage_metadata["memory_type"] == "episodic"
    assert atom.lineage_metadata["importance"] == 0.8


@pytest.mark.asyncio
async def test_store_export():
    """Test store state export."""
    
    store = NeuralStore()
    
    # Add some atoms
    await store.register_atom(NeuralAtom("atom1", "value1"))
    await store.register_atom(NeuralAtom("atom2", "value2"))
    
    # Export state
    state = await store.export_state()
    
    assert "atoms" in state
    assert "values" in state
    assert "dependencies" in state
    assert len(state["atoms"]) == 2
    assert state["values"]["atom1"] == "value1"


def test_store_stats():
    """Test store statistics."""
    
    store = NeuralStore()
    
    # Get initial stats
    stats = store.get_stats()
    
    assert "total_atoms" in stats
    assert "computed_atoms" in stats
    assert "total_subscriptions" in stats
    assert stats["total_atoms"] == 0


if __name__ == "__main__":
    pytest.main([__file__])
