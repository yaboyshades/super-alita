"""
Test the genealogy system functionality.
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta

from src.core.genealogy import (
    LineageNode, LineageEdge, GenealogyTracer,
    trace_birth, trace_fitness, get_global_tracer
)


def test_lineage_node_creation():
    """Test lineage node creation and properties."""
    
    node = LineageNode(
        key="test_node",
        node_type="skill",
        birth_event="creation",
        parent_keys=["parent1", "parent2"],
        metadata={"test": "value"}
    )
    
    assert node.key == "test_node"
    assert node.node_type == "skill"
    assert node.birth_event == "creation"
    assert node.parent_keys == ["parent1", "parent2"]
    assert node.metadata == {"test": "value"}
    assert node.is_active


def test_lineage_node_fitness():
    """Test lineage node fitness tracking."""
    
    node = LineageNode("test", "skill")
    
    # Initially no fitness scores
    assert node.get_average_fitness() == 0.0
    
    # Add fitness scores
    node.add_fitness_score(0.8)
    node.add_fitness_score(0.9)
    node.add_fitness_score(0.7)
    
    assert len(node.fitness_scores) == 3
    assert node.get_average_fitness() == 0.8


def test_lineage_edge_creation():
    """Test lineage edge creation."""
    
    edge = LineageEdge(
        parent_key="parent",
        child_key="child",
        edge_type="influence",
        strength=0.8,
        metadata={"context": "test"}
    )
    
    assert edge.parent_key == "parent"
    assert edge.child_key == "child"
    assert edge.edge_type == "influence"
    assert edge.strength == 0.8
    assert edge.metadata == {"context": "test"}


def test_genealogy_tracer_basic_operations():
    """Test basic genealogy tracer operations."""
    
    tracer = GenealogyTracer()
    
    # Add nodes
    node1 = tracer.add_node("node1", "skill", "creation")
    node2 = tracer.add_node("node2", "skill", "evolution", ["node1"])
    
    assert "node1" in tracer.nodes
    assert "node2" in tracer.nodes
    assert node2.parent_keys == ["node1"]
    
    # Check graph structure
    assert tracer.graph.has_node("node1")
    assert tracer.graph.has_node("node2")
    assert tracer.graph.has_edge("node1", "node2")


def test_genealogy_tracer_generations():
    """Test generation tracking."""
    
    tracer = GenealogyTracer()
    
    # Add root node
    tracer.add_node("root", "skill")
    assert tracer.get_generation("root") == 0
    
    # Add child
    tracer.add_node("child", "skill", parent_keys=["root"])
    assert tracer.get_generation("child") == 1
    
    # Add grandchild
    tracer.add_node("grandchild", "skill", parent_keys=["child"])
    assert tracer.get_generation("grandchild") == 2
    
    # Add node with multiple parents
    tracer.add_node("multi", "skill", parent_keys=["child", "grandchild"])
    assert tracer.get_generation("multi") == 3  # Max parent gen + 1


def test_genealogy_tracer_ancestry():
    """Test ancestry and descendant queries."""
    
    tracer = GenealogyTracer()
    
    # Build lineage: root -> child1 -> grandchild
    #                      -> child2
    tracer.add_node("root", "skill")
    tracer.add_node("child1", "skill", parent_keys=["root"])
    tracer.add_node("child2", "skill", parent_keys=["root"])
    tracer.add_node("grandchild", "skill", parent_keys=["child1"])
    
    # Test ancestors
    ancestors = tracer.get_ancestors("grandchild")
    assert "child1" in ancestors
    assert "root" in ancestors
    assert "child2" not in ancestors
    
    # Test descendants
    descendants = tracer.get_descendants("root")
    assert "child1" in descendants
    assert "child2" in descendants
    assert "grandchild" in descendants
    
    # Test with depth limit
    ancestors_limited = tracer.get_ancestors("grandchild", max_depth=1)
    assert "child1" in ancestors_limited
    assert "root" not in ancestors_limited


def test_genealogy_tracer_fitness_evolution():
    """Test fitness evolution analysis."""
    
    tracer = GenealogyTracer()
    
    # Create nodes across generations with fitness scores
    tracer.add_node("gen0_1", "skill")
    tracer.update_fitness("gen0_1", 0.5)
    tracer.update_fitness("gen0_1", 0.6)
    
    tracer.add_node("gen1_1", "skill", parent_keys=["gen0_1"])
    tracer.update_fitness("gen1_1", 0.7)
    tracer.update_fitness("gen1_1", 0.8)
    
    tracer.add_node("gen1_2", "skill", parent_keys=["gen0_1"])
    tracer.update_fitness("gen1_2", 0.4)
    
    # Analyze fitness evolution
    analysis = tracer.analyze_fitness_evolution("skill")
    
    assert analysis["total_generations"] == 2
    assert 0 in analysis["generation_stats"]
    assert 1 in analysis["generation_stats"]
    
    gen0_stats = analysis["generation_stats"][0]
    assert gen0_stats["count"] == 1
    assert gen0_stats["mean_fitness"] == 0.55  # (0.5 + 0.6) / 2
    
    gen1_stats = analysis["generation_stats"][1]
    assert gen1_stats["count"] == 2


def test_genealogy_tracer_evolutionary_branches():
    """Test evolutionary branch detection."""
    
    tracer = GenealogyTracer()
    
    # Create branching structure
    tracer.add_node("root", "skill")
    tracer.add_node("branch1", "skill", parent_keys=["root"])
    tracer.add_node("branch2", "skill", parent_keys=["root"])
    tracer.add_node("branch3", "skill", parent_keys=["root"])
    
    # Add fitness scores
    tracer.update_fitness("branch1", 0.9)
    tracer.update_fitness("branch2", 0.7)
    tracer.update_fitness("branch3", 0.5)
    
    # Find branches
    branches = tracer.find_evolutionary_branches()
    
    assert len(branches) == 1
    branch = branches[0]
    assert branch["branch_point"] == "root"
    assert len(branch["children"]) == 3
    assert "branch1" in branch["children"]


def test_genealogy_tracer_export_import():
    """Test genealogy export and import."""
    
    tracer = GenealogyTracer()
    
    # Build test genealogy
    tracer.add_node("root", "skill", "creation", metadata={"test": "data"})
    tracer.add_node("child", "skill", "evolution", ["root"])
    tracer.update_fitness("root", 0.8)
    tracer.update_fitness("child", 0.9)
    
    # Export to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        tracer.export_to_json(temp_path)
        
        # Create new tracer and import
        new_tracer = GenealogyTracer()
        new_tracer.import_from_json(temp_path)
        
        # Verify import
        assert "root" in new_tracer.nodes
        assert "child" in new_tracer.nodes
        assert new_tracer.nodes["root"].metadata == {"test": "data"}
        assert len(new_tracer.nodes["root"].fitness_scores) == 1
        assert new_tracer.get_generation("child") == 1
        
    finally:
        os.unlink(temp_path)


def test_genealogy_tracer_pruning():
    """Test genealogy pruning functionality."""
    
    tracer = GenealogyTracer()
    
    # Create nodes with different fitness and ages
    tracer.add_node("good", "skill")
    tracer.update_fitness("good", 0.9)
    
    tracer.add_node("old_bad", "skill")
    tracer.nodes["old_bad"].birth_time = datetime.utcnow() - timedelta(days=8)
    tracer.update_fitness("old_bad", 0.05)
    
    tracer.add_node("recent_bad", "skill")
    tracer.update_fitness("recent_bad", 0.05)
    
    # Prune with 7-day age threshold
    pruned = tracer.prune_lineage(fitness_threshold=0.1, age_threshold_hours=168)
    
    # Only old, low-fitness node should be pruned
    assert "old_bad" in pruned
    assert "good" not in pruned
    assert "recent_bad" not in pruned  # Too recent despite low fitness
    
    # Check node is deactivated
    assert not tracer.nodes["old_bad"].is_active


def test_genealogy_tracer_statistics():
    """Test genealogy statistics."""
    
    tracer = GenealogyTracer()
    
    # Add some test data
    tracer.add_node("skill1", "skill")
    tracer.add_node("skill2", "skill")
    tracer.add_node("memory1", "memory")
    tracer.update_fitness("skill1", 0.8)
    tracer.update_fitness("skill2", 0.6)
    
    stats = tracer.get_statistics()
    
    assert stats["total_nodes"] == 3
    assert stats["active_nodes"] == 3
    assert stats["node_types"]["skill"] == 2
    assert stats["node_types"]["memory"] == 1
    assert stats["fitness_stats"]["total_scores"] == 2
    assert stats["fitness_stats"]["mean_fitness"] == 0.7


def test_global_tracer():
    """Test global tracer singleton."""
    
    tracer1 = get_global_tracer()
    tracer2 = get_global_tracer()
    
    # Should be same instance
    assert tracer1 is tracer2


def test_trace_birth_helper():
    """Test trace_birth helper function."""
    
    # Clear global tracer
    global_tracer = get_global_tracer()
    global_tracer.nodes.clear()
    global_tracer.edges.clear()
    
    # Use helper function
    node = trace_birth(
        key="test_skill",
        node_type="skill",
        birth_event="creation",
        parent_keys=["parent1"],
        metadata={"test": "data"}
    )
    
    assert node.key == "test_skill"
    assert "test_skill" in global_tracer.nodes


def test_trace_fitness_helper():
    """Test trace_fitness helper function."""
    
    global_tracer = get_global_tracer()
    
    # Add a node first
    trace_birth("fitness_test", "skill")
    
    # Trace fitness
    trace_fitness("fitness_test", 0.85)
    
    node = global_tracer.nodes["fitness_test"]
    assert 0.85 in node.fitness_scores


if __name__ == "__main__":
    pytest.main([__file__])
