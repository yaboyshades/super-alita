"""
Genealogy system for Super Alita - tracks lineage and exports to GraphML.
Provides Darwin-Gödel style traceability for all cognitive primitives.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import networkx as nx
import json
import uuid
from pathlib import Path


@dataclass
class LineageNode:
    """A node in the genealogy graph."""
    
    key: str
    node_type: str  # "atom", "skill", "memory", "goal", "event"
    birth_event: Optional[str] = None
    birth_time: datetime = field(default_factory=datetime.utcnow)
    parent_keys: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    fitness_scores: List[float] = field(default_factory=list)
    is_active: bool = True
    
    def add_fitness_score(self, score: float) -> None:
        """Add a fitness score to this node."""
        self.fitness_scores.append(score)
    
    def get_average_fitness(self) -> float:
        """Get average fitness score."""
        if not self.fitness_scores:
            return 0.0
        return sum(self.fitness_scores) / len(self.fitness_scores)
    
    def get_age_seconds(self) -> float:
        """Get age in seconds."""
        return (datetime.utcnow() - self.birth_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "node_type": self.node_type,
            "birth_event": self.birth_event,
            "birth_time": self.birth_time.isoformat(),
            "parent_keys": self.parent_keys,
            "metadata": self.metadata,
            "fitness_scores": self.fitness_scores,
            "is_active": self.is_active,
            "average_fitness": self.get_average_fitness(),
            "age_seconds": self.get_age_seconds()
        }


@dataclass
class LineageEdge:
    """An edge in the genealogy graph."""
    
    parent_key: str
    child_key: str
    edge_type: str = "parent_child"  # "parent_child", "influence", "mutation", "merge"
    strength: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "parent_key": self.parent_key,
            "child_key": self.child_key,
            "edge_type": self.edge_type,
            "strength": self.strength,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


class GenealogyTracer:
    """
    Tracks and manages genealogy of all cognitive primitives.
    
    Provides full Darwin-Gödel style lineage tracking with export
    capabilities for analysis and visualization.
    """
    
    def __init__(self):
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: List[LineageEdge] = []
        self.graph: nx.DiGraph = nx.DiGraph()
        self._generation_counter: Dict[str, int] = {}
    
    def add_node(
        self,
        key: str,
        node_type: str,
        birth_event: Optional[str] = None,
        parent_keys: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> LineageNode:
        """Add a new node to the genealogy."""
        
        node = LineageNode(
            key=key,
            node_type=node_type,
            birth_event=birth_event,
            parent_keys=parent_keys or [],
            metadata=metadata or {}
        )
        
        self.nodes[key] = node
        self.graph.add_node(key, **node.to_dict())
        
        # Add edges to parents
        for parent_key in node.parent_keys:
            self.add_edge(parent_key, key, "parent_child")
        
        # Track generation
        if parent_keys:
            max_parent_gen = max(
                self._generation_counter.get(pk, 0) for pk in parent_keys
            )
            self._generation_counter[key] = max_parent_gen + 1
        else:
            self._generation_counter[key] = 0
        
        return node
    
    def add_edge(
        self,
        parent_key: str,
        child_key: str,
        edge_type: str = "parent_child",
        strength: float = 1.0,
        metadata: Dict[str, Any] = None
    ) -> LineageEdge:
        """Add an edge between nodes."""
        
        edge = LineageEdge(
            parent_key=parent_key,
            child_key=child_key,
            edge_type=edge_type,
            strength=strength,
            metadata=metadata or {}
        )
        
        self.edges.append(edge)
        self.graph.add_edge(
            parent_key, 
            child_key, 
            **edge.to_dict()
        )
        
        return edge
    
    def update_fitness(self, key: str, fitness_score: float) -> None:
        """Update fitness score for a node."""
        if key in self.nodes:
            self.nodes[key].add_fitness_score(fitness_score)
            # Update graph node attributes
            self.graph.nodes[key]['fitness_scores'] = self.nodes[key].fitness_scores
            self.graph.nodes[key]['average_fitness'] = self.nodes[key].get_average_fitness()
    
    def deactivate_node(self, key: str) -> None:
        """Mark a node as inactive (pruned)."""
        if key in self.nodes:
            self.nodes[key].is_active = False
            self.graph.nodes[key]['is_active'] = False
    
    def get_ancestors(self, key: str, max_depth: Optional[int] = None) -> Set[str]:
        """Get all ancestors of a node."""
        ancestors = set()
        to_visit = [(key, 0)]
        
        while to_visit:
            current_key, depth = to_visit.pop()
            
            if max_depth is not None and depth >= max_depth:
                continue
            
            if current_key in self.nodes:
                for parent_key in self.nodes[current_key].parent_keys:
                    if parent_key not in ancestors:
                        ancestors.add(parent_key)
                        to_visit.append((parent_key, depth + 1))
        
        return ancestors
    
    def get_descendants(self, key: str, max_depth: Optional[int] = None) -> Set[str]:
        """Get all descendants of a node."""
        descendants = set()
        to_visit = [(key, 0)]
        
        while to_visit:
            current_key, depth = to_visit.pop()
            
            if max_depth is not None and depth >= max_depth:
                continue
            
            # Find children
            children = [
                edge.child_key for edge in self.edges
                if edge.parent_key == current_key
            ]
            
            for child_key in children:
                if child_key not in descendants:
                    descendants.add(child_key)
                    to_visit.append((child_key, depth + 1))
        
        return descendants
    
    def get_generation(self, key: str) -> int:
        """Get generation number of a node."""
        return self._generation_counter.get(key, 0)
    
    def get_lineage_path(self, key: str) -> List[str]:
        """Get the primary lineage path to root for a node."""
        path = [key]
        current = key
        
        while current in self.nodes and self.nodes[current].parent_keys:
            # Choose the parent with highest fitness as primary lineage
            parents = self.nodes[current].parent_keys
            if len(parents) == 1:
                current = parents[0]
            else:
                # Select parent with highest average fitness
                best_parent = max(
                    parents,
                    key=lambda p: self.nodes[p].get_average_fitness() if p in self.nodes else 0
                )
                current = best_parent
            
            path.append(current)
        
        return path
    
    def analyze_fitness_evolution(self, node_type: Optional[str] = None) -> Dict[str, Any]:
        """Analyze fitness evolution over generations."""
        
        nodes_to_analyze = [
            node for node in self.nodes.values()
            if node_type is None or node.node_type == node_type
        ]
        
        # Group by generation
        generation_fitness: Dict[int, List[float]] = {}
        for node in nodes_to_analyze:
            gen = self.get_generation(node.key)
            if gen not in generation_fitness:
                generation_fitness[gen] = []
            
            avg_fitness = node.get_average_fitness()
            if avg_fitness > 0:  # Only include nodes with fitness scores
                generation_fitness[gen].append(avg_fitness)
        
        # Calculate statistics per generation
        generation_stats = {}
        for gen, fitnesses in generation_fitness.items():
            if fitnesses:
                generation_stats[gen] = {
                    "count": len(fitnesses),
                    "mean_fitness": sum(fitnesses) / len(fitnesses),
                    "max_fitness": max(fitnesses),
                    "min_fitness": min(fitnesses)
                }
        
        return {
            "total_generations": max(generation_fitness.keys()) + 1 if generation_fitness else 0,
            "generation_stats": generation_stats,
            "total_nodes_analyzed": len(nodes_to_analyze)
        }
    
    def find_evolutionary_branches(self) -> List[Dict[str, Any]]:
        """Find major evolutionary branches in the genealogy."""
        
        branches = []
        
        # Find nodes with multiple children (branch points)
        for node_key, node in self.nodes.items():
            children = [
                edge.child_key for edge in self.edges
                if edge.parent_key == node_key
            ]
            
            if len(children) > 1:
                branch_info = {
                    "branch_point": node_key,
                    "generation": self.get_generation(node_key),
                    "children": children,
                    "child_fitness": {
                        child: self.nodes[child].get_average_fitness()
                        for child in children if child in self.nodes
                    }
                }
                branches.append(branch_info)
        
        # Sort by generation
        branches.sort(key=lambda x: x["generation"])
        return branches
    
    def export_to_graphml(self, filepath: str) -> None:
        """Export genealogy to GraphML format for visualization."""
        
        # Create a clean graph for export
        export_graph = nx.DiGraph()
        
        # Add nodes with all attributes
        for key, node in self.nodes.items():
            node_attrs = node.to_dict()
            # Convert datetime and other non-serializable types
            for attr_key, attr_value in node_attrs.items():
                if isinstance(attr_value, (list, dict)):
                    node_attrs[attr_key] = json.dumps(attr_value)
                elif not isinstance(attr_value, (str, int, float, bool)):
                    node_attrs[attr_key] = str(attr_value)
            
            export_graph.add_node(key, **node_attrs)
        
        # Add edges with attributes
        for edge in self.edges:
            edge_attrs = edge.to_dict()
            # Convert non-serializable types
            for attr_key, attr_value in edge_attrs.items():
                if isinstance(attr_value, (list, dict)):
                    edge_attrs[attr_key] = json.dumps(attr_value)
                elif not isinstance(attr_value, (str, int, float, bool)):
                    edge_attrs[attr_key] = str(attr_value)
            
            export_graph.add_edge(
                edge.parent_key,
                edge.child_key,
                **edge_attrs
            )
        
        # Write to GraphML
        nx.write_graphml(export_graph, filepath)
    
    def export_to_json(self, filepath: str) -> None:
        """Export genealogy to JSON format."""
        
        export_data = {
            "nodes": {key: node.to_dict() for key, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "generation_counter": self._generation_counter,
            "export_timestamp": datetime.utcnow().isoformat(),
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges)
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def import_from_json(self, filepath: str) -> None:
        """Import genealogy from JSON format."""
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Clear existing data
        self.nodes.clear()
        self.edges.clear()
        self.graph.clear()
        
        # Import nodes
        for key, node_data in data["nodes"].items():
            node_data["birth_time"] = datetime.fromisoformat(node_data["birth_time"])
            node = LineageNode(**node_data)
            self.nodes[key] = node
            self.graph.add_node(key, **node.to_dict())
        
        # Import edges
        for edge_data in data["edges"]:
            edge_data["created_at"] = datetime.fromisoformat(edge_data["created_at"])
            edge = LineageEdge(**edge_data)
            self.edges.append(edge)
            self.graph.add_edge(
                edge.parent_key,
                edge.child_key,
                **edge.to_dict()
            )
        
        # Import generation counter
        self._generation_counter = data.get("generation_counter", {})
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get genealogy statistics."""
        
        active_nodes = sum(1 for node in self.nodes.values() if node.is_active)
        node_types = {}
        fitness_stats = []
        
        for node in self.nodes.values():
            # Count node types
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
            
            # Collect fitness scores
            if node.fitness_scores:
                fitness_stats.extend(node.fitness_scores)
        
        return {
            "total_nodes": len(self.nodes),
            "active_nodes": active_nodes,
            "inactive_nodes": len(self.nodes) - active_nodes,
            "total_edges": len(self.edges),
            "node_types": node_types,
            "max_generation": max(self._generation_counter.values()) if self._generation_counter else 0,
            "fitness_stats": {
                "total_scores": len(fitness_stats),
                "mean_fitness": sum(fitness_stats) / len(fitness_stats) if fitness_stats else 0,
                "max_fitness": max(fitness_stats) if fitness_stats else 0,
                "min_fitness": min(fitness_stats) if fitness_stats else 0
            }
        }
    
    def prune_lineage(self, fitness_threshold: float = 0.1, age_threshold_hours: float = 168) -> List[str]:
        """
        Prune lineage based on fitness and age thresholds.
        
        Args:
            fitness_threshold: Minimum fitness to keep
            age_threshold_hours: Maximum age in hours to keep low-fitness nodes
            
        Returns:
            List of pruned node keys
        """
        
        pruned_keys = []
        current_time = datetime.utcnow()
        
        for key, node in list(self.nodes.items()):
            should_prune = False
            
            # Check fitness
            avg_fitness = node.get_average_fitness()
            if avg_fitness < fitness_threshold:
                # Check age
                age_hours = (current_time - node.birth_time).total_seconds() / 3600
                if age_hours > age_threshold_hours:
                    should_prune = True
            
            if should_prune:
                self.deactivate_node(key)
                pruned_keys.append(key)
        
        return pruned_keys


# Global genealogy tracer instance
_global_tracer: Optional[GenealogyTracer] = None


def get_global_tracer() -> GenealogyTracer:
    """Get the global genealogy tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = GenealogyTracer()
    return _global_tracer


def trace_birth(
    key: str,
    node_type: str,
    birth_event: Optional[str] = None,
    parent_keys: List[str] = None,
    metadata: Dict[str, Any] = None
) -> LineageNode:
    """Convenience function to trace birth of a new entity."""
    
    tracer = get_global_tracer()
    return tracer.add_node(
        key=key,
        node_type=node_type,
        birth_event=birth_event,
        parent_keys=parent_keys,
        metadata=metadata
    )


def trace_fitness(key: str, fitness_score: float) -> None:
    """Convenience function to trace fitness score."""
    
    tracer = get_global_tracer()
    tracer.update_fitness(key, fitness_score)
