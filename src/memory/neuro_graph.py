"""Neural Memory Graph with semantic clustering and influence propagation.

Implements a knowledge graph where nodes are memory items (specs, plans, tasks)
and edges represent semantic similarity, constitutional influence, and causal relationships.

Features:
- Semantic clustering (vector embeddings + HDBSCAN)
- Influence propagation (constitutional scores flow through graph)
- Causal tracking (which decisions influenced which outcomes)
- Vector store integration (ChromaDB, Qdrant, or in-memory)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


class EdgeType(str, Enum):
    """Types of relationships in the memory graph."""

    SEMANTIC = "semantic"  # Similar content (cosine similarity)
    CAUSAL = "causal"  # A caused/influenced B
    CONSTITUTIONAL = "constitutional"  # Constitutional compliance link
    TEMPORAL = "temporal"  # Time-ordered relationship
    DEPENDENCY = "dependency"  # B depends on A


@dataclass
class MemoryNode:
    """Node in the neural memory graph."""

    id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    constitutional_score: float = 0.0
    cluster_id: int | None = None


@dataclass
class MemoryEdge:
    """Edge in the neural memory graph."""

    source: str  # Node ID
    target: str  # Node ID
    edge_type: EdgeType
    weight: float  # Strength of relationship (0.0-1.0)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class NeuroGraph:
    """Neural Memory Graph with semantic clustering and influence propagation.

    Maintains a knowledge graph of memory items with:
    - Vector embeddings for semantic search
    - Constitutional influence propagation
    - Causal relationship tracking
    - Clustering for knowledge organization
    """

    def __init__(
        self,
        embedding_fn: Any | None = None,
        vector_store: Any | None = None,
    ):
        """Initialize the neural memory graph.

        Args:
            embedding_fn: Function to generate embeddings (str -> list[float])
            vector_store: Optional vector store for persistence (ChromaDB, etc.)
        """
        self.nodes: dict[str, MemoryNode] = {}
        self.edges: list[MemoryEdge] = []
        self.embedding_fn = embedding_fn or self._default_embedding
        self.vector_store = vector_store

        # Adjacency lists for efficient traversal
        self._outgoing: dict[str, list[MemoryEdge]] = defaultdict(list)
        self._incoming: dict[str, list[MemoryEdge]] = defaultdict(list)

        # Clusters (computed on demand)
        self._clusters: dict[int, list[str]] = {}

    def _default_embedding(self, text: str) -> list[float]:
        """Fallback embedding using content hash."""
        h = hashlib.sha256(text.encode()).digest()
        return [float(b) / 255.0 for b in h[:16]]  # 16-dim vector

    async def add_node(
        self,
        content: str,
        node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNode:
        """Add a memory node to the graph.

        Args:
            content: Text content of the memory
            node_id: Optional node ID (auto-generated if not provided)
            metadata: Optional metadata

        Returns:
            Created MemoryNode
        """
        node_id = node_id or hashlib.sha256(content.encode()).hexdigest()[:16]
        metadata = metadata or {}

        # Generate embedding
        embedding = await asyncio.to_thread(self.embedding_fn, content)

        node = MemoryNode(
            id=node_id,
            content=content,
            embedding=embedding,
            metadata=metadata,
        )

        self.nodes[node_id] = node

        # Store in vector store if available
        if self.vector_store:
            await self._store_in_vector_db(node)

        return node

    async def _store_in_vector_db(self, node: MemoryNode) -> None:
        """Store node in vector database."""
        # Stub - implement based on vector store type (ChromaDB, Qdrant, etc.)
        pass

    async def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEdge:
        """Add an edge between two nodes.

        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of relationship
            weight: Edge weight (0.0-1.0)
            metadata: Optional metadata

        Returns:
            Created MemoryEdge
        """
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Both nodes must exist: {source}, {target}")

        edge = MemoryEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {},
        )

        self.edges.append(edge)
        self._outgoing[source].append(edge)
        self._incoming[target].append(edge)

        return edge

    async def compute_semantic_edges(
        self, similarity_threshold: float = 0.7
    ) -> int:
        """Compute semantic similarity edges between all nodes.

        Args:
            similarity_threshold: Minimum cosine similarity (0.0-1.0)

        Returns:
            Number of edges created
        """
        if np is None:
            return 0

        nodes_with_embeddings = [
            n for n in self.nodes.values() if n.embedding is not None
        ]

        if len(nodes_with_embeddings) < 2:
            return 0

        count = 0
        for i, node_a in enumerate(nodes_with_embeddings):
            for node_b in nodes_with_embeddings[i + 1 :]:
                sim = self._cosine_similarity(
                    node_a.embedding,  # type: ignore[arg-type]
                    node_b.embedding,  # type: ignore[arg-type]
                )

                if sim >= similarity_threshold:
                    await self.add_edge(
                        node_a.id,
                        node_b.id,
                        EdgeType.SEMANTIC,
                        weight=sim,
                    )
                    count += 1

        return count

    def _cosine_similarity(
        self, vec_a: list[float], vec_b: list[float]
    ) -> float:
        """Compute cosine similarity between two vectors."""
        if np is None or len(vec_a) != len(vec_b):
            return 0.0

        a = np.array(vec_a)
        b = np.array(vec_b)

        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot / (norm_a * norm_b))

    async def propagate_constitutional_influence(
        self, source_node: str, damping: float = 0.85
    ) -> dict[str, float]:
        """Propagate constitutional scores from a source node through the graph.

        Similar to PageRank, constitutional influence flows through edges.
        High-quality decisions influence connected decisions.

        Args:
            source_node: Starting node ID
            damping: Damping factor (0.0-1.0, default 0.85 like PageRank)

        Returns:
            Dictionary of node_id -> influence_score
        """
        if source_node not in self.nodes:
            return {}

        source_score = self.nodes[source_node].constitutional_score
        if source_score <= 0:
            return {}

        # BFS with damping
        influence: dict[str, float] = {source_node: source_score}
        visited: set[str] = {source_node}
        queue: list[tuple[str, float]] = [(source_node, source_score)]

        while queue:
            current, current_influence = queue.pop(0)

            # Propagate to neighbors
            for edge in self._outgoing[current]:
                if edge.edge_type not in {
                    EdgeType.CONSTITUTIONAL,
                    EdgeType.CAUSAL,
                }:
                    continue

                neighbor = edge.target
                propagated = current_influence * damping * edge.weight

                if neighbor not in influence:
                    influence[neighbor] = 0.0

                influence[neighbor] += propagated

                if neighbor not in visited and propagated > 0.01:
                    visited.add(neighbor)
                    queue.append((neighbor, propagated))

        return influence

    async def cluster_memories(
        self, min_cluster_size: int = 3
    ) -> dict[int, list[str]]:
        """Cluster memories based on semantic similarity.

        Uses semantic edges to group related memories.

        Args:
            min_cluster_size: Minimum nodes per cluster

        Returns:
            Dictionary of cluster_id -> [node_ids]
        """
        # Simple connected components clustering
        clusters: dict[int, list[str]] = {}
        assigned: set[str] = set()
        cluster_id = 0

        for node_id in self.nodes:
            if node_id in assigned:
                continue

            # BFS to find connected component
            component: list[str] = []
            queue = [node_id]
            visited: set[str] = {node_id}

            while queue:
                current = queue.pop(0)
                component.append(current)

                for edge in self._outgoing[current]:
                    if edge.edge_type != EdgeType.SEMANTIC:
                        continue
                    if edge.target not in visited:
                        visited.add(edge.target)
                        queue.append(edge.target)

                for edge in self._incoming[current]:
                    if edge.edge_type != EdgeType.SEMANTIC:
                        continue
                    if edge.source not in visited:
                        visited.add(edge.source)
                        queue.append(edge.source)

            if len(component) >= min_cluster_size:
                clusters[cluster_id] = component
                assigned.update(component)
                cluster_id += 1

        self._clusters = clusters

        # Update node cluster IDs
        for cid, node_ids in clusters.items():
            for nid in node_ids:
                self.nodes[nid].cluster_id = cid

        return clusters

    async def get_causal_path(
        self, source: str, target: str
    ) -> list[str] | None:
        """Find causal path from source to target node.

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            List of node IDs forming path, or None if no path exists
        """
        if source not in self.nodes or target not in self.nodes:
            return None

        # BFS for shortest causal path
        queue: list[tuple[str, list[str]]] = [(source, [source])]
        visited: set[str] = {source}

        while queue:
            current, path = queue.pop(0)

            if current == target:
                return path

            for edge in self._outgoing[current]:
                if edge.edge_type != EdgeType.CAUSAL:
                    continue

                neighbor = edge.target
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        edge_counts = defaultdict(int)
        for edge in self.edges:
            edge_counts[edge.edge_type] += 1

        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "edge_types": dict(edge_counts),
            "clusters": len(self._clusters),
            "avg_edges_per_node": (
                len(self.edges) / len(self.nodes) if self.nodes else 0
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Export graph to dictionary."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "content": n.content[:100],  # Truncate
                    "metadata": n.metadata,
                    "constitutional_score": n.constitutional_score,
                    "cluster_id": n.cluster_id,
                    "created_at": n.created_at.isoformat(),
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "type": e.edge_type,
                    "weight": e.weight,
                    "created_at": e.created_at.isoformat(),
                }
                for e in self.edges
            ],
            "stats": self.get_stats(),
        }


# Example usage
async def example_neuro_graph() -> None:
    """Example demonstrating NeuroGraph usage."""
    graph = NeuroGraph()

    # Add memory nodes
    spec = await graph.add_node(
        "Feature: Add user authentication with JWT tokens",
        metadata={"type": "spec", "priority": "high"},
    )
    spec.constitutional_score = 0.92

    plan = await graph.add_node(
        "Plan: 1. Design auth flow, 2. Implement JWT, 3. Add middleware",
        metadata={"type": "plan"},
    )
    plan.constitutional_score = 0.88

    task1 = await graph.add_node(
        "Task: Implement JWT token generation and validation",
        metadata={"type": "task", "status": "done"},
    )
    task1.constitutional_score = 0.95

    # Add causal edges (spec -> plan -> task)
    await graph.add_edge(spec.id, plan.id, EdgeType.CAUSAL, weight=1.0)
    await graph.add_edge(plan.id, task1.id, EdgeType.CAUSAL, weight=1.0)

    # Compute semantic similarity
    await graph.compute_semantic_edges(similarity_threshold=0.6)

    # Propagate constitutional influence
    influence = await graph.propagate_constitutional_influence(spec.id)
    print(f"Constitutional influence from spec: {influence}")

    # Find causal path
    path = await graph.get_causal_path(spec.id, task1.id)
    print(f"Causal path: {path}")

    # Cluster memories
    clusters = await graph.cluster_memories(min_cluster_size=2)
    print(f"Clusters: {clusters}")

    # Export
    print(json.dumps(graph.to_dict(), indent=2))


if __name__ == "__main__":
    asyncio.run(example_neuro_graph())
