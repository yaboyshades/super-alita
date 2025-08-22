# cortex/kg/leanrag.py
import heapq

import networkx as nx
import numpy as np
from sklearn.mixture import GaussianMixture

from cortex.config.flags import LEANRAG as LR_FLAGS
from cortex.config.flags import LeanRAGFlags


class LeanRAG:
    """LeanRAG 3.0 implementation with hierarchical KG aggregation and LCA retrieval"""

    AGG_TYPE = "aggregate"
    EDGE_CONTAINS = "aggregates"  # align with Cortex nomenclature
    EDGE_RELATES = "relates"

    def __init__(self, embedder, flags: LeanRAGFlags | None = None):
        self.embedder = embedder
        # LR_FLAGS is a singleton dataclass instance; do NOT call it.
        self.flags: LeanRAGFlags = flags or LR_FLAGS
        self._update_count = 0
        self._snapshots: list[dict[str, np.ndarray]] = []
        # simple heuristics for rehearsal cadence
        self.snapshot_interval = 5
        self.rehearsal_samples = 3

    # --- helpers -------------------------------------------------------------
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v) + 1e-8
        return (v / n).astype(np.float32)

    def _node_to_text(self, node_data: dict) -> str:
        parts = []
        if "name" in node_data:
            parts.append(node_data["name"])
        if "description" in node_data:
            parts.append(node_data["description"])
        if "content" in node_data:
            parts.append(node_data["content"])
        return " ".join(parts)

    # --- rehearsal & snapshotting -------------------------------------------
    def _snapshot_embeddings(self, graph: nx.Graph) -> None:
        """Store a snapshot of current node embeddings."""
        snap = {
            n: np.copy(graph.nodes[n]["vec"])
            for n in graph.nodes
            if "vec" in graph.nodes[n]
        }
        self._snapshots.append(snap)
        if len(self._snapshots) > self.rehearsal_samples:
            self._snapshots.pop(0)

    def _rehearse_from_snapshots(self, graph: nx.Graph) -> None:
        """Reinforce current graph with stored snapshots."""
        for snap in self._snapshots:
            for node, vec in snap.items():
                if node in graph.nodes and "vec" in graph.nodes[node]:
                    graph.nodes[node]["vec"] = self._normalize(
                        (graph.nodes[node]["vec"] + vec) / 2
                    )

    def build_hierarchy(self, graph: nx.Graph) -> nx.Graph:
        """Build hierarchical knowledge graph using GMM clustering"""
        self._update_count += 1
        # reinforce previous snapshots before building
        self._rehearse_from_snapshots(graph)

        # Ensure all base nodes have vectors and level=0 with parents=[]
        for node in graph.nodes:
            nd = graph.nodes[node]
            if "vec" not in nd:
                text = self._node_to_text(graph.nodes[node])
                vec = self.embedder.embed_text(text)
                graph.nodes[node]["vec"] = self._normalize(
                    np.asarray(vec, dtype=np.float32)
                )
            if "type" not in nd:
                nd["type"] = "entity"
            nd.setdefault("level", 0)
            nd.setdefault("parents", [])

        # Recursive aggregation
        current_level = 1  # aggregate levels start at 1
        current_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("type") != self.AGG_TYPE
        ]

        while (
            current_level < self.flags.max_depth
            and len(current_nodes) > self.flags.min_cluster_size
        ):
            # Cluster nodes using GMM
            embeddings = np.array(
                [graph.nodes[n]["vec"] for n in current_nodes], dtype=np.float32
            )
            n_clusters = min(
                self.flags.gmm_k_init, len(current_nodes) // self.flags.min_cluster_size
            )

            if n_clusters < 2:
                break
            # safer GMM settings for medium-dim data
            gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type="diag",
                reg_covar=1e-6,
                random_state=42,
            )
            clusters = gmm.fit_predict(embeddings)

            # Create aggregate nodes for each cluster
            new_nodes = []
            for cluster_id in range(n_clusters):
                cluster_nodes = [
                    current_nodes[i]
                    for i in range(len(clusters))
                    if clusters[i] == cluster_id
                ]

                if len(cluster_nodes) < self.flags.min_cluster_size:
                    continue

                # Create aggregate node
                aggregate_id = f"agg:L{current_level}:{cluster_id}"
                aggregate_vec = self._normalize(
                    np.mean([graph.nodes[n]["vec"] for n in cluster_nodes], axis=0)
                )

                graph.add_node(
                    aggregate_id,
                    type=self.AGG_TYPE,
                    level=current_level,
                    vec=aggregate_vec,
                    children=cluster_nodes,
                    name=f"Cluster L{current_level}:{cluster_id} ({len(cluster_nodes)} items)",
                    description=f"Aggregated entity from {len(cluster_nodes)} children",
                )

                # Add edges from aggregate to children
                for child in cluster_nodes:
                    graph.add_edge(
                        aggregate_id,
                        child,
                        kind=self.EDGE_CONTAINS,
                        level=current_level,
                    )
                    # maintain parent chain meta for fast LCA
                    parents = list(graph.nodes[child].get("parents", []))
                    parents.append(aggregate_id)
                    graph.nodes[child]["parents"] = parents

                new_nodes.append(aggregate_id)

            # Create cross-cluster relations
            self._create_cross_cluster_relations(graph, new_nodes, current_level)

            current_nodes = new_nodes
            current_level += 1

        # Periodically snapshot embeddings to reinforce older knowledge
        if self._update_count % self.snapshot_interval == 0:
            self._snapshot_embeddings(graph)

        return graph

    def _create_cross_cluster_relations(
        self, graph: nx.Graph, aggregate_nodes: list[str], level: int
    ):
        """Create relations between aggregate nodes based on underlying connections"""
        for i, agg1 in enumerate(aggregate_nodes):
            for j, agg2 in enumerate(aggregate_nodes[i + 1 :], i + 1):
                # Calculate connection strength between clusters
                strength = self._calculate_cluster_connection_strength(
                    graph, agg1, agg2
                )

                if strength >= self.flags.link_threshold:
                    graph.add_edge(
                        agg1,
                        agg2,
                        kind=self.EDGE_RELATES,
                        weight=float(strength),
                        level=level,
                    )
                    graph.add_edge(
                        agg2,
                        agg1,
                        kind=self.EDGE_RELATES,
                        weight=float(strength),
                        level=level,
                    )

    def _calculate_cluster_connection_strength(
        self, graph: nx.Graph, agg1: str, agg2: str
    ) -> float:
        """Calculate connection strength between two aggregate nodes"""
        children1 = graph.nodes[agg1].get("children", [])
        children2 = graph.nodes[agg2].get("children", [])

        if not children1 or not children2:
            return 0.0

        # Count connections between children (bidirectional presence)
        connection_count = 0
        for child1 in children1:
            for child2 in children2:
                if graph.has_edge(child1, child2) or graph.has_edge(child2, child1):
                    connection_count += 1

        # Normalize by geometric mean of cluster sizes for scale-invariance
        total_possible = (len(children1) * len(children2)) ** 0.5
        return connection_count / total_possible if total_possible > 0 else 0.0

    def retrieve(self, graph: nx.Graph, query: str) -> dict:
        """Retrieve context using LCA-based approach"""
        # Anchor seeds - find most relevant base nodes
        query_embedding = self._normalize(
            np.asarray(self.embedder.embed_text(query), dtype=np.float32)
        )
        seeds = self._find_seeds(graph, query_embedding)

        if not seeds:
            return {"brief": "No relevant context found", "subgraph": nx.DiGraph()}

        # Find LCA of seeds
        lca = self._find_lca(graph, seeds)

        # Build minimal subgraph
        subgraph = self._build_minimal_subgraph(graph, seeds, lca)

        # Generate brief
        brief = self._generate_brief(subgraph)

        return {"seeds": seeds, "lca": lca, "subgraph": subgraph, "brief": brief}

    def _find_seeds(
        self, graph: nx.Graph, query_embedding: np.ndarray
    ) -> list[tuple[str, float]]:
        """Find most relevant base nodes using cosine similarity"""
        base_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("type") != self.AGG_TYPE
        ]

        scored = []
        for node in base_nodes:
            if "vec" not in graph.nodes[node]:
                continue

            similarity = self._cosine_similarity(
                query_embedding, graph.nodes[node]["vec"]
            )
            heapq.heappush(scored, (similarity, node))

            if len(scored) > self.flags.seeds_k:
                heapq.heappop(scored)

        return [(node, score) for score, node in sorted(scored, reverse=True)]

    def _find_lca(self, graph: nx.Graph, seeds: list[tuple[str, float]]) -> str | None:
        """Find lowest common ancestor of seeds in the hierarchy"""
        if not seeds:
            return None

        seed_nodes = [seed[0] for seed in seeds]

        # Get all ancestors for each seed
        all_ancestors = []
        for seed in seed_nodes:
            ancestors: set[str] = set()
            current = seed
            # Prefer explicit parent chains if present (fast path)
            parents_meta = graph.nodes[current].get("parents", [])
            if parents_meta:
                for p in parents_meta[::-1]:
                    ancestors.add(p)
                    current = p
            else:
                # fallback: traverse predecessors that are aggregates
                while True:
                    parents = [
                        n
                        for n in graph.predecessors(current)
                        if graph.nodes[n].get("type") == self.AGG_TYPE
                    ]
                    if not parents:
                        break
                    current = parents[0]
                    ancestors.add(current)
            all_ancestors.append(ancestors)

        # Find common ancestors
        common_ancestors = set.intersection(*all_ancestors) if all_ancestors else set()

        if not common_ancestors:
            return None

        # Find the lowest (deepest) common ancestor (max level)
        lca = None
        max_level = -1
        for ancestor in common_ancestors:
            level = graph.nodes[ancestor].get("level", -1)
            if level > max_level:
                max_level = level
                lca = ancestor

        return lca

    def _build_minimal_subgraph(
        self, graph: nx.Graph, seeds: list[tuple[str, float]], lca: str | None
    ) -> nx.DiGraph:
        """Build minimal subgraph containing seeds and paths to LCA"""
        subgraph = nx.DiGraph()
        seed_nodes = [seed[0] for seed in seeds]

        # Add seeds and their paths to LCA
        for seed in seed_nodes:
            path = self._get_path_to_ancestor(graph, seed, lca)
            for i in range(len(path)):
                node = path[i]
                subgraph.add_node(node, **graph.nodes[node])
                if i > 0:
                    prev_node = path[i - 1]
                    # Copy edge attributes if available
                    if graph.has_edge(node, prev_node):
                        # direction in our hierarchy is parent -> child; ensure consistency
                        edge_data = graph.get_edge_data(node, prev_node) or {}
                        subgraph.add_edge(node, prev_node, **edge_data)
                    else:
                        subgraph.add_edge(
                            node,
                            prev_node,
                            kind=self.EDGE_CONTAINS,
                            level=graph.nodes[node].get("level"),
                        )

        # Add cross-relations between aggregate nodes
        aggregate_nodes = [
            n for n in subgraph.nodes if subgraph.nodes[n].get("type") == self.AGG_TYPE
        ]

        for i, agg1 in enumerate(aggregate_nodes):
            for agg2 in aggregate_nodes[i + 1 :]:
                if (
                    graph.has_edge(agg1, agg2)
                    and graph.edges[agg1, agg2].get("kind") == self.EDGE_RELATES
                ):
                    subgraph.add_edge(agg1, agg2, **graph.get_edge_data(agg1, agg2))
                if (
                    graph.has_edge(agg2, agg1)
                    and graph.edges[agg2, agg1].get("kind") == self.EDGE_RELATES
                ):
                    subgraph.add_edge(agg2, agg1, **graph.get_edge_data(agg2, agg1))

        return subgraph

    def _get_path_to_ancestor(
        self, graph: nx.Graph, node: str, ancestor: str | None
    ) -> list[str]:
        """Get path from node to ancestor (including both)"""
        path = [node]
        current = node
        if ancestor is None:
            # climb full chain if no ancestor given
            parents = graph.nodes[current].get("parents", [])
            return [current] + parents

        # Prefer parent chain metadata
        parents_chain = graph.nodes[current].get("parents", [])
        if parents_chain:
            for p in parents_chain:
                path.append(p)
                if p == ancestor:
                    break
            return path

        # Fallback to graph traversal
        while current != ancestor:
            parents = [
                n
                for n in graph.predecessors(current)
                if graph.nodes[n].get("type") == self.AGG_TYPE
            ]
            if not parents:
                break
            current = parents[0]  # Take first parent
            path.append(current)

        return path

    def _generate_brief(self, subgraph: nx.DiGraph) -> str:
        """Generate textual brief from subgraph"""
        lines = []

        # Add leaf nodes (seeds) first
        leaves = [n for n in subgraph.nodes if subgraph.out_degree(n) == 0]
        for leaf in leaves[: max(1, self.flags.brief_max_nodes // 2)]:
            node_data = subgraph.nodes[leaf]
            line = f"- {self._format_node(leaf, node_data)}"
            lines.append(line)

        # Add aggregate nodes
        aggregates = [
            n for n in subgraph.nodes if subgraph.nodes[n].get("type") == self.AGG_TYPE
        ]
        for agg in aggregates[: max(1, self.flags.brief_max_nodes // 2)]:
            node_data = subgraph.nodes[agg]
            line = f"- Cluster: {self._format_node(agg, node_data)}"
            if "children" in node_data:
                line += f" ({len(node_data['children'])} items)"
            lines.append(line)

        return "\n".join(lines)

    def _format_node(self, node_id: str, node_data: dict) -> str:
        """Format a node for display in brief"""
        name = node_data.get("name", node_id)
        desc = node_data.get("description", "")
        return f"{name}: {desc}" if desc else name

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b + 1e-8)
