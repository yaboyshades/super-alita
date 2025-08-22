from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx
import numpy as np
from sklearn.mixture import GaussianMixture

from cortex.config.flags import LEANRAG

from .embeddings import EmbeddingProvider

AGG_TYPE = "aggregate"
BASE_LEVEL = 0


def _ensure_vec(g: nx.Graph, node: str, embedder: EmbeddingProvider) -> np.ndarray:
    if "vec" in g.nodes[node]:
        return g.nodes[node]["vec"]
    # Build text from available fields
    name = g.nodes[node].get("name") or node
    desc = g.nodes[node].get("description", "")
    text = f"{name} {desc}".strip()
    vec = embedder.embed_text(text)
    g.nodes[node]["vec"] = vec
    return vec


def _aggregate_vecs(vecs: Iterable[np.ndarray]) -> np.ndarray:
    V = list(vecs)
    if not V:
        return np.zeros(256, dtype=np.float32)
    m = np.mean(np.stack(V, axis=0), axis=0)
    n = np.linalg.norm(m) + 1e-8
    return m / n


def _cluster_level(
    g: nx.Graph,
    nodes: list[str],
    embedder: EmbeddingProvider,
    level: int,
    k_init: int,
    min_cluster: int,
) -> list[tuple[str, list[str]]]:
    """Cluster given nodes with GMM into <= k clusters, return list of (parent, children)."""
    if len(nodes) < max(2 * min_cluster, 4):
        return []
    X = np.stack([_ensure_vec(g, n, embedder) for n in nodes], axis=0)
    # decide k based on size
    max_k = max(2, min(k_init, len(nodes) // min_cluster))
    if max_k < 2:
        return []
    gmm = GaussianMixture(n_components=max_k, covariance_type="diag", random_state=42)
    labels = gmm.fit_predict(X)
    clusters: dict[int, list[str]] = {}
    for n, lab in zip(nodes, labels, strict=False):
        clusters.setdefault(int(lab), []).append(n)
    # filter too-small clusters by merging to nearest centroid
    centroids = gmm.means_
    for lab, members in list(clusters.items()):
        if len(members) < min_cluster:
            # merge to closest other centroid
            dists = np.linalg.norm(centroids - centroids[lab], axis=1)
            dists[lab] = 1e9
            tgt = int(np.argmin(dists))
            clusters.setdefault(tgt, []).extend(members)
            del clusters[lab]
    # build parents
    out: list[tuple[str, list[str]]] = []
    for idx, members in clusters.items():
        if len(members) < min_cluster:
            continue
        pid = f"agg:L{level}:{idx}"
        # synthesize a label from top tokens (simple; replace with LLM if desired)
        vecs = [_ensure_vec(g, n, embedder) for n in members]
        pvec = _aggregate_vecs(vecs)
        g.add_node(
            pid,
            type=AGG_TYPE,
            level=level,
            children=list(members),
            name=f"Cluster L{level}:{idx} ({len(members)} items)",
            description=f"Aggregated entity from {len(members)} children",
            vec=pvec,
        )
        # parent->child edges
        for c in members:
            g.add_edge(pid, c, kind="aggregates", level=level)
        out.append((pid, members))
        # attach parent pointer list on children
        for c in members:
            parents = list(g.nodes[c].get("parents", []))
            parents.append(pid)
            g.nodes[c]["parents"] = parents
    return out


def _cross_cluster_link_strength(g: nx.Graph, A: list[str], B: list[str]) -> float:
    """Estimate cross-links strength between two child-sets."""
    if not A or not B:
        return 0.0
    total = 0
    links = 0
    Aset = set(A)
    Bset = set(B)
    for u in A:
        for v in g.neighbors(u):
            if v in Bset:
                links += 1
            total += 1
    for u in B:
        for v in g.neighbors(u):
            if v in Aset:
                links += 1
            total += 1
    if total == 0:
        return 0.0
    return links / total


def _build_aggregate_links(
    g: nx.Graph, parents: list[tuple[str, list[str]]], level: int, link_threshold: float
) -> None:
    """Create inter-parent relations based on cross-cluster connectivity."""
    for i in range(len(parents)):
        p_i, ch_i = parents[i]
        for j in range(i + 1, len(parents)):
            p_j, ch_j = parents[j]
            strength = _cross_cluster_link_strength(g, ch_i, ch_j)
            if strength >= link_threshold:
                g.add_edge(
                    p_i, p_j, kind="relates", level=level, weight=float(strength)
                )
                g.add_edge(
                    p_j, p_i, kind="relates", level=level, weight=float(strength)
                )


@dataclass
class HierarchicalAggregator:
    embedder: EmbeddingProvider
    max_depth: int = LEANRAG.max_depth
    min_cluster_size: int = LEANRAG.min_cluster_size
    gmm_k_init: int = LEANRAG.gmm_k_init
    link_threshold: float = LEANRAG.link_threshold

    def build(
        self, g: nx.Graph, base_nodes: list[str] | None = None
    ) -> list[list[str]]:
        """
        Build hierarchy in-place on graph g.
        Returns list of levels with their parent node IDs (level 1..max_depth).
        """
        # Initialize base nodes and ensure embeddings
        if base_nodes is None:
            base_nodes = [n for n, d in g.nodes(data=True) if d.get("type") != AGG_TYPE]
        for n in base_nodes:
            _ensure_vec(g, n, self.embedder)
            g.nodes[n]["level"] = BASE_LEVEL
            g.nodes[n].setdefault("parents", [])

        current = base_nodes
        levels: list[list[str]] = []
        for L in range(1, self.max_depth + 1):
            parents = _cluster_level(
                g,
                current,
                self.embedder,
                level=L,
                k_init=self.gmm_k_init,
                min_cluster=self.min_cluster_size,
            )
            if not parents:
                break
            _build_aggregate_links(
                g, parents, level=L, link_threshold=self.link_threshold
            )
            parent_ids = [pid for pid, _ in parents]
            levels.append(parent_ids)
            current = parent_ids
        return levels


def build_hierarchy(
    g: nx.Graph,
    embedder: EmbeddingProvider,
    max_depth: int | None = None,
    min_cluster_size: int | None = None,
    gmm_k_init: int | None = None,
    link_threshold: float | None = None,
) -> list[list[str]]:
    agg = HierarchicalAggregator(
        embedder=embedder,
        max_depth=max_depth if max_depth is not None else LEANRAG.max_depth,
        min_cluster_size=min_cluster_size
        if min_cluster_size is not None
        else LEANRAG.min_cluster_size,
        gmm_k_init=gmm_k_init if gmm_k_init is not None else LEANRAG.gmm_k_init,
        link_threshold=link_threshold
        if link_threshold is not None
        else LEANRAG.link_threshold,
    )
    return agg.build(g)
