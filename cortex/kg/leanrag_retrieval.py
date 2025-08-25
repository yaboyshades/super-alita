from __future__ import annotations

import heapq

import networkx as nx
import numpy as np

from cortex.config.flags import LEANRAG

from .embeddings import EmbeddingProvider


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)


def _base_nodes(g: nx.Graph) -> list[str]:
    return [n for n, d in g.nodes(data=True) if d.get("type") != "aggregate"]


def _ensure_vec(g: nx.Graph, n: str) -> np.ndarray | None:
    return g.nodes[n].get("vec")


def anchor_seeds(
    g: nx.Graph, embedder: EmbeddingProvider, query: str, k: int = LEANRAG.seeds_k
) -> list[tuple[str, float]]:
    """Find top-k base-layer nodes by cosine similarity to the query embedding."""
    qvec = embedder.embed_text(query)
    scored: list[tuple[float, str]] = []
    for n in _base_nodes(g):
        vec = _ensure_vec(g, n)
        if vec is None:
            # Skip nodes without vectors; hierarchy builder should have created them
            continue
        sim = _cosine(qvec, vec)
        heapq.heappush(scored, (sim, n))
        if len(scored) > k:
            heapq.heappop(scored)
    scored.sort(reverse=True)
    return [(n, s) for s, n in scored]


def _ancestors_chain(g: nx.Graph, n: str) -> list[str]:
    """Return chain from node up through aggregate parents (including self)."""
    chain = [n]
    cur = n
    # parents stored as list (ordered by levels ascending)
    while True:
        parents = g.nodes[cur].get("parents", [])
        if not parents:
            break
        # last parent is the highest added; but use the last as current step
        cur = parents[-1]
        chain.append(cur)
    return chain


def find_lca(g: nx.Graph, seeds: list[str]) -> str | None:
    """
    Find the (lowest) common ancestor across all seeds in the aggregation hierarchy.
    If multiple, pick the one with minimal max distance from seeds.
    """
    if not seeds:
        return None
    # Build ancestor sets with depth maps
    chains = [list(_ancestors_chain(g, s)) for s in seeds]
    sets = [set(c) for c in chains]
    common = set.intersection(*sets) if sets else set()
    if not common:
        # fallback: return None to indicate no shared ancestor; caller can handle
        return None
    # distance from seed = index in chain
    best = None
    best_score = 1e9
    for cand in common:
        # max depth among seeds
        maxd = max(chains[i].index(cand) for i in range(len(chains)))
        if maxd < best_score:
            best = cand
            best_score = maxd
    return best


def _path_to_ancestor(g: nx.Graph, n: str, anc: str) -> list[str]:
    """Return node chain from n up to anc (inclusive)."""
    chain = _ancestors_chain(g, n)
    if anc in chain:
        idx = chain.index(anc)
        return chain[: idx + 1]
    return chain  # if no anc, return full chain


def minimal_subgraph(g: nx.Graph, seeds: list[str], lca: str | None) -> nx.DiGraph:
    """
    Construct minimal subgraph containing seeds, their paths to LCA, and inter-cluster relations along those paths.
    Returns a new directed graph.
    """
    H = nx.DiGraph()
    included: set[str] = set()
    paths: list[list[str]] = []
    for s in seeds:
        p = _path_to_ancestor(g, s, lca) if lca else _ancestors_chain(g, s)
        paths.append(p)
        for i, node in enumerate(p):
            included.add(node)
            H.add_node(node, **g.nodes[node])
            if i > 0:
                parent = p[i]
                child = p[i - 1]
                if g.has_edge(parent, child):
                    H.add_edge(parent, child, **g.edges[parent, child])
                else:
                    # ensure parent->child "aggregates" link is present
                    H.add_edge(parent, child, kind="aggregates")
    # add inter-cluster relations among included aggregate nodes on same level
    agg_nodes = [n for n in included if g.nodes[n].get("type") == "aggregate"]
    for i in range(len(agg_nodes)):
        a = agg_nodes[i]
        for j in range(i + 1, len(agg_nodes)):
            b = agg_nodes[j]
            if g.has_edge(a, b):
                e = g.edges[a, b]
                if e.get("kind") == "relates":
                    H.add_edge(a, b, **e)
    return H


def format_brief(g: nx.DiGraph, max_nodes: int = LEANRAG.brief_max_nodes) -> str:
    """
    Convert a minimal subgraph into a compact textual brief for prompts.
    Prefers listing leaf seeds, then rolling up via aggregates with their summaries.
    """
    lines: list[str] = []
    # Identify leaves (no outgoing "aggregates")
    leaves = [n for n, d in g.out_degree() if d == 0]
    # Cap node listing
    listed = set()
    for n in leaves:
        if len(listed) >= max_nodes:
            break
        nd = g.nodes[n]
        name = nd.get("name") or n
        desc = nd.get("description", "")
        lines.append(f"- Seed: {name} :: {desc}".strip())
        listed.add(n)
    # Roll-up aggregates
    # Topologically order so parents after children
    try:
        order = list(nx.topological_sort(g))
    except Exception:
        order = list(g.nodes())
    for n in order:
        if len(listed) >= max_nodes:
            break
        nd = g.nodes[n]
        if nd.get("type") == "aggregate":
            name = nd.get("name") or n
            desc = nd.get("description", "")
            ch = nd.get("children", [])
            inter = ", ".join(ch[:5]) + ("..." if len(ch) > 5 else "")
            lines.append(f"- Cluster: {name} :: {desc} :: children≈[{inter}]")
            listed.add(n)
    return "\n".join(lines[:max_nodes])


def retrieve_minimal_brief(
    g: nx.Graph,
    embedder: EmbeddingProvider,
    query: str,
    seeds_k: int = LEANRAG.seeds_k,
    brief_max_nodes: int = LEANRAG.brief_max_nodes,
) -> dict[str, object]:
    """
    LeanRAG pipeline:
      1) Anchor seeds by dense similarity to base-layer entities.
      2) Find LCA in aggregation hierarchy.
      3) Build minimal subgraph (seeds + paths + inter-cluster).
      4) Return textual brief and graph stats.
    """
    top = anchor_seeds(g, embedder, query, k=seeds_k)
    seeds = [n for n, _ in top]
    lca = find_lca(g, seeds)
    H = minimal_subgraph(g, seeds, lca)
    brief = format_brief(H, max_nodes=brief_max_nodes)
    return {
        "seeds": top,
        "lca": lca,
        "nodes": list(H.nodes()),
        "edges": list(H.edges()),
        "brief": brief,
    }
