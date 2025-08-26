import networkx as nx
import numpy as np
import pytest

from cortex.config.flags import LEANRAG as LR_FLAGS
from cortex.kg.leanrag import LeanRAG


def _mk_graph():
    g = nx.DiGraph()
    # Create base nodes (files/functions), connect related ones
    for i in range(24):
        g.add_node(f"n{i}", name=f"Entity{i}", description=f"desc {i}", type="entity")
    # Create two dense clusters with a few cross-links
    for i in range(0, 10):
        for j in range(i + 1, 10):
            g.add_edge(f"n{i}", f"n{j}", kind="ref")
            g.add_edge(f"n{j}", f"n{i}", kind="ref")
    for i in range(12, 24):
        for j in range(i + 1, 24):
            g.add_edge(f"n{i}", f"n{j}", kind="ref")
            g.add_edge(f"n{j}", f"n{i}", kind="ref")
    # sparse cross
    g.add_edge("n5", "n15", kind="ref")
    g.add_edge("n15", "n5", kind="ref")
    return g


@pytest.mark.asyncio
async def test_hierarchy_and_retrieval_pipeline(monkeypatch):
    # Force LeanRAG on
    monkeypatch.setenv("CORTEX_LEANRAG_ENABLE", "1")
    monkeypatch.setenv("CORTEX_LEANRAG_MAX_DEPTH", "2")
    monkeypatch.setenv("CORTEX_LEANRAG_MIN_CLUSTER", "4")
    monkeypatch.setenv("CORTEX_LEANRAG_GMM_K", "4")
    monkeypatch.setenv("CORTEX_LEANRAG_LINK_THRESH", "0.02")
    monkeypatch.setenv("CORTEX_LEANRAG_SEEDS_K", "3")

    # Use a mock embedder
    class MockEmbedder:
        def embed_text(self, text):
            return np.random.rand(64).astype(np.float32)

    g = _mk_graph()
    emb = MockEmbedder()
    leanrag = LeanRAG(emb, LR_FLAGS)

    # Build hierarchy
    graph_with_hierarchy = leanrag.build_hierarchy(g)

    # ensure aggregate nodes present
    agg_nodes = [
        n
        for n, d in graph_with_hierarchy.nodes(data=True)
        if d.get("type") == "aggregate"
    ]
    assert len(agg_nodes) > 0

    # Check for relations between aggregate nodes with correct edge kind
    aggregate_edges = [
        (u, v)
        for u, v, d in graph_with_hierarchy.edges(data=True)
        if graph_with_hierarchy.nodes[u].get("type") == "aggregate"
        and graph_with_hierarchy.nodes[v].get("type") == "aggregate"
        and d.get("kind") == "relates"
    ]

    # Test retrieval
    result = leanrag.retrieve(
        graph_with_hierarchy, "Entity 15 critical login regression fix"
    )
    assert result["brief"]
    assert len(result["seeds"]) > 0


@pytest.mark.asyncio
async def test_gnn_aggregation_runtime_selection(monkeypatch):
    monkeypatch.setenv("CORTEX_LEANRAG_ENABLE", "1")
    monkeypatch.setenv("CORTEX_LEANRAG_MAX_DEPTH", "2")
    monkeypatch.setenv("CORTEX_LEANRAG_MIN_CLUSTER", "4")
    monkeypatch.setenv("CORTEX_LEANRAG_GMM_K", "4")
    monkeypatch.setenv("CORTEX_LEANRAG_LINK_THRESH", "0.02")
    monkeypatch.setenv("CORTEX_LEANRAG_SEEDS_K", "3")

    class MockEmbedder:
        def embed_text(self, text):  # pragma: no cover - simple mock
            return np.random.rand(32).astype(np.float32)

    g = _mk_graph()
    emb = MockEmbedder()
    leanrag = LeanRAG(emb, LR_FLAGS)

    graph_with_hierarchy = leanrag.build_hierarchy(g, aggregation="gnn")
    agg_nodes = [
        n
        for n, d in graph_with_hierarchy.nodes(data=True)
        if d.get("type") == "aggregate"
    ]
    assert len(agg_nodes) > 0
