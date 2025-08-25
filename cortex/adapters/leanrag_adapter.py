# cortex/adapters/leanrag_adapter.py
from typing import Any

import networkx as nx

from cortex.config.flags import LEANRAG as LR_FLAGS
from cortex.config.flags import LeanRAGFlags
from cortex.kg.leanrag import LeanRAG


def build_situation_brief(
    kg_graph: nx.Graph, query: str, embedder=None, flags: LeanRAGFlags = None
) -> dict[str, Any]:
    """
    Produce a concise situation brief using LeanRAG
    """
    flags = flags or LR_FLAGS
    if not flags.enable:
        return {"enabled": False, "brief": "", "reason": "LeanRAG disabled"}

    # Fallback embedder (deterministic hashing) if orchestrator didn't provide one
    if embedder is None:
        try:
            from cortex.kg.embeddings import HashingEmbedder

            embedder = HashingEmbedder(dim=256)
        except Exception:
            raise RuntimeError("No embedder provided and HashingEmbedder unavailable")

    leanrag = LeanRAG(embedder, flags)

    # Ensure hierarchy is built
    if not any(n for n, d in kg_graph.nodes(data=True) if d.get("type") == "aggregate"):
        kg_graph = leanrag.build_hierarchy(kg_graph)

    # Get retrieval result
    result = leanrag.retrieve(kg_graph, query)

    return {
        "enabled": True,
        "brief": result["brief"],
        "seeds": result["seeds"],
        "lca": result["lca"],
        "subgraph_size": len(result["subgraph"].nodes),
        "strategy": "LeanRAG::LCA",
    }
