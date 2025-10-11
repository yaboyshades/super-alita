from __future__ import annotations

from src.controller.consolidate import (
    _ace_context_clustering,
    _get_ace_consolidation_candidates,
    run_ace_informed_consolidation,
    run_consolidation,
)
from src.models import Memory
from src.stores.episodic import episodic_store


def _store_memory(text: str, **meta) -> Memory:
    memory = Memory(text=text, importance=meta.get("importance", 0.65), meta=meta)
    episodic_store.add(memory)
    return memory


class TestACEConsolidation:
    def test_ace_context_clustering(self) -> None:
        strong = _store_memory(
            "Energy topic with conflicting evidence.",
            topic="energy",
            contradiction_count=2,
            stance="counterexample",
        )
        similar = _store_memory(
            "Energy topic supporting detail.",
            topic="energy",
            contradiction_count=0,
            stance="support",
        )
        other = _store_memory(
            "Completely different topic.",
            topic="hobby",
            contradiction_count=0,
            stance="support",
        )
        candidates = _get_ace_consolidation_candidates()
        assert strong in candidates
        clusters = _ace_context_clustering(candidates)
        assert any(strong in cluster and similar in cluster for cluster in clusters)
        assert all(other not in cluster or strong not in cluster for cluster in clusters if other in cluster)

    def test_strategy_performance_tracking(self) -> None:
        _store_memory(
            "Conflicting beverage preference noted.",
            topic="beverage",
            contradiction_count=1,
            stance="counterexample",
        )
        _store_memory(
            "Follow-up beverage insight.",
            topic="beverage",
            contradiction_count=0,
            stance="support",
        )
        result = run_ace_informed_consolidation()
        assert result["clusters_processed"] >= 1
        assert result["strategies"]
        assert {"strategy", "cluster_size", "succeeded"}.issubset(result["strategies"][0].keys())

    def test_backward_compatibility(self) -> None:
        _store_memory("Legacy consolidation item.", topic="legacy", contradiction_count=0)
        result = run_consolidation()
        assert "consolidated" in result
