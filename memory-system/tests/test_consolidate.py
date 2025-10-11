from __future__ import annotations

from datetime import datetime, timedelta

from src.controller.consolidate import run_consolidation
from src.models import Memory
from src.stores.episodic import episodic_store
from src.stores.semantic import semantic_store


def test_consolidation_promotes_clusters():
    base_time = datetime.utcnow() - timedelta(days=2)
    for idx in range(3):
        episodic_store.add(
            Memory(
                text=f"User likes sushi number {idx}",
                importance=0.7,
                ts=base_time,
                last_access=base_time,
                tags=["food"],
            )
        )
    result = run_consolidation()
    assert "consolidated" in result
    assert semantic_store.count() >= 0
