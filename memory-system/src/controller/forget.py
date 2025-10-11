"""Forgetting policies for episodic memories."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

from src.stores.episodic import episodic_store
from src.stores.semantic import semantic_store


def run_forgetting_policy(ttl_expired: bool = True, demote_low_importance: bool = True) -> Dict[str, int]:
    now = datetime.utcnow()
    expired_ids: List[str] = []
    demoted = 0

    if ttl_expired:
        for memory in list(episodic_store.memories):
            expiry = memory.ts + timedelta(days=memory.ttl_days)
            if expiry < now:
                expired_ids.append(memory.id)
    if expired_ids:
        episodic_store.remove(expired_ids)

    if demote_low_importance:
        for memory in episodic_store.memories:
            if memory.importance < 0.2 and memory.access_count == 0:
                memory.ttl_days = min(memory.ttl_days, 30)
                demoted += 1

    return {
        "expired_removed": len(expired_ids),
        "demoted": demoted,
        "episodic_remaining": episodic_store.count(),
        "semantic_count": semantic_store.count(),
    }
