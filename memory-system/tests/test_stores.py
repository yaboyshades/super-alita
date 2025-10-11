from __future__ import annotations

import numpy as np

from src.models import Memory
from src.stores import episodic as episodic_module


def test_episodic_store_numpy_fallback(monkeypatch, tmp_path):
    """Episodic store should operate when FAISS is unavailable."""

    monkeypatch.setattr(episodic_module, "faiss", None)
    store = episodic_module.EpisodicStore(persist_path=tmp_path)

    memory = Memory(text="Fallback indexing works", importance=0.4)
    store.add(memory, embedding=list(np.ones(store.dimension)))

    results = store.search("fallback", k=1)
    assert results and results[0].id == memory.id
    assert getattr(store.index, "ntotal", 0) == 1
