from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure src/ is importable when tests are executed from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stores.episodic import episodic_store
from src.stores.semantic import semantic_store
from src.stores.working import working_buffer


@pytest.fixture(autouse=True)
def clear_stores(tmp_path):
    episodic_store.memories.clear()
    episodic_store.memory_index.clear()
    episodic_store.embeddings_map.clear()
    if hasattr(episodic_store.index, "reset"):
        episodic_store.index.reset()
    semantic_store.db_path = tmp_path / "semantic.db"
    semantic_store._init_db()  # type: ignore[attr-defined]
    working_buffer.clear()
    yield
