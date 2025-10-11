from __future__ import annotations

from src.controller.conflict import detect_and_resolve_conflicts
from src.models import Memory
from src.stores.episodic import episodic_store


def test_conflict_detection_identifies_contradictions():
    episodic_store.add(Memory(text="I like coffee", importance=0.6))
    episodic_store.add(Memory(text="I do not like coffee", importance=0.6))
    conflicts = detect_and_resolve_conflicts(limit=5, auto_resolve=False)
    assert conflicts
    assert any(conflict.conflict_type == "contradiction" for conflict in conflicts)


def test_conflict_auto_resolution_demotes_lower_confidence():
    mem_a = Memory(text="Event happens tomorrow", importance=0.6, confidence=0.9)
    mem_b = Memory(text="Event happens tomorrow", importance=0.6, confidence=0.2)
    episodic_store.add(mem_a)
    episodic_store.add(mem_b)
    conflicts = detect_and_resolve_conflicts(limit=5, auto_resolve=True)
    assert conflicts
    assert any(conflict.resolution for conflict in conflicts)
    assert mem_b.importance < 0.6
