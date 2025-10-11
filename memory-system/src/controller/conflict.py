"""Conflict detection between memories."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

from src.models import Conflict, Memory
from src.stores.episodic import episodic_store

_NEGATION_WORDS = {"not", "never", "no", "don't", "doesn't", "isn't"}
_POSITIVE_WORDS = {"like", "love", "enjoy", "prefer"}
_NEGATIVE_WORDS = {"dislike", "hate", "avoid"}


@dataclass
class _AnalyzedMemory:
    memory: Memory
    tokens: set[str]
    polarity: str


def detect_and_resolve_conflicts(limit: int = 10, auto_resolve: bool = False) -> List[Conflict]:
    analyzed = [_analyse(memory) for memory in episodic_store.memories]
    conflicts: List[Conflict] = []

    for idx, current in enumerate(analyzed):
        for other in analyzed[idx + 1 :]:
            conflict_type = _classify_conflict(current, other)
            if conflict_type:
                conflict = Conflict(
                    memory_a=current.memory.id,
                    memory_b=other.memory.id,
                    conflict_type=conflict_type,
                )
                if auto_resolve:
                    _resolve(conflict, current.memory, other.memory)
                conflicts.append(conflict)
            if len(conflicts) >= limit:
                return conflicts
    return conflicts


def _analyse(memory: Memory) -> _AnalyzedMemory:
    tokens = {token.strip(".,!?\"'") for token in memory.text.lower().split()}
    polarity = "neutral"
    if tokens & _NEGATIVE_WORDS or tokens & _NEGATION_WORDS:
        polarity = "negative"
    elif tokens & _POSITIVE_WORDS:
        polarity = "positive"
    return _AnalyzedMemory(memory=memory, tokens=tokens, polarity=polarity)


def _classify_conflict(a: _AnalyzedMemory, b: _AnalyzedMemory) -> str | None:
    overlap = a.tokens & b.tokens
    shared = {token for token in overlap if token not in _NEGATION_WORDS}
    if not shared:
        return None
    if abs(a.memory.confidence - b.memory.confidence) > 0.5:
        return "confidence"
    if {"yesterday", "today", "tomorrow"} & shared:
        return "temporal"
    if a.polarity != b.polarity and {"like", "dislike", "hate", "love", "prefer"} & shared:
        return "contradiction"
    return None


def _resolve(conflict: Conflict, a: Memory, b: Memory) -> None:
    if conflict.conflict_type == "confidence":
        if a.confidence > b.confidence:
            b.importance *= 0.5
            conflict.resolution = f"demoted {b.id}"
        else:
            a.importance *= 0.5
            conflict.resolution = f"demoted {a.id}"
    elif conflict.conflict_type == "contradiction":
        a.importance *= 0.9
        b.importance *= 0.9
        conflict.resolution = "soft_demote"
    elif conflict.conflict_type == "temporal":
        newer = a if a.last_access > b.last_access else b
        newer.importance += 0.1
        conflict.resolution = f"boosted {newer.id}"
    conflict.resolved_at = datetime.utcnow()
