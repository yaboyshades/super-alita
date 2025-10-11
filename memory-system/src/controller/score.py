"""Deterministic scoring utilities for memories."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Iterable


def calculate_importance(text: str, role: str, meta: Dict[str, Any]) -> float:
    score = 0.0

    if "```" in text:
        score += 0.4
    if "?" in text and len(text) > 40:
        score += 0.3

    length = len(text)
    if length > 400:
        score += 0.3
    elif length > 200:
        score += 0.2
    elif length > 100:
        score += 0.1

    if role == "user":
        score += 0.1

    patterns: Iterable[tuple[str, float]] = [
        (r"\b(prefer|like|dislike|love|hate)\b", 0.3),
        (r"\b(always|never)\b", 0.2),
        (r"\b(important|critical|crucial|essential)\b", 0.25),
        (r"\b(address|email|phone|deadline|schedule)\b", 0.3),
        (r"\b(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})\b", 0.2),
    ]
    for pattern, boost in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += boost

    if meta.get("priority") == "high":
        score += 0.2
    if meta.get("topic") in {"preferences", "profile", "settings"}:
        score += 0.15

    return min(score, 1.0)


def calculate_recency_boost(memory_ts: datetime, half_life_days: int = 120) -> float:
    days_old = (datetime.utcnow() - memory_ts).days
    return max(0.2, 2 ** (-days_old / half_life_days))


def calculate_diversity_penalty(memories: Iterable, new_memory, threshold: float = 0.8) -> float:
    recent = list(memories)[-10:]
    max_similarity = 0.0
    for memory in recent:
        similarity = _text_similarity(memory.text, new_memory.text)
        max_similarity = max(max_similarity, similarity)
    if max_similarity > threshold:
        return -0.3
    return 0.0


def _text_similarity(text1: str, text2: str) -> float:
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union
