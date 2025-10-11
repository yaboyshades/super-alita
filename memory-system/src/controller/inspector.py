"""Inspector producing structured decisions for retrieved memories."""
from __future__ import annotations

from typing import List

from src.models import Decision, Memory


def make_decisions(memories: List[Memory], query: str) -> List[Decision]:
    decisions: List[Decision] = []
    for memory in memories:
        actions: List[str] = []
        caveats: List[str] = []
        confidence = min(1.0, 0.5 + memory.importance / 2)
        if memory.importance > 0.7:
            actions.append("promote")
        elif memory.importance < 0.2:
            actions.append("discard")
            caveats.append("low_importance")
        if "?" in query and memory.importance > 0.4:
            actions.append("summarize_for_context")
        decisions.append(
            Decision(
                claim=_summarize_claim(memory.text),
                evidence_ids=[memory.id],
                confidence=confidence,
                caveats=caveats,
                actions=actions,
            )
        )
    return decisions


def _summarize_claim(text: str, max_words: int = 16) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."
