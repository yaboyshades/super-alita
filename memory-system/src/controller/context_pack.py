"""Context pack composition."""
from __future__ import annotations

from typing import Dict, List

from src.models import ContextPack, Decision, Memory


def compose_context_pack(
    decisions: List[Decision], memories: List[Memory], budget: int = 700, query: str = ""
) -> ContextPack:
    memory_lookup = {memory.id: memory for memory in memories}

    sections: List[str] = []
    citations: List[str] = []
    used_tokens = 0

    header = f"# Context for: {query}\n\n" if query else "# Context (auto-compiled)\n\n"
    sections.append(header)
    used_tokens += len(header.split())

    if decisions:
        sections.append("## Decisions\n")
        for decision in decisions:
            decision_text = f"- {decision.claim} (confidence: {decision.confidence:.2f})\n"
            if decision.caveats:
                decision_text += f"  Caveats: {', '.join(decision.caveats)}\n"
            sections.append(decision_text)
        sections.append("\n")

    sections.append("## Evidence\n")
    for memory in memories[:12]:
        snippet = _truncate_words(memory.text, 25)
        tags = " ".join(f"#{tag}" for tag in memory.tags[:3])
        entry = (
            f"- {snippet} [id:{memory.id[:8]} importance:{memory.importance:.2f} {tags}]\n"
        )
        entry_tokens = len(entry.split())
        if used_tokens + entry_tokens > budget * 0.9:
            break
        sections.append(entry)
        used_tokens += entry_tokens
        citations.append(memory.id)

    if citations:
        sections.append(f"\n## Citations ({len(citations)})\n")
        sections.append(f"Referenced memory IDs: {', '.join(citations)}\n")

    full_text = "".join(sections)
    truncated = _truncate_tokens(full_text, budget)

    return ContextPack(
        text=truncated,
        citations=citations,
        decisions=decisions,
        provenance={
            "policy": "rules.default.yaml",
            "query": query,
            "memory_count": str(len(memories)),
            "decision_count": str(len(decisions)),
        },
        budget_used=len(truncated.split()),
        budget_total=budget,
    )


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def _truncate_tokens(text: str, max_tokens: int) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])
