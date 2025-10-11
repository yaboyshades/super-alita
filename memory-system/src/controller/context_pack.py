"""Context pack composition."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.models import ContextPack, Decision, Memory

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.controller.ace_evolver import ACEvolver


logger = logging.getLogger(__name__)


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


def compose_ace_context(
    decisions: List[Decision],
    memories: List[Memory],
    *,
    budget: int = 700,
    query: str = "",
    feedback: Optional[Dict[str, Any]] = None,
    evolver: Optional["ACEvolver"] = None,
) -> ContextPack:
    """Compose an ACE-aware context pack.

    The function first composes a traditional context pack and then optionally evolves it
    via the provided ``ACEvolver``. Provenance keys flag whether ACE processing occurred
    and surface the applied feedback keys for observability.
    """

    base_pack = compose_context_pack(decisions, memories, budget=budget, query=query)
    if not feedback or evolver is None:
        return base_pack

    payload: Dict[str, Any] = dict(feedback)
    payload.setdefault("memories", list(memories))
    payload.setdefault("decisions", list(decisions))
    payload.setdefault("budget", budget)
    payload.setdefault("query", query)
    payload.setdefault("composer", compose_context_pack)

    try:
        evolved_pack = evolver.evolve_context(base_pack, payload)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("ACE evolution failed, falling back to base context: %s", exc)
        base_pack.provenance.setdefault("ace_error", str(exc))
        return base_pack

    evolved_pack.provenance.setdefault("ace_enabled", "true")
    evolved_pack.provenance["ace_feedback_keys"] = ",".join(sorted(payload.keys()))
    return evolved_pack
