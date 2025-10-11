"""ACE context evolution utilities."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List

from src.models import ContextPack, Decision, Memory
from src.stores.episodic import episodic_store

logger = logging.getLogger(__name__)


class ACEvolver:
    """Apply ACE-inspired transformations to context packs."""

    def __init__(self) -> None:
        self.evolution_cycle = 0

    def evolve_context(self, context_pack: ContextPack, feedback: Dict[str, Any]) -> ContextPack:
        """Return an evolved context pack based on runtime feedback."""

        self.evolution_cycle += 1
        if feedback is None:
            return context_pack

        composer = feedback.get("composer")
        if composer is None:
            raise ValueError("ACE evolution requires a composer callable in feedback['composer']")

        memories: List[Memory] = list(feedback.get("memories", []))
        decisions: List[Decision] = list(feedback.get("decisions", context_pack.decisions))
        operations: List[str] = []

        if self._needs_evidence_expansion(decisions, feedback):
            memories = self._expand_evidence(memories, decisions, feedback)
            operations.append("expand_evidence")

        if feedback.get("contradiction_detected") or feedback.get("contradictions"):
            memories = self._add_counterexamples(memories, decisions, feedback)
            operations.append("add_counterexamples")

        if feedback.get("clarity_feedback") or feedback.get("restructure", True):
            memories = self._restructure_for_clarity(memories, feedback)
            operations.append("restructure")

        unique_memories = self._deduplicate(memories)
        budget = int(feedback.get("budget", context_pack.budget_total))
        query = feedback.get("query") or context_pack.provenance.get("query", "")
        evolved = composer(decisions, unique_memories, budget=budget, query=query)

        provenance_updates = {
            "ace_cycle": str(self.evolution_cycle),
            "ace_operations": ",".join(operations) if operations else "none",
        }
        if "success_metrics" in feedback:
            provenance_updates["ace_success_metrics"] = ",".join(
                sorted(str(metric) for metric in feedback["success_metrics"])
            )
        evolved.provenance.update({k: str(v) for k, v in provenance_updates.items()})
        return evolved

    def _needs_evidence_expansion(
        self, decisions: Iterable[Decision], feedback: Dict[str, Any]
    ) -> bool:
        if feedback.get("low_confidence") or feedback.get("trigger_condition") == "low_confidence":
            return True
        threshold = float(feedback.get("confidence_threshold", 0.55))
        return any(decision.confidence < threshold for decision in decisions)

    def _expand_evidence(
        self, memories: List[Memory], decisions: Iterable[Decision], feedback: Dict[str, Any]
    ) -> List[Memory]:
        expanded = list(memories)
        seen_ids = {memory.id for memory in expanded}
        desired = max(len(expanded) + 2, int(feedback.get("evidence_target", len(expanded) + 2)))
        query = feedback.get("query")

        for decision in decisions:
            if decision.confidence >= float(feedback.get("confidence_threshold", 0.55)) and not feedback.get(
                "low_confidence"
            ):
                continue
            search_terms = [term for term in [decision.claim, query] if term]
            for term in search_terms:
                for candidate in episodic_store.search(term, k=3):
                    if candidate.id in seen_ids:
                        continue
                    expanded.append(candidate)
                    seen_ids.add(candidate.id)
                    if len(expanded) >= desired:
                        return expanded
        return expanded

    def _add_counterexamples(
        self,
        memories: List[Memory],
        decisions: Iterable[Decision],
        feedback: Dict[str, Any],
    ) -> List[Memory]:
        expanded = list(memories)
        seen_ids = {memory.id for memory in expanded}
        contradictions = feedback.get("contradictions") or []

        if isinstance(contradictions, list) and contradictions:
            for entry in contradictions:
                if isinstance(entry, dict):
                    term = entry.get("claim") or entry.get("text") or feedback.get("query")
                else:
                    term = str(entry)
                if not term:
                    continue
                for candidate in episodic_store.search(term, k=2):
                    if candidate.id in seen_ids:
                        continue
                    expanded.append(candidate)
                    seen_ids.add(candidate.id)
        else:
            for decision in decisions:
                if "demote" not in decision.actions and "discard" not in decision.actions:
                    continue
                for candidate in episodic_store.search(decision.claim, k=1):
                    if candidate.id in seen_ids:
                        continue
                    expanded.append(candidate)
                    seen_ids.add(candidate.id)
        return expanded

    def _restructure_for_clarity(
        self, memories: List[Memory], feedback: Dict[str, Any]
    ) -> List[Memory]:
        if not memories:
            return memories
        clarity_feedback = feedback.get("clarity_feedback") or {}

        def clarity_key(memory: Memory) -> float:
            hint = clarity_feedback.get(memory.id)
            if isinstance(hint, (int, float)):
                return float(hint)
            meta_score = memory.meta.get("clarity_score") if isinstance(memory.meta, dict) else None
            if isinstance(meta_score, (int, float)):
                return float(meta_score)
            return float(memory.confidence)

        sorted_memories = sorted(
            memories,
            key=lambda memory: (clarity_key(memory), memory.importance),
            reverse=True,
        )
        return sorted_memories

    @staticmethod
    def _deduplicate(memories: List[Memory]) -> List[Memory]:
        unique: List[Memory] = []
        seen = set()
        for memory in memories:
            if memory.id in seen:
                continue
            unique.append(memory)
            seen.add(memory.id)
        return unique
