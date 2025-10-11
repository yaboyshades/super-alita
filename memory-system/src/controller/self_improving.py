"""Self-improving strategy controller for ACE evolution."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from src.models import ACEContextStrategy, ContextPack


class SelfImprovingMemoryController:
    """Track and evolve ACE context strategies based on feedback."""

    def __init__(self) -> None:
        self.strategy_performance: Dict[str, List[float]] = {}
        self.evolution_history: List[Dict[str, Any]] = []
        self.evolution_cycle = 0

    def evaluate_context_strategy(
        self, context_pack: ContextPack, llm_feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate the latest context against feedback and evolve strategies."""

        self.evolution_cycle += 1
        metrics = {
            "completeness": self._calculate_completeness(context_pack, llm_feedback),
            "contradiction_resolution": 1.0 if llm_feedback.get("contradictions_resolved", True) else 0.0,
            "evidence_strength": float(
                llm_feedback.get(
                    "evidence_strength",
                    min(1.0, len(context_pack.citations) / max(1, len(context_pack.decisions) or 1)),
                )
            ),
            "clarity": float(llm_feedback.get("clarity", max(0.0, 1.0 - float(llm_feedback.get("confusion", 0.0)))))
            if "clarity" in llm_feedback or "confusion" in llm_feedback
            else min(1.0, context_pack.budget_used / max(1, context_pack.budget_total)),
        }
        strategies = self._evolve_strategies(metrics)

        timestamp = datetime.utcnow().isoformat()
        for strategy in strategies:
            strategy.last_applied = datetime.fromisoformat(timestamp)

        record = {
            "cycle": self.evolution_cycle,
            "timestamp": timestamp,
            "metrics": metrics,
            "strategies": [strategy.model_dump() for strategy in strategies],
        }
        self.evolution_history.append(record)
        return record

    def _calculate_completeness(
        self, context_pack: ContextPack, feedback: Dict[str, Any]
    ) -> float:
        if not context_pack.citations:
            return 0.0
        expected = max(1, int(feedback.get("expected_citations", len(context_pack.citations))))
        coverage = min(1.0, len(context_pack.citations) / expected)
        missing = feedback.get("missing_information")
        if isinstance(missing, (int, float)):
            coverage -= min(0.5, float(missing))
        elif missing:
            coverage -= 0.2
        penalty = feedback.get("penalty", 0.0)
        if isinstance(penalty, (int, float)):
            coverage -= min(0.5, float(penalty))
        return max(0.0, min(1.0, coverage))

    def _evolve_strategies(self, metrics: Dict[str, float]) -> List[ACEContextStrategy]:
        strategies: List[ACEContextStrategy] = []
        now = datetime.utcnow()

        def register(strategy: ACEContextStrategy, metric_key: str) -> None:
            history = self.strategy_performance.setdefault(strategy.strategy_id, [])
            history.append(metrics.get(metric_key, 0.0))
            strategy.success_metrics = [metric_key]
            strategy.success_rate = sum(history) / len(history) if history else 0.0
            strategy.last_applied = now
            strategies.append(strategy)

        if metrics.get("completeness", 0.0) < 0.7:
            register(
                ACEContextStrategy(
                    strategy_id="ace.expand_evidence",
                    trigger_condition="low_confidence",
                    context_transform="expand_evidence",
                ),
                "completeness",
            )
        if metrics.get("contradiction_resolution", 1.0) < 0.9:
            register(
                ACEContextStrategy(
                    strategy_id="ace.counterexamples",
                    trigger_condition="contradiction_detected",
                    context_transform="add_counterexamples",
                ),
                "contradiction_resolution",
            )
        if metrics.get("clarity", 1.0) < 0.85:
            register(
                ACEContextStrategy(
                    strategy_id="ace.clarity",
                    trigger_condition="new_insight",
                    context_transform="restructure",
                ),
                "clarity",
            )
        return strategies
