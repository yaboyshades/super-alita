from __future__ import annotations

from src.controller.ace_evolver import ACEvolver
from src.controller.context_pack import compose_ace_context
from src.models import Decision, Memory
from src.stores.episodic import episodic_store


def _make_memory(text: str, importance: float = 0.5, **meta) -> Memory:
    memory = Memory(text=text, importance=importance, meta=meta or {})
    episodic_store.add(memory)
    return memory


class TestACEContext:
    def test_evidence_expansion_low_confidence(self) -> None:
        base = _make_memory("Solar adoption is rising in urban regions.", importance=0.6)
        _ = _make_memory("Further solar studies highlight rooftop benefits.", importance=0.5)
        decisions = [
            Decision(
                claim="Solar adoption is rising",
                evidence_ids=[base.id],
                confidence=0.4,
                caveats=[],
                actions=[],
            )
        ]
        pack = compose_ace_context(
            decisions,
            [base],
            budget=400,
            query="solar adoption",
            feedback={"low_confidence": True},
            evolver=ACEvolver(),
        )
        assert len(pack.citations) >= 2
        assert "ace_enabled" in pack.provenance

    def test_counterexamples_for_contradictions(self) -> None:
        original = _make_memory("User prefers coffee every morning.", importance=0.7)
        counter = _make_memory(
            "Despite prior claims, user preference shifts to tea.",
            importance=0.5,
            contradiction_count=1,
            stance="counterexample",
        )
        decisions = [
            Decision(
                claim="User prefers coffee every morning.",
                evidence_ids=[original.id],
                confidence=0.9,
                caveats=[],
                actions=["promote"],
            )
        ]
        pack = compose_ace_context(
            decisions,
            [original],
            budget=300,
            query="coffee preference",
            feedback={"contradictions": [{"claim": "coffee preference"}]},
            evolver=ACEvolver(),
        )
        assert counter.id in pack.citations

    def test_clarity_restructuring(self) -> None:
        low = _make_memory(
            "Low clarity evidence should appear later.",
            importance=0.4,
            clarity_score=0.2,
        )
        high = _make_memory(
            "High clarity evidence must surface first.",
            importance=0.4,
            clarity_score=0.9,
        )
        decisions = [
            Decision(
                claim="Clarity ordering",
                evidence_ids=[low.id, high.id],
                confidence=0.6,
                caveats=[],
                actions=[],
            )
        ]
        pack = compose_ace_context(
            decisions,
            [low, high],
            budget=300,
            query="clarity",
            feedback={"clarity_feedback": {}},
            evolver=ACEvolver(),
        )
        low_index = pack.text.index("Low clarity evidence")
        high_index = pack.text.index("High clarity evidence")
        assert high_index < low_index
