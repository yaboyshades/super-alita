from __future__ import annotations

from typing import Any

from src.core.decision_policy_v1 import DecisionPolicyEngine

try:
    from src.core.neural_atom import NeuralAtom, NeuralStore  # type: ignore
except Exception:  # pragma: no cover
    NeuralStore = object  # type: ignore
    NeuralAtom = object  # type: ignore

class DecisionPolicyWithMemory:
    def __init__(self):
        self.engine = DecisionPolicyEngine()
        try:
            self.neural_store = NeuralStore()
        except Exception:
            self.neural_store = None

    def register_neural_atom_capability(self, atom: Any):
        if getattr(atom, "atom_type", "") == "tool" and hasattr(
            self.engine, "register_capability"
        ):
            get_meta = getattr(
                getattr(atom, "metadata", {}), "get", lambda _k, d=None: d
            )
            cap = {
                "name": f"neural_{getattr(atom, 'atom_id', 'unknown')}",
                "description": getattr(atom, "description", ""),
                "type": "neural_capability",
                "confidence": get_meta("confidence", 0.5),
            }
            self.engine.register_capability(cap)  # type: ignore[attr-defined]

    async def enhance_context_with_memory(self, goal_desc: str) -> dict[str, Any]:
        if not self.neural_store or not hasattr(self.neural_store, "similarity_search"):
            return {"memories": [], "memory_confidence": 0.0}
        memories = await self.neural_store.similarity_search(goal_desc, top_k=5)  # type: ignore[attr-defined]
        return {"memories": memories, "memory_confidence": min(1.0, len(memories)/5.0)}
