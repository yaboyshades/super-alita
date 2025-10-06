"""Trimmed NeuralAtom tests (post-merge cleanup)."""

import asyncio

import numpy as np
import pytest

from src.core.neural_atom import (
    NeuralAtom,
    NeuralAtomMetadata,
    NeuralStore,
    TextualMemoryAtom,
)


class MockNeuralAtom(NeuralAtom):
    def __init__(self, metadata: NeuralAtomMetadata, test_value: str = "test"):
        super().__init__(metadata)
        self.test_value = test_value

    async def execute(self, input_data=None):  # pragma: no cover - trivial
        return {"result": self.test_value, "input": input_data}

    def get_embedding(self):  # pragma: no cover - deterministic
        return [0.1] * 128

    def can_handle(self, task_description: str):  # pragma: no cover - simple logic
        return 0.8 if "test" in task_description.lower() else 0.2


class SkillAtom(NeuralAtom):
    def __init__(
        self,
        metadata: NeuralAtomMetadata,
        instructions: str,
        activation_scale: float = 0.5,
    ):
        vector = (
            np.ones(NeuralAtom.DEFAULT_VECTOR_DIM, dtype=np.float32) * activation_scale
        )
        super().__init__(metadata, vector=vector, bias=0.1)
        self.instructions = instructions

    async def execute(self, input_data=None):  # pragma: no cover - simple mapping
        return {
            "instructions": self.instructions,
            "input": input_data,
            "bias": self.bias,
        }

    def get_embedding(self):  # pragma: no cover - deterministic
        self.vector = np.asarray(self.vector, dtype=np.float32)
        return self.vector.tolist()

    def can_handle(self, task_description: str):  # pragma: no cover - simple logic
        return 1.0 if "skill" in task_description.lower() else 0.3


def _md(name: str, caps):
    return NeuralAtomMetadata(name=name, description=name, capabilities=caps)


def test_execute_and_safe_metrics():
    atom = MockNeuralAtom(_md("exec", ["run"]))
    res = asyncio.run(atom.execute("x"))
    assert res["result"] == "test"
    safe = asyncio.run(atom.safe_execute("y"))
    assert safe["success"] and atom.metadata.usage_count == 1


def test_textual_memory_embedding():
    mem = TextualMemoryAtom(_md("mem", ["memory"]), "content")
    emb = mem.get_embedding()
    assert isinstance(emb, list) and len(emb) == 128


def test_performance_tracking_updates():
    atom = MockNeuralAtom(_md("perf", ["perf"]))
    atom._update_performance_metrics(0.05, True)
    atom._update_performance_metrics(0.10, False)
    assert atom.metadata.usage_count == 2
    assert 0 < atom.metadata.success_rate < 1.0
    assert atom.metadata.avg_execution_time > 0


def test_neural_store_forward_pass_and_genealogy():
    store = NeuralStore()
    text_atom = TextualMemoryAtom(_md("memory_atom", ["memory"]), "stored content")
    skill_atom = SkillAtom(
        _md("skill_atom", ["skill", "plan"]),
        instructions="Do something clever",
        activation_scale=0.2,
    )

    store.register(text_atom)
    store.register(skill_atom)
    store.add_synapse(text_atom.key, skill_atom.key)

    asyncio.run(store.forward_pass())
    assert text_atom.activation_count == 1
    assert skill_atom.activation_count == 1

    child_atom = TextualMemoryAtom(
        _md("child_memory", ["memory", "history"]),
        "child content",
    )
    store.register_with_lineage(
        child_atom,
        parents=[text_atom, skill_atom],
        birth_event="unit_test_birth",
        lineage_metadata={"source": "unit_test"},
    )

    stored_child = store.get(child_atom.key)
    assert stored_child is child_atom
    assert set(child_atom.parent_keys) == {text_atom.key, skill_atom.key}
    assert child_atom.depth == 1 + max(
        parent.depth for parent in [text_atom, skill_atom]
    )
    assert child_atom.key in text_atom.children_keys
    assert child_atom.key in skill_atom.children_keys

    lineage_related = asyncio.run(
        store.genealogy_attention(child_atom.key, generations=1)
    )
    related_keys = {entry[0] for entry in lineage_related}
    assert text_atom.key in related_keys


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
