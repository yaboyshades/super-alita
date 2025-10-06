"""Trimmed NeuralAtom tests (post-merge cleanup)."""

import numpy as np
import pytest

from src.core.neural_atom import (
    NeuralAtom,
    NeuralAtomMetadata,
    TextualMemoryAtom,
    create_goal_atom,
    create_memory_atom,
    create_memory_chunk_atom,
    create_skill_atom,
)


class MockNeuralAtom(NeuralAtom):
    def __init__(self, metadata: NeuralAtomMetadata, test_value: str = "test"):
        super().__init__(
            key=metadata.name,
            value=test_value,
            metadata=metadata,
            birth_event="mock:test",
        )
        self.test_value = test_value

    async def execute(self, input_data=None):  # pragma: no cover - trivial
        return {"result": self.test_value, "input": input_data}

    def get_embedding(self):  # pragma: no cover - deterministic
        return [0.1] * 128

    def can_handle(self, task_description: str):  # pragma: no cover - simple logic
        return 0.8 if "test" in task_description.lower() else 0.2


def _md(name: str, caps):
    return NeuralAtomMetadata(name=name, description=name, capabilities=caps)


@pytest.mark.asyncio
async def test_execute_and_safe_metrics():
    atom = MockNeuralAtom(_md("exec", ["run"]))
    res = await atom.execute("x")
    assert res["result"] == "test"
    safe = await atom.safe_execute("y")
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


def test_factory_skill_atom_instantiation():
    atom = create_skill_atom(
        skill_name="planner",
        skill_code="def run(): return True",
        skill_description="planning utility",
        metadata={"capabilities": ["plan"]},
    )

    assert isinstance(atom, NeuralAtom)
    assert atom.key == "skill:planner"
    assert atom.value["type"] == "skill"
    assert atom.metadata.name == "skill:planner"


def test_factory_memory_atom_instantiation():
    atom = create_memory_atom(
        memory_key="session-1",
        content={"note": "remember"},
        memory_type="session",
        confidence=0.9,
    )

    assert atom.key == "memory:session:session-1"
    assert atom.lineage_metadata["memory_type"] == "session"
    assert atom.value["content"]["note"] == "remember"


def test_factory_goal_atom_instantiation():
    atom = create_goal_atom(
        goal_id="goal-123",
        goal_description="Ship feature",
        priority=0.75,
    )

    assert atom.key == "goal:goal-123"
    assert atom.value["priority"] == 0.75
    assert atom.lineage_metadata["goal_type"] == "primary"


def test_factory_memory_chunk_atom_instantiation():
    vector = np.zeros(8, dtype=np.float32)
    atom = create_memory_chunk_atom(
        key="memory:chunk:1",
        content={"data": "value"},
        hierarchy_path=["root", "child"],
        vector=vector,
    )

    assert atom.key == "memory:chunk:1"
    assert atom.value["content"]["data"] == "value"
    assert atom.value["hierarchy_path"] == ["root", "child"]


import pytest

def test_factory_memory_chunk_atom_invalid_vector_shape():
    # Vector with wrong shape
    vector = np.zeros(7, dtype=np.float32)
    with pytest.raises(ValueError):
        create_memory_chunk_atom(
            key="memory:chunk:bad-shape",
            content={"data": "bad"},
            hierarchy_path=["root"],
            vector=vector,
        )

def test_factory_memory_chunk_atom_invalid_vector_dtype():
    # Vector with wrong dtype
    vector = np.zeros(8, dtype=np.float64)
    with pytest.raises(ValueError):
        create_memory_chunk_atom(
            key="memory:chunk:bad-dtype",
            content={"data": "bad"},
            hierarchy_path=["root"],
            vector=vector,
        )

def test_factory_memory_chunk_atom_vector_none():
    # Vector is None
    with pytest.raises(ValueError):
        create_memory_chunk_atom(
            key="memory:chunk:none-vector",
            content={"data": "bad"},
            hierarchy_path=["root"],
            vector=None,
        )
if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
