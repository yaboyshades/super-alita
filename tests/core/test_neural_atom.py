"""Trimmed NeuralAtom tests (post-merge cleanup)."""

import pytest
from src.core.neural_atom import NeuralAtom, NeuralAtomMetadata, TextualMemoryAtom


class MockNeuralAtom(NeuralAtom):
    def __init__(self, metadata: NeuralAtomMetadata, test_value: str = "test"):
        super().__init__(metadata)
        self.test_value = test_value
        self.key = metadata.name

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


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
