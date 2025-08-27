from __future__ import annotations

from cortex.planner import parallel_wrapper as pw


class DummyRunnable:
    """Simple runnable that records invocation order."""

    def __init__(self, name: str, record: list[str]):
        self.name = name
        self.record = record

    def invoke(self, value: list[str]) -> list[str]:
        self.record.append(self.name)
        return value + [self.name]


def test_sequential_execution_without_langchain(monkeypatch):
    """Simulate missing LangChain and ensure execution is sequential."""

    record: list[str] = []
    runnables = {
        "first": DummyRunnable("first", record),
        "second": DummyRunnable("second", record),
    }

    monkeypatch.setattr(pw, "HAS_LANGCHAIN", False)

    wrapper = pw.parallel_wrapper(runnables)
    result = wrapper.invoke([])

    assert record == ["first", "second"]
    assert result == ["first", "second"]
    assert not pw.should_parallelize(runnables)
