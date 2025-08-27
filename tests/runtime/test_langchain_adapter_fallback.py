import pytest

from src.reug_runtime import adapter


def test_local_runnable_fallback(monkeypatch):
    """When HAS_LANGCHAIN is False, factories return local implementations."""
    monkeypatch.setattr(adapter, "HAS_LANGCHAIN", False)
    add_one = adapter.runnable_lambda(lambda x: x + 1)
    times_two = adapter.runnable_lambda(lambda x: x * 2)
    seq = adapter.runnable_sequence([add_one, times_two])
    assert isinstance(add_one, adapter.LocalRunnableLambda)
    assert isinstance(seq, adapter.LocalRunnableSequence)
    assert seq.invoke(3) == 8
