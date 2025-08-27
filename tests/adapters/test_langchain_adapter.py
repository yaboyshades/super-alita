import pytest

from src.adapters.langchain_adapter import LangChainAdapter


def sample_func(a: int, b: str) -> str:
    return b * a


def test_invoke_valid_kwargs():
    adapter = LangChainAdapter()
    assert adapter.invoke(sample_func, a=2, b="x") == "xx"


def test_invoke_unexpected_kwargs_raises():
    adapter = LangChainAdapter()
    with pytest.raises(TypeError):
        adapter.invoke(sample_func, a=2, b="x", c=1)
