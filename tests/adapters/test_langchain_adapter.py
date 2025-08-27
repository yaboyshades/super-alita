"""Tests for LangChain adapter fallbacks."""

import pytest

from src.adapters.langchain_adapter import Runnable


def test_runnable_init_raises_not_implemented() -> None:
    """Instantiating the fallback class should raise."""
    with pytest.raises(NotImplementedError, match="LangChain not installed"):
        Runnable()


def test_runnable_invoke_raises_not_implemented() -> None:
    """Calling `invoke` on a bypassed instance should raise."""
    runnable = object.__new__(Runnable)
    with pytest.raises(NotImplementedError, match="LangChain not installed"):
        runnable.invoke()
