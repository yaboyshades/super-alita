"""Optional LangChain runnable compatibility layer.

This module attempts to import lightweight "Runnable" helpers from
``langchain_core``. When the dependency is unavailable, it falls back to
minimal local implementations so calling code can continue to operate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

try:  # pragma: no cover - optional dependency
    from langchain_core.runnables import RunnableLambda as _LCRunnableLambda
    from langchain_core.runnables import RunnableSequence as _LCRunnableSequence

    HAS_LANGCHAIN = True
except Exception:  # pragma: no cover - no langchain at runtime
    _LCRunnableLambda = None  # type: ignore[assignment]
    _LCRunnableSequence = None  # type: ignore[assignment]
    HAS_LANGCHAIN = False


class LocalRunnableLambda:
    """Simple callable wrapper mirroring LangChain's RunnableLambda."""

    def __init__(self, func: Callable[[Any], Any]) -> None:
        self._func = func

    def invoke(self, input_: Any) -> Any:
        return self._func(input_)


class LocalRunnableSequence:
    """Sequentially execute contained runnables."""

    def __init__(self, steps: Iterable[Any]) -> None:
        self._steps = list(steps)

    def invoke(self, input_: Any) -> Any:
        value = input_
        for step in self._steps:
            value = step.invoke(value)
        return value


def runnable_lambda(func: Callable[[Any], Any]) -> Any:
    """Factory returning a runnable lambda.

    Uses LangChain's implementation when available; otherwise returns a
    lightweight local version.
    """

    if HAS_LANGCHAIN and _LCRunnableLambda:
        return _LCRunnableLambda(func)
    return LocalRunnableLambda(func)


def runnable_sequence(steps: Iterable[Any]) -> Any:
    """Factory returning a runnable sequence.

    Uses LangChain's implementation when available; otherwise returns a
    lightweight local version.
    """

    if HAS_LANGCHAIN and _LCRunnableSequence:
        return _LCRunnableSequence(steps)
    return LocalRunnableSequence(steps)


__all__ = [
    "HAS_LANGCHAIN",
    "runnable_lambda",
    "runnable_sequence",
    "LocalRunnableLambda",
    "LocalRunnableSequence",
]
