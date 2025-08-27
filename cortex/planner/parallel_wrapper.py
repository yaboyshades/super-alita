from __future__ import annotations

from typing import Any, Dict, Iterable

try:
    from langchain_core.runnables import RunnableParallel, RunnableSequence

    HAS_LANGCHAIN = True
except Exception:  # pragma: no cover - fallback when LangChain missing
    HAS_LANGCHAIN = False

    class RunnableParallel:
        """Simplistic local fallback for LangChain's RunnableParallel."""

        def __init__(self, steps: Dict[str, Any]):
            self.steps = steps

        def invoke(self, input: Any, **kwargs: Any) -> Dict[str, Any]:
            return {
                name: runnable.invoke(input, **kwargs)
                for name, runnable in self.steps.items()
            }

    class RunnableSequence:
        """Simplistic local fallback for LangChain's RunnableSequence."""

        def __init__(self, steps: Iterable[Any]):
            self.steps = list(steps)

        def invoke(self, input: Any, **kwargs: Any) -> Any:
            result = input
            for step in self.steps:
                result = step.invoke(result, **kwargs)
            return result


def should_parallelize(runnables: Dict[str, Any]) -> bool:
    """Determine whether to parallelize a set of runnables."""
    if not HAS_LANGCHAIN:
        return False
    return len(runnables) > 1


def parallel_wrapper(runnables: Dict[str, Any]) -> RunnableParallel | RunnableSequence:
    """Return a parallel or sequential wrapper depending on availability."""
    if should_parallelize(runnables):
        return RunnableParallel(runnables)
    return RunnableSequence(runnables.values())
