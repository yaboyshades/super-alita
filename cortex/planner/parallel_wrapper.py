from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any

# Simple telemetry hook; tests monkeypatch this
TELEMETRY_EVENTS: list[dict[str, Any]] = []

logger = logging.getLogger(__name__)

try:
    from langchain_core.runnables import RunnableParallel, RunnableSequence

    HAS_LANGCHAIN = True
except Exception:  # pragma: no cover - fallback when LangChain missing
    HAS_LANGCHAIN = False

    class RunnableParallel:
        """Simplistic local fallback for LangChain's RunnableParallel."""

        def __init__(self, steps: dict[str, Any]):
            self.steps = steps

        def invoke(self, input: Any, **kwargs: Any) -> dict[str, Any]:
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


def should_parallelize_runnables(runnables: dict[str, Any]) -> bool:
    """Determine whether to parallelize a set of runnables."""
    if not HAS_LANGCHAIN:
        return False
    return len(runnables) > 1


def parallel_wrapper(runnables: dict[str, Any]) -> RunnableParallel | RunnableSequence:
    """Return a parallel or sequential wrapper depending on availability."""
    if should_parallelize_runnables(runnables):
        return RunnableParallel(runnables)
    return RunnableSequence(runnables.values())


class ParallelWrapper:
    """Simple wrapper to normalize parallel planner results."""

    def __init__(self, planner: Any) -> None:
        self._planner = planner

    def _process_parallel_results(
        self, parallel_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert result dict to {"steps": ..., "parallel": True} shape."""
        steps = parallel_results.get("steps")
        if steps is None:
            # Assume dict mapping step identifiers -> step payloads
            steps = (
                list(parallel_results.values())
                if isinstance(parallel_results, dict)
                else parallel_results
            )
        # Ensure steps is a list
        if not isinstance(steps, list):
            steps = [steps]
        return {"steps": steps, "parallel": True}

    async def ainvoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Invoke underlying planner asynchronously and normalize output."""
        result = await self._planner.ainvoke(*args, **kwargs)
        return self._process_parallel_results(result)

    async def decide_and_run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Execute decide-and-run on underlying planner and normalize output."""
        result = await self._planner.decide_and_run(*args, **kwargs)
        return self._process_parallel_results(result)


class ParallelLadderWrapper:
    """Execute multiple runnables in parallel and aggregate their results."""

    def __init__(self, steps: Iterable[tuple[str, Any]]):
        self.steps = list(steps)

    async def ainvoke(self, input: Any) -> dict[str, Any]:
        async def run_one(name: str, runnable: Any) -> tuple[str, dict[str, Any]]:
            try:
                result = await runnable.ainvoke(input)
                payload = {"result": result, "success": True}
            except Exception as e:
                logger.exception("Parallel step %s failed", name, exc_info=e)
                payload = {"error": str(e), "success": False}
            return name, payload

        tasks = [asyncio.create_task(run_one(name, r)) for name, r in self.steps]
        results = await asyncio.gather(*tasks)
        return {name: payload for name, payload in results}


def _emit_telemetry(mode: str, steps: int, duration: float, success: bool) -> None:
    """Internal helper to emit telemetry events."""
    TELEMETRY_EVENTS.append(
        {
            "mode": mode,
            "steps": steps,
            "duration": duration,
            "success": success,
        }
    )


async def decide_and_run(
    tasks: Iterable[Callable[[], Awaitable[Any]]], *, parallel: bool = True
) -> None:
    """Execute tasks either in parallel or sequentially.

    Telemetry is emitted indicating execution mode, step count, duration,
    and whether execution succeeded.
    """
    functions = list(tasks)
    mode = "parallel" if parallel else "sequential"
    start = time.perf_counter()
    success = False
    try:
        if parallel:
            await asyncio.gather(*(fn() for fn in functions))
        else:
            for fn in functions:
                await fn()
    except Exception:
        success = False
        raise
    else:
        success = True
    finally:
        duration = time.perf_counter() - start
        _emit_telemetry(mode, len(functions), duration, success)


def _have_shared_dependencies(substeps: Sequence[object]) -> bool:
    """Return True if any substeps share a tool dependency."""
    seen: set[str] = set()
    for step in substeps:
        tool = getattr(step, "tool", None) or getattr(step, "tool_hint", None)
        if tool is None:
            continue
        if tool in seen:
            return True
        seen.add(tool)
    return False


def _estimate_total_time(substeps: Sequence[object]) -> float:
    """Estimate the total sequential execution time for substeps."""
    return float(sum(getattr(step, "estimated_time", 0.0) or 0.0 for step in substeps))


def _estimate_parallel_time(substeps: Sequence[object]) -> float:
    """Estimate execution time if substeps run in parallel."""
    times = [getattr(step, "estimated_time", 0.0) or 0.0 for step in substeps]
    return float(max(times) if times else 0.0)


def should_parallelize(
    substeps: Sequence[object],
    *,
    parallel_threshold: int = 2,
    min_parallel_benefit: float = 0.0,
) -> bool:
    """Determine if substeps should execute in parallel.

    Parallelization is permitted only when all conditions are met:
    - Number of substeps exceeds ``parallel_threshold``
    - Substeps do not share tool dependencies
    - Expected time savings exceed ``min_parallel_benefit``
    """

    if len(substeps) <= parallel_threshold:
        return False

    if _have_shared_dependencies(substeps):
        return False

    sequential_time = _estimate_total_time(substeps)
    parallel_time = _estimate_parallel_time(substeps)
    benefit = sequential_time - parallel_time
    return benefit > min_parallel_benefit
