from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Iterable, Any

# Simple telemetry hook; tests monkeypatch this
TELEMETRY_EVENTS: list[dict[str, Any]] = []


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
=======
from typing import Sequence


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

