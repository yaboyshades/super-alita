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
