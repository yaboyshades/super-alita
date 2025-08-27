from __future__ import annotations

from typing import Any


class ParallelWrapper:
    """Simple wrapper to normalize parallel planner results."""

    def __init__(self, planner: Any) -> None:
        self._planner = planner

    def _process_parallel_results(self, parallel_results: dict[str, Any]) -> dict[str, Any]:
        """Convert result dict to {"steps": ..., "parallel": True} shape."""
        steps = parallel_results.get("steps")
        if steps is None:
            # Assume dict mapping step identifiers -> step payloads
            steps = list(parallel_results.values()) if isinstance(parallel_results, dict) else parallel_results
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
