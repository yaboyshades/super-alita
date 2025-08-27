import asyncio
import logging
from typing import Any, Iterable


logger = logging.getLogger(__name__)


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
