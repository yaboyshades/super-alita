"""LangChain tool adapter."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any


class LangChainAdapter:
    """Async wrapper for LangChain tools.

    Executes ``tool.run`` in a thread and awaits the result with a timeout.
    Handles both synchronous return values and coroutines uniformly.
    """

    def __init__(self, tool: Any, timeout: float | None = None) -> None:
        self.tool = tool
        self.timeout = timeout or 10.0

    async def run(self, **args: Any) -> Any:
        """Execute the underlying tool with a timeout.

        ``tool.run`` is executed in a background thread via ``asyncio.to_thread``.
        The resulting coroutine is then awaited with ``asyncio.wait_for``.  If
        ``tool.run`` itself returns a coroutine, it will be awaited to ensure a
        uniform synchronous result.
        """

        coro = asyncio.to_thread(self.tool.run, **args)
        result = await asyncio.wait_for(coro, timeout=self.timeout)

        if inspect.isawaitable(result):
            result = await result
        return result
