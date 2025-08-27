"""LangChain adapter utilities.

This module exposes :class:`AlitaToolRunnable`, a lightweight wrapper that allows
LangChain to call Super Alita tools.

Input format
------------
The ``ainvoke`` method expects a dictionary mapping argument names to their
values. For simple string input, supply ``{"input": "<text>"}``. Non-dict
inputs raise :class:`TypeError`.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional


class AlitaToolRunnable:
    """Async runnable wrapper for Alita tools.

    Parameters
    ----------
    tool:
        Awaitable callable executed when invoking the runnable. The callable
        must accept a single ``dict`` of arguments.
    """

    def __init__(self, tool: Callable[[dict[str, Any]], Awaitable[Any]]):
        self._tool = tool

    async def ainvoke(
        self, input: Any, config: Optional[dict[str, Any]] = None
    ) -> Any:
        """Invoke the underlying tool asynchronously.

        Parameters
        ----------
        input:
            Dictionary of arguments for the tool. If the input is not a dict a
            :class:`TypeError` is raised.
        config:
            Optional configuration passed through to the tool. Currently
            unused.
        """

        if not isinstance(input, dict):
            raise TypeError("AlitaToolRunnable expects dict input")

        return await self._tool(input)
