"""LangChain adapter utilities for Super Alita tools."""

from __future__ import annotations

import inspect
from typing import Any


class AlitaToolRunnable:
    """Minimal runnable wrapper around tools.

    The wrapper calls either ``aexecute`` (async) or ``run`` (sync) on the
    provided ``tool`` and gracefully captures any errors so downstream chains
    can handle them without exceptions leaking through.
    """

    def __init__(self, tool: Any) -> None:
        self.tool = tool

    async def ainvoke(
        self, input: dict[str, Any] | None = None, config: Any | None = None
    ) -> Any:
        """Invoke the underlying tool safely.

        Parameters
        ----------
        input:
            Optional dictionary of inputs to pass to the tool.
        config:
            Optional invocation configuration (unused).
        """

        input = input or {}
        try:
            if hasattr(self.tool, "aexecute"):
                return await self.tool.aexecute(**input)
            if hasattr(self.tool, "run"):
                result = self.tool.run(**input)
                if inspect.isawaitable(result):
                    result = await result
                return result
            raise AttributeError("Tool does not implement aexecute or run")
        except Exception as e:  # pragma: no cover - defensive
            return {"error": str(e), "success": False}
