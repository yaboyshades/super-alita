"""Lightweight FastMCP stub used for testing without the upstream package."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class FastMCP:
    """Minimal decorator-based tool registry."""

    def __init__(self, app_name: str) -> None:
        self.app_name = app_name
        self._tools: dict[str, Callable[..., Any]] = {}

    def tool(self, name: str, description: str | None = None, **metadata: Any):
        """Register an async tool function; returns decorator."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._tools[name] = func
            return func

        return decorator

    def run(self, transport: str = "stdio", **kwargs: Any) -> None:
        # In tests we do not spin up transports; just log intent.
        print(
            f"[FastMCP shim] run called (app={self.app_name}, transport={transport})"
        )


__all__ = ["FastMCP"]
