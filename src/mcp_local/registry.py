"""Minimal dynamic tool registry."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

try:  # pragma: no cover - telemetry optional
    from cortex import telemetry  # type: ignore
except Exception:  # pragma: no cover - fallback when cortex not installed
    class _Telemetry:
        @staticmethod
        def emit(*args: Any, **kwargs: Any) -> None:
            """No-op telemetry emitter."""

    telemetry = _Telemetry()

ToolFunc = Callable[..., Awaitable[Any]]


class ToolRegistry:
    """Registry for dynamically loaded tools."""

    def __init__(self, path: str | Path = "~/.alita_tools") -> None:
        self.path = Path(path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self._tools: dict[str, ToolFunc] = {}

    def register(self, name: str, fn: ToolFunc) -> None:
        """Register an async tool function.

        Args:
            name: Tool name.
            fn: Asynchronous callable implementing the tool.
        """

        self._tools[name] = fn

    def register_from_code(self, name: str, code: str) -> None:
        """Persist tool code and register it.

        The code string must define a coroutine function with the given ``name``.

        Args:
            name: Tool name and function identifier.
            code: Python source implementing the tool.
        """

        module_path = self.path / f"{name}.py"
        module_path.write_text(code)
        namespace: dict[str, Any] = {}
        exec(code, namespace)
        fn = namespace.get(name)
        if not callable(fn):  # pragma: no cover - safety check
            raise ValueError(f"No callable '{name}' found in provided code")
        self.register(name, fn)  # type: ignore[arg-type]

    async def _timed_invoke(self, name: str, args: dict[str, Any]) -> Any:
        """Execute a tool with timing and telemetry."""
        start = time.perf_counter()
        success = True
        try:
            if name not in self._tools:
                success = False
                raise KeyError(name)
            return await self._tools[name](**args)
        except Exception:
            success = False
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            try:
                telemetry.emit(
                    "tool_invocation",
                    tool=name,
                    duration_ms=duration_ms,
                    success=success,
                )
            except Exception:  # pragma: no cover - telemetry failures ignored
                pass

    async def ainvoke(self, name: str, args: dict[str, Any]) -> Any:
        """Asynchronously invoke a registered tool with telemetry."""
        return await self._timed_invoke(name, args)

    def invoke(self, name: str, args: dict[str, Any]) -> Any:
        """Synchronously invoke a registered tool with telemetry."""
        return asyncio.run(self._timed_invoke(name, args))

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools."""

        return sorted(self._tools.keys())
