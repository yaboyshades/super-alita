"""Minimal dynamic tool registry with streaming and telemetry, secured."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable
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

from src.core.secure_executor import get_secure_executor

ToolFunc = Callable[..., Awaitable[Any] | Any]


class ToolRegistry:
    """Registry for dynamically loaded tools with secure code loading.

    Features:
    - Secure code evaluation (no raw exec)
    - Streaming helpers (invoke_stream/astream)
    - Telemetry with execution timing
    """

    def __init__(self, path: str | Path = "~/.alita_tools") -> None:
        self.path = Path(path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self._tools: dict[str, ToolFunc] = {}

    def register(self, name: str, fn: ToolFunc) -> None:
        """Register an async or sync tool function."""
        self._tools[name] = fn

    def register_from_code(self, name: str, code: str) -> None:
        """Persist tool code and register it using the secure executor.

        The code must define a callable named ``name``. We evaluate with a
        restricted environment and attach an audit log.
        """

        module_path = self.path / f"{name}.py"
        module_path.write_text(code)

        executor = get_secure_executor()
        try:
            fn, _audit = executor.execute_with_audit(
                code=code,
                params={},
                user_id="super_alita_mcp",
                context_id=f"register:{name}",
                function_name=name,
            )
        except Exception as e:  # pragma: no cover - safety path
            raise ValueError(f"Failed to register '{name}': {e}") from e

        if not callable(fn):  # pragma: no cover - double safety check
            raise ValueError(f"No callable '{name}' found in provided code")

        self.register(name, fn)  # type: ignore[arg-type]

    async def _timed_call(self, fn: ToolFunc, **kwargs: Any) -> Any:
        start = time.perf_counter()
        success = True
        try:
            result = fn(**kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        except Exception:
            success = False
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            try:
                telemetry.emit(
                    "tool_invocation", duration_ms=duration_ms, success=success
                )
            except Exception:  # pragma: no cover - optional telemetry
                pass

    async def ainvoke(self, name: str, args: dict[str, Any]) -> Any:
        """Asynchronously invoke a registered tool with telemetry."""
        if name not in self._tools:
            raise KeyError(name)
        return await self._timed_call(self._tools[name], **args)

    def invoke(self, name: str, args: dict[str, Any]) -> Any:
        """Synchronously invoke a registered tool (wraps ainvoke)."""
        return asyncio.run(self.ainvoke(name, args))

    async def invoke_stream(
        self, name: str, args: dict[str, Any]
    ) -> AsyncIterator[Any]:
        """Stream results from a registered tool.

        Supports tools that:
        - implement ``astream``
        - return async generators
        - return sync generators
        - return a single result (yield once)
        """

        if name not in self._tools:
            raise KeyError(name)

        tool = self._tools[name]

        if hasattr(tool, "astream") and callable(tool.astream):
            async for chunk in tool.astream(**args):
                yield chunk
            return

        result = tool(**args)

        if asyncio.iscoroutine(result):
            result = await result

        if inspect.isasyncgen(result):
            async for chunk in result:
                yield chunk
            return

        if inspect.isgenerator(result):
            for chunk in result:
                yield chunk
            return

        yield result

    async def astream(self, name: str, args: dict[str, Any]) -> AsyncIterator[Any]:
        """Alias for :meth:`invoke_stream` to match common streaming API."""
        async for chunk in self.invoke_stream(name, args):
            yield chunk

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools."""
        return sorted(self._tools.keys())
