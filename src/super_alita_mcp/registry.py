"""Minimal dynamic tool registry."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
import asyncio
import inspect
from pathlib import Path
from typing import Any

ToolFunc = Callable[..., Awaitable[Any] | Any]


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

    async def invoke(self, name: str, args: dict[str, Any]) -> Any:
        """Invoke a registered tool.

        Args:
            name: Registered tool name.
            args: Argument dictionary passed to the tool.

        Returns:
            Any: Result from the tool.

        Raises:
            KeyError: If the tool is not registered.
        """

        if name not in self._tools:
            raise KeyError(name)
        result = self._tools[name](**args)
        if asyncio.iscoroutine(result):
            return await result
        if inspect.isasyncgen(result):
            return "".join([chunk async for chunk in result])
        if inspect.isgenerator(result):
            return "".join(list(result))
        return result

    async def invoke_stream(
        self, name: str, args: dict[str, Any]
    ) -> AsyncIterator[Any]:
        """Stream results from a registered tool.

        If the tool defines an ``astream`` method, it will be used to yield
        chunks. Otherwise the standard ``invoke`` result is yielded as a
        single chunk. Tools that directly return an async or sync generator
        are also supported.
        """

        if name not in self._tools:
            raise KeyError(name)

        tool = self._tools[name]

        if hasattr(tool, "astream") and callable(getattr(tool, "astream")):
            async for chunk in tool.astream(**args):
                yield chunk
            return

        result = tool(**args)

        if inspect.isasyncgen(result):
            async for chunk in result:
                yield chunk
            return

        if inspect.isgenerator(result):
            for chunk in result:
                yield chunk
            return

        if asyncio.iscoroutine(result):
            result = await result
        yield result

    async def astream(self, name: str, args: dict[str, Any]) -> AsyncIterator[Any]:
        """Alias for :meth:`invoke_stream` to match common streaming API."""

        async for chunk in self.invoke_stream(name, args):
            yield chunk

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools."""

        return sorted(self._tools.keys())
