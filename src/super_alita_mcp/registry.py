"""Minimal dynamic tool registry."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

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
        return await self._tools[name](**args)

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools."""

        return sorted(self._tools.keys())
