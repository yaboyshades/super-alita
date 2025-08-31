"""
MCP Adapter Plugin

Bridges locally-registered MCP tools (via src.mcp_local.registry.ToolRegistry)
into the runtime ability registry. Each discovered tool is exposed with a
generic schema and executed by delegating to the MCP local registry.

This does not connect to a remote MCP server; instead it leverages the local
registry intended for dynamically registered tools. If no tools are present,
the plugin is effectively a no-op.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from src.core.plugin_interface import PluginInterface


class MCPAdapterPlugin(PluginInterface):
    """Adapter that exposes MCP-local tools to the runtime as abilities."""

    def __init__(self) -> None:
        super().__init__()
        self._registry = None

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "mcp_adapter"

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)
        try:
            from src.mcp_local.registry import ToolRegistry  # type: ignore

            self._registry = ToolRegistry()
        except Exception:
            self._registry = None

    async def start(self) -> None:  # pragma: no cover - no background tasks
        await super().start()

    async def shutdown(self) -> None:  # pragma: no cover - trivial
        self._registry = None
        await super().shutdown()

    def get_tools(self) -> list[dict[str, Any]]:
        """Return MCP tool descriptors as generic contracts.

        Each tool contract includes only basic info and an open parameters
        schema (allow any kwargs) since detailed schemas are not available.
        """
        if not self._registry:
            return []
        try:
            names: list[str] = list(self._registry.list_tools())
        except Exception:
            names = []
        tools: list[dict[str, Any]] = []
        for n in names:
            tools.append(
                {
                    "name": n,
                    "description": f"MCP tool '{n}' (local registry)",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                }
            )
        return tools

    def get_tool_executor(
        self, tool_name: str
    ) -> Callable[[dict[str, Any]], Awaitable[Any]]:
        """Return an async executor for a given tool name."""
        if not self._registry:

            async def _missing(_: dict[str, Any]) -> Any:
                raise RuntimeError("MCP registry not available")

            return _missing

        async def _exec(args: dict[str, Any]) -> Any:
            return await self._registry.ainvoke(tool_name, args)

        return _exec
