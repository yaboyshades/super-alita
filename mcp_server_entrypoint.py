"""MCP Server entrypoint for Super Alita Codex runtime.

Bridges the agent's PluginInterface and EventBus into MCP so Codex (and IDEs)
can talk to it.
"""

from __future__ import annotations

import asyncio
import inspect

from src.core.event_bus import EventBus
from src.main import create_app

try:
    from mcp.server import StdIOServer
except Exception:  # pragma: no cover - library not available
    class StdIOServer:  # type: ignore[override]
        """Fallback stub when anthropic/mcp StdIOServer is unavailable."""

        def __init__(self, app, event_bus):
            self.app = app
            self.event_bus = event_bus

        async def run_forever(self) -> None:  # pragma: no cover - stub
            raise RuntimeError("StdIOServer requires the mcp.server package")


async def main() -> None:
    """Start the MCP server using stdio transport."""
    bus = EventBus()
    maybe_app = create_app(event_bus=bus)
    app = await maybe_app if inspect.isawaitable(maybe_app) else maybe_app
    server = StdIOServer(app, event_bus=bus)
    await server.run_forever()


if __name__ == "__main__":  # pragma: no cover - manual start
    asyncio.run(main())
