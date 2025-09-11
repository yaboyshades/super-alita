#!/usr/bin/env python3
"""
EventBus Viewer (MCP tool events)

Subscribes to the 'mcp_tool_event' channel and prints events as they occur.

Usage:
  python tools/eventbus_viewer.py

Notes:
  - For cross‑process viewing, a Redis/Memurai instance must be available.
  - The in‑memory fallback (EVENTBUS_MODE=in_memory) is per‑process and
    will not work across different processes.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from src.core.event_bus import EventBus


async def main() -> None:
    bus = EventBus()
    await bus.initialize()

    async def handler(event: Any) -> None:
        try:
            # Event may be dict-like or a Pydantic model
            if hasattr(event, "model_dump"):
                payload = event.model_dump()
            elif isinstance(event, dict):
                payload = event
            else:
                payload = json.loads(json.dumps(event, default=str))
        except Exception:
            payload = {"raw": str(event)}

        print("\n=== mcp_tool_event ===")
        print(json.dumps(payload, indent=2, default=str))

    # Subscribe to MCP tool events
    await bus.subscribe("mcp_tool_event", handler)  # type: ignore[arg-type]
    print("Listening for 'mcp_tool_event' (Ctrl+C to stop)...")
    try:
        while True:
            await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    asyncio.run(main())

