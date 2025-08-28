import asyncio
import os

import pytest


@pytest.mark.asyncio
async def test_mcp_broadcaster_wraps_event_bus(tmp_path):
    # Enable broadcaster via env
    os.environ["MCP_BROADCAST_ENABLED"] = "1"
    # Also direct file sink to a temp file to exercise emit path safely
    os.environ["MCP_EMIT_FILE"] = str(tmp_path / "telemetry.jsonl")

    # Import here to pick up env vars during app creation
    from src.main import create_app

    app = create_app()

    # Enter lifespan to start broadcaster and wrap emit
    async with app.router.lifespan_context(app):  # type: ignore[attr-defined]
        # Sanity: broadcaster attached
        bc = getattr(app.state, "mcp_broadcaster", None)  # type: ignore
        assert bc is not None

        # Emit an event and ensure it is captured in broadcaster history
        event = {"event_type": "tool_call", "source": "pytest", "payload": {"a": 1}}
        await app.state.event_bus.emit(event)  # type: ignore[attr-defined]

        # Small delay to allow async wrapper to append
        await asyncio.sleep(0.05)

        history = bc.get_event_history()  # type: ignore[assignment]
        assert any(e.get("event_type") == "tool_call" for e in history)

    # Cleanup env to avoid leaking into other tests
    os.environ.pop("MCP_BROADCAST_ENABLED", None)
    os.environ.pop("MCP_EMIT_FILE", None)

