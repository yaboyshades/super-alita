import sys
import types

import pytest

from src.telemetry.mcp_broadcaster import MCPTelemetryBroadcaster, EventTypes


@pytest.mark.asyncio
async def test_broadcaster_uses_telemetry_api(monkeypatch) -> None:
    emitted: list[dict] = []
    telemetry_module = types.SimpleNamespace(emit=lambda evt: emitted.append(evt))
    monkeypatch.setitem(sys.modules, "cortex.telemetry", telemetry_module)

    broadcaster = MCPTelemetryBroadcaster()
    await broadcaster.start()
    await broadcaster.broadcast_event(
        event_type=EventTypes.TOOL_CALL,
        source="unit_test",
        data={"a": 1},
        session_id="sess",
    )
    assert emitted, "telemetry.emit was not called"
    event = emitted[-1]
    assert event["event_type"] == EventTypes.TOOL_CALL
    assert event["source"] == "unit_test"
    assert event["session_id"] == "sess"
