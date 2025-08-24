import json
import httpx
import pytest

from src.telemetry.mcp_broadcaster import MCPTelemetryBroadcaster


@pytest.mark.asyncio
async def test_broadcaster_posts_events(monkeypatch):
    received: list[dict[str, object]] = []
    auth_headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.content.decode()))
        auth_headers.append(request.headers.get("Authorization"))
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        monkeypatch.setenv("MCP_BROADCAST_URL", "https://mcp.test/telemetry")
        monkeypatch.setenv("MCP_BROADCAST_TOKEN", "testtoken")
        broadcaster = MCPTelemetryBroadcaster(http_client=client)
        await broadcaster.start()
        await broadcaster.broadcast_event("tool_call", "unit_test", {"foo": "bar"})
        await broadcaster.stop()

    tool_events = [e for e in received if e["event_type"] == "tool_call"]
    assert tool_events and tool_events[0]["data"] == {"foo": "bar"}
    assert any(h == "Bearer testtoken" for h in auth_headers)
