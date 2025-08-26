import pytest

from mcp_server_entrypoint import main


class DummyServer:
    def __init__(self, app, event_bus):
        self.app = app
        self.event_bus = event_bus
        self.ran = False

    async def run_forever(self) -> None:
        self.ran = True


@pytest.mark.asyncio
async def test_entrypoint_runs(monkeypatch):
    captured = {}

    async def fake_create_app(event_bus):
        captured["event_bus"] = event_bus
        return object()

    monkeypatch.setattr("mcp_server_entrypoint.create_app", fake_create_app)

    server = DummyServer(None, None)

    def fake_server(app, event_bus):
        server.app = app
        server.event_bus = event_bus
        return server

    monkeypatch.setattr("mcp_server_entrypoint.StdIOServer", fake_server)

    await main()

    assert server.ran
    assert server.event_bus is captured["event_bus"]
