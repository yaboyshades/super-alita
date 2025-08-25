import pytest
from aiohttp import web

import os

from src.core.events import create_event
from src.plugins.puter_plugin import PuterPlugin, PuterOperationAtom
from tests.runtime.puter_fakes import FakePuterServer


class MockEventBus:
    """Minimal event bus capturing published events."""

    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, event_type: str, **kwargs) -> None:
        event = create_event(event_type, **kwargs)
        self.events.append(event)
        return event

    async def publish(self, event) -> None:  # pragma: no cover - compatibility
        self.events.append(event)

    async def subscribe(self, event_type: str, handler) -> None:  # pragma: no cover
        pass


class MockNeuralStore:
    def __init__(self) -> None:
        self.registered_atoms: list = []

    async def register(self, atom) -> None:
        self.registered_atoms.append(atom)


@pytest.fixture
async def mock_event_bus() -> MockEventBus:
    return MockEventBus()


@pytest.fixture
async def mock_neural_store() -> MockNeuralStore:
    return MockNeuralStore()


@pytest.fixture
async def fake_puter_server():
    server = FakePuterServer()
    app = server.create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    base_url = f"http://localhost:{port}"
    try:
        yield base_url
    finally:
        await runner.cleanup()


@pytest.fixture
async def puter_plugin(mock_event_bus, mock_neural_store, fake_puter_server):
    for key in ["PUTER_BASE_URL", "PUTER_API_KEY", "PUTER_WORKSPACE_ID"]:
        os.environ.pop(key, None)
    plugin = PuterPlugin()
    config = {
        "puter_base_url": fake_puter_server,
        "puter_api_key": "test",
        "puter_workspace_id": "ws",
    }
    await plugin.setup(mock_event_bus, mock_neural_store, config)
    await plugin.start()
    try:
        yield plugin
    finally:
        await plugin.shutdown()


class TestPuterOperationAtom:
    def test_deterministic_uuid(self) -> None:
        data = {"operation": "read", "file_path": "/a.txt"}
        atom1 = PuterOperationAtom("file_operation", data)
        atom2 = PuterOperationAtom("file_operation", data)
        assert atom1.get_deterministic_uuid() == atom2.get_deterministic_uuid()


@pytest.mark.integration_puter
@pytest.mark.asyncio
async def test_file_operations_emit_ability_events(puter_plugin, mock_event_bus):
    write_event = create_event(
        "puter_file_operation", source_plugin="test", conversation_id="conv1"
    )
    write_event.metadata = {
        "operation": "write",
        "file_path": "/new.txt",
        "content": "hello",
    }
    await puter_plugin._handle_file_operation(write_event)

    read_event = create_event(
        "puter_file_operation", source_plugin="test", conversation_id="conv1"
    )
    read_event.metadata = {"operation": "read", "file_path": "/new.txt"}
    await puter_plugin._handle_file_operation(read_event)

    called = [e for e in mock_event_bus.events if e.event_type == "AbilityCalled"]
    succeeded = [e for e in mock_event_bus.events if e.event_type == "AbilitySucceeded"]

    assert len(called) == 2
    assert len(succeeded) == 2
    assert len(puter_plugin.operation_history) == 2


@pytest.mark.integration_puter
@pytest.mark.asyncio
async def test_failed_file_operation_emits_ability_failed(puter_plugin, mock_event_bus):
    event = create_event(
        "puter_file_operation", source_plugin="test", conversation_id="conv2"
    )
    event.metadata = {"operation": "read", "file_path": "/missing.txt"}
    await puter_plugin._handle_file_operation(event)

    failed = [e for e in mock_event_bus.events if e.event_type == "AbilityFailed"]
    assert len(failed) == 1
    assert failed[0].tool == "puter_file_operation"
