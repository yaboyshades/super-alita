import pytest

from src.core.events import create_event
from src.plugins.puter_plugin import PuterPlugin


class DummyStore:
    async def register(self, atom):
        pass


class DummyEventBus:
    def __init__(self) -> None:
        self.events = []

    async def emit(self, event_type: str, **kwargs):
        event = create_event(event_type, **kwargs)
        self.events.append(event)
        return event

    async def publish(self, event):
        self.events.append(event)

    async def subscribe(self, event_type: str, handler) -> None:
        return None


@pytest.fixture
async def plugin(tmp_path):
    bus = DummyEventBus()
    store = DummyStore()
    config = {"workspace_root": str(tmp_path)}
    plugin = PuterPlugin()
    await plugin.setup(bus, store, config)
    await plugin.start()
    return plugin, bus, tmp_path


@pytest.mark.asyncio
async def test_file_operation_rejects_path_traversal(plugin):
    plugin_obj, bus, _ = plugin
    event = create_event(
        "puter_file_operation",
        source_plugin="test",
        conversation_id="test",
    )
    event.metadata = {"operation": "read", "file_path": "../secret.txt"}
    await plugin_obj._handle_file_operation(event)
    failures = [
        e for e in bus.events if getattr(e, "event_type", "") == "puter_operation_failed"
    ]
    assert failures
    assert "workspace root" in failures[0].error


@pytest.mark.asyncio
async def test_process_execution_whitelists_commands(plugin):
    plugin_obj, bus, root = plugin
    event = create_event(
        "puter_process_execution",
        source_plugin="test",
        conversation_id="test",
    )
    event.metadata = {"command": "rm", "args": [], "working_dir": str(root)}
    await plugin_obj._handle_process_execution(event)
    failures = [
        e for e in bus.events if getattr(e, "event_type", "") == "puter_operation_failed"
    ]
    assert failures
    assert "not allowed" in failures[0].error
