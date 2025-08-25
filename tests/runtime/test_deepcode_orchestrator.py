import asyncio
from collections import defaultdict

import pytest

from src.plugins.deepcode_orchestrator_plugin import DeepCodeOrchestratorPlugin
from src.plugins.deepcode_puter_bridge_plugin import DeepCodePuterBridgePlugin
from src.plugins.puter_plugin import PuterPlugin
from src.core.events import create_event


class FakeBus:
    def __init__(self) -> None:
        self._subs: dict[str, list] = defaultdict(list)
        self.events = []

    async def subscribe(self, event_type: str, handler):
        self._subs[event_type].append(handler)

    async def publish(self, event):
        self.events.append(event)
        for handler in self._subs.get(event.event_type, []):
            await handler(event)

    async def emit(self, event_type: str, **kwargs):
        event = create_event(event_type, **kwargs)
        await self.publish(event)
        return event


@pytest.mark.asyncio
async def test_deepcode_diffs_mirrored_to_puter():
    bus = FakeBus()
    dc_plugin = DeepCodeOrchestratorPlugin()
    bridge = DeepCodePuterBridgePlugin()
    puter = PuterPlugin()

    for plugin in (dc_plugin, bridge, puter):
        await plugin.setup(bus, store=None, config={})
        await plugin.start()

    await dc_plugin._on_request(
        {
            "request_id": "req1",
            "task_kind": "text2code",
            "requirements": "test",
            "conversation_id": "c1",
            "correlation_id": "corr1",
        }
    )

    await asyncio.sleep(0.5)

    write_events = [e for e in bus.events if e.event_type == "puter_file_write"]
    expected_paths = {
        "docs/deepcode_result.md",
        "src/new_dir/deepcode_module.py",
        "tests/test_deepcode_result.py",
        "tests/new_dir/test_deepcode_module.py",
        "docs/plan.json",
        "docs/new_dir/summary.md",
    }
    assert {e.file_path for e in write_events} == expected_paths

    for e in write_events:
        await bus.emit(
            "puter_file_operation",
            metadata={"operation": "write", "file_path": e.file_path, "content": e.content},
            conversation_id="c1",
            source_plugin="deepcode_puter_bridge",
        )

    assert len(puter.operation_history) == len(expected_paths)
    hist_paths = {atom.operation_data["file_path"] for atom in puter.operation_history}
    assert hist_paths == expected_paths
