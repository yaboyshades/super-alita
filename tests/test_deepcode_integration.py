import asyncio
import pytest
from src.plugins.deepcode_generator_plugin import DeepCodeGeneratorBridgePlugin


class FakeBus:
    async def subscribe(self, event_type: str, handler):
        return None


@pytest.mark.asyncio
async def test_bridge_initialization():
    plugin = DeepCodeGeneratorBridgePlugin()
    await plugin.setup(event_bus=FakeBus(), store=None, config={})
    assert plugin.name == "deepcode_generator"


@pytest.mark.asyncio
async def test_generation_event_bridges_to_deepcode_request(monkeypatch):
    captured = []

    class P(DeepCodeGeneratorBridgePlugin):
        async def emit_event(self, event_type: str, **fields):
            captured.append({"event_type": event_type, **fields})
            return await super().emit_event(event_type, **fields)

    plugin = P()
    await plugin.setup(event_bus=FakeBus(), store=None, config={})
    await plugin.start()
    await plugin._handle_generation({
        "prompt": "Create a FastAPI endpoint",
        "repo_path": ".",
        "conversation_id": "conv1",
    })
    await asyncio.sleep(0.01)
    kinds = [e["event_type"] for e in captured]
    assert "cognitive_turn" in kinds
    assert "deepcode_request" in kinds
    reqs = [e for e in captured if e["event_type"] == "deepcode_request"][0]
    assert reqs["task_kind"] == "text2backend"
    assert reqs["requirements"].startswith("Create a FastAPI")
