import pytest
import time
from unittest.mock import AsyncMock

from src.core.event_bus import EventBus
from src.core.temporal_graph import TemporalGraph
from src.core.navigation import NeuralNavigator, NavigationConfig
from src.plugins.cortex_adapter_plugin import CortexAdapterPlugin, GitHubCopilotCortex


@pytest.mark.asyncio
async def test_cortex_adapter_learning_flow():
    bus = AsyncMock(spec=EventBus)
    g = TemporalGraph()
    g.create_atom("rate limiter", "topic", {})
    nav = NeuralNavigator(graph=g, config=NavigationConfig(enable_telemetry=False))

    adapter = CortexAdapterPlugin(event_bus=bus, graph=g, navigator=nav)
    adapter.register_cortex("github_copilot", GitHubCopilotCortex())

    await adapter.handle_reasoning_request({"data": {"prompt": "Implement a rate limiter", "context": {"user": "t"}}})
    assert bus.publish.called
    payloads = [c.args[0] for c in bus.publish.call_args_list]
    assert any(getattr(p, "event_type", getattr(p, "type", None)) == "cortex_knowledge_learned" for p in payloads)
    assert len(g.atoms) >= 2
    learned = [p for p in payloads if getattr(p, "event_type", getattr(p, "type", None)) == "cortex_knowledge_learned"][0]
    assert hasattr(learned, "redaction_summary")


@pytest.mark.asyncio
async def test_gap_event_triggers_reasoning_request():
    bus = AsyncMock(spec=EventBus)
    g = TemporalGraph()
    nav = NeuralNavigator(graph=g, config=NavigationConfig(enable_telemetry=False))
    adapter = CortexAdapterPlugin(event_bus=bus, graph=g, navigator=nav)
    adapter.register_cortex("github_copilot", GitHubCopilotCortex())

    await adapter.handle_knowledge_gap({"data": {"gap_description": "missing rate limiter concept"}})
    assert bus.publish.called
    payload = bus.publish.call_args.args[0]
    assert getattr(payload, "event_type", getattr(payload, "type", None)) == "reasoning_request"


@pytest.mark.asyncio
async def test_budget_and_circuit_breaker_paths():
    bus = AsyncMock(spec=EventBus)
    g = TemporalGraph()
    nav = NeuralNavigator(graph=g, config=NavigationConfig(enable_telemetry=False))
    adapter = CortexAdapterPlugin(event_bus=bus, graph=g, navigator=nav)
    adapter._budget_max_per_min = 1
    adapter._circuit.failure_threshold = 1
    await adapter.handle_reasoning_request({"data": {"prompt": "A"}})
    await adapter.handle_reasoning_request({"data": {"prompt": "B"}})
    payloads = [c.args[0] for c in bus.publish.call_args_list]
    assert any(getattr(p, "event_type", getattr(p, "type", None)) == "cortex_budget_exceeded" for p in payloads)

    class BadCortex:
        async def reason(self, p, c):
            raise RuntimeError("boom")

    adapter.cortex_providers = {"bad": BadCortex()}
    adapter._budget_window_start = time.time()
    adapter._budget_calls = 0
    adapter._budget_max_per_min = 100
    await adapter.handle_reasoning_request({"data": {"prompt": "C"}})
    await adapter.handle_reasoning_request({"data": {"prompt": "C"}})
    payloads2 = [c.args[0] for c in bus.publish.call_args_list]
    assert any(getattr(p, "event_type", getattr(p, "type", None)) == "cortex_circuit_open" for p in payloads2)
