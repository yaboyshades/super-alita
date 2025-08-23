import pytest
from unittest.mock import AsyncMock

from src.core.event_bus import EventBus
from src.core.temporal_graph import TemporalGraph
from src.core.navigation import NeuralNavigator, NavigationConfig
from src.plugins.cortex_adapter_plugin import CortexAdapterPlugin, GitHubCopilotCortex
from src.integration.cortex_integration import CortexIntegration


@pytest.mark.asyncio
async def test_cortex_adapter_reasoning_flow():
    bus = AsyncMock(spec=EventBus)
    g = TemporalGraph()
    g.create_atom("rate limiter", "topic", {})
    nav = NeuralNavigator(graph=g, config=NavigationConfig(enable_telemetry=False))

    adapter = CortexAdapterPlugin(event_bus=bus, graph=g, navigator=nav)
    adapter.register_cortex("github_copilot", GitHubCopilotCortex())

    await adapter.handle_reasoning_request({"data": {"prompt": "Implement a rate limiter", "context": {"user": "t"}}})
    # event emitted
    assert bus.publish.called
    payloads = [c.args[0] for c in bus.publish.call_args_list]
    assert any(getattr(p, "event_type", getattr(p, "type", None)) == "cortex_knowledge_learned" for p in payloads)
    # graph grew
    assert len(g.atoms) >= 2


@pytest.mark.asyncio
async def test_integration_phase_advance_event():
    bus = AsyncMock(spec=EventBus)
    g = TemporalGraph()
    nav = NeuralNavigator(graph=g, config=NavigationConfig(enable_telemetry=False))
    integ = CortexIntegration(event_bus=bus, graph=g, navigator=nav)

    # Trigger autonomy_update with solid score to jump phase
    await integ.handle_autonomy_update({"data": {"current_score": 0.9}})
    assert bus.publish.called
    payloads = [c.args[0] for c in bus.publish.call_args_list]
    types = [getattr(p, "event_type", getattr(p, "type", None)) for p in payloads]
    assert "phase_advanced" in types
