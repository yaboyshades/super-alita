import pytest
from unittest.mock import AsyncMock

from src.core.event_bus import EventBus
from src.core.temporal_graph import TemporalGraph
from src.core.navigation import NeuralNavigator, NavigationConfig
from src.integration.cortex_integration import CortexIntegration


@pytest.mark.asyncio
async def test_integration_phase_advance_event():
    bus = AsyncMock(spec=EventBus)
    g = TemporalGraph()
    nav = NeuralNavigator(graph=g, config=NavigationConfig(enable_telemetry=False))
    integ = CortexIntegration(event_bus=bus, graph=g, navigator=nav)

    await integ.handle_autonomy_update({"data": {"current_score": 0.9}})
    await integ.handle_autonomy_update({"data": {"current_score": 0.9}})
    await integ.handle_autonomy_update({"data": {"current_score": 0.9}})
    assert bus.publish.called
    payloads = [c.args[0] for c in bus.publish.call_args_list]
    types = [getattr(p, "event_type", getattr(p, "type", None)) for p in payloads]
    assert "phase_advanced" in types


@pytest.mark.asyncio
async def test_should_use_cortex_policy():
    bus = AsyncMock(spec=EventBus)
    g = TemporalGraph()
    nav = NeuralNavigator(graph=g, config=NavigationConfig(enable_telemetry=False))
    integ = CortexIntegration(event_bus=bus, graph=g, navigator=nav)
    assert await integ.should_use_cortex(0.2) is True
    assert await integ.should_use_cortex(0.95) is False


@pytest.mark.asyncio
async def test_phase_demotion_event():
    bus = AsyncMock(spec=EventBus)
    g = TemporalGraph()
    nav = NeuralNavigator(graph=g, config=NavigationConfig(enable_telemetry=False))
    integ = CortexIntegration(event_bus=bus, graph=g, navigator=nav)
    await integ.handle_autonomy_update({"data": {"current_score": 0.35}})
    await integ.handle_autonomy_update({"data": {"current_score": 0.36}})
    await integ.handle_autonomy_update({"data": {"current_score": 0.37}})
    await integ.handle_autonomy_update({"data": {"current_score": 0.2}})
    await integ.handle_autonomy_update({"data": {"current_score": 0.2}})
    await integ.handle_autonomy_update({"data": {"current_score": 0.2}})
    assert bus.publish.called
    payloads = [c.args[0] for c in bus.publish.call_args_list]
    types = [getattr(p, "event_type", getattr(p, "type", None)) for p in payloads]
    assert "phase_demoted" in types
