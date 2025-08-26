from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.plugins.autonomy_tracker import AutonomyTracker
from src.plugins.knowledge_gap_detector import KnowledgeGapDetector
from src.plugins.cortex_adapter_plugin import CortexAdapterPlugin


class DummyEventBus:
    def __init__(self) -> None:
        self.published: list[Any] = []
        self.subscriptions: list[tuple[str, Any]] = []

    async def publish(self, event: Any) -> None:  # pragma: no cover - simple stub
        self.published.append(event)

    async def subscribe(self, event_type: str, handler: Any) -> None:
        self.subscriptions.append((event_type, handler))


@pytest.mark.asyncio
async def test_autonomy_tracker_setup_sets_event_bus() -> None:
    bus = DummyEventBus()
    plugin = AutonomyTracker()
    await plugin.setup(bus, store={}, config={})
    assert plugin.event_bus is bus


@pytest.mark.asyncio
async def test_knowledge_gap_detector_setup_subscribes() -> None:
    bus = DummyEventBus()
    plugin = KnowledgeGapDetector()
    await plugin.setup(bus, store={}, config={})
    assert plugin.event_bus is bus
    assert {e for e, _ in bus.subscriptions} == {
        "reasoning_complete",
        "navigation_complete",
        "conversation_turn",
    }


@pytest.mark.asyncio
async def test_cortex_adapter_plugin_setup_subscribes_and_sets_deps() -> None:
    bus = DummyEventBus()
    graph = object()
    navigator = object()
    plugin = CortexAdapterPlugin()
    await plugin.setup(bus, store={}, config={"graph": graph, "navigator": navigator})
    assert plugin.event_bus is bus
    assert plugin.graph is graph
    assert plugin.navigator is navigator
    assert {e for e, _ in bus.subscriptions} == {"reasoning_request", "knowledge_gap"}


@pytest.mark.asyncio
async def test_autonomy_tracker_emits_oak_prefixed_event() -> None:
    bus = DummyEventBus()
    plugin = AutonomyTracker()
    await plugin.setup(bus, store={}, config={})
    plugin.emit_event = AsyncMock()
    await plugin.record_metrics(tasks_completed=5, assistance_requests=1)
    plugin.emit_event.assert_called_once()
    args, _ = plugin.emit_event.call_args
    assert args[0] == "oak.autonomy_update"


@pytest.mark.asyncio
async def test_knowledge_gap_detector_emits_oak_prefixed_event() -> None:
    bus = DummyEventBus()
    plugin = KnowledgeGapDetector()
    await plugin.setup(bus, store={}, config={})
    plugin.emit_event = AsyncMock()
    await plugin._maybe_publish_gap(gap_description="desc", context={}, gap_type="test")
    plugin.emit_event.assert_called_once()
    args, _ = plugin.emit_event.call_args
    assert args[0] == "oak.knowledge_gap"
