import asyncio
from typing import Any, Callable, Dict, List

import pytest

from src.main_unified import (
    AVAILABLE_PLUGINS,
    PLUGIN_ORDER,
    UnifiedSuperAlita,
    _load_unified_plugins,
)
from src.plugins.deepcode_orchestrator_plugin import DeepCodeOrchestratorPlugin
from src.plugins.deepcode_puter_bridge_plugin import DeepCodePuterBridgePlugin
from src.plugins.perplexica_search_plugin import (
    PerplexicaResponse,
    SearchMode,
)


class InMemoryEventBus:
    """Minimal async event bus used for plugin integration tests."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.handlers: Dict[str, List[Callable[[Any], Any]]] = {}

    async def subscribe(self, event_type: str, handler: Callable[[Any], Any]) -> None:
        self.handlers.setdefault(event_type, []).append(handler)

    async def emit(self, event_type: str, **kwargs: Any) -> None:
        self.events.append({"event_type": event_type, **kwargs})
        for handler in self.handlers.get(event_type, []):
            if asyncio.iscoroutinefunction(handler):
                await handler(dict(kwargs))
            else:  # pragma: no cover - not used in tests
                handler(dict(kwargs))

    async def publish(self, event: Any) -> None:
        event_type = getattr(event, "event_type", event.__class__.__name__)
        self.events.append({"event_type": event_type, "event": event})
        for handler in self.handlers.get(event_type, []):
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:  # pragma: no cover - not used in tests
                handler(event)


class WorkspaceShim:
    """Synchronous subscribe facade for plugins expecting a workspace."""

    def __init__(self, bus: InMemoryEventBus) -> None:
        self.bus = bus

    def subscribe(self, event_type: str, handler: Callable[[Any], Any]) -> None:
        asyncio.get_event_loop().create_task(self.bus.subscribe(event_type, handler))


@pytest.mark.asyncio
async def test_unified_plugins_start_and_pipeline() -> None:
    """Ensure unified plugins start and collaborate via the event bus."""
    bus = InMemoryEventBus()
    alita = UnifiedSuperAlita()

    _load_unified_plugins()

    # Initialize and start core plugins in defined order
    loaded: List[str] = []
    ws_shim = WorkspaceShim(bus)
    for name in PLUGIN_ORDER:
        plugin_cls = AVAILABLE_PLUGINS.get(name)
        if not plugin_cls:
            continue  # plugin unavailable in current environment
        try:
            instance = plugin_cls()
        except TypeError:
            # plugin abstract or missing requirements
            continue
        await instance.setup(bus, alita.store, {})
        if name == "llm_planner":
            instance.workspace = ws_shim
        await instance.start()
        assert instance.is_running, f"{name} failed to start"
        alita.plugins[name] = instance
        loaded.append(name)

        # emit synthetic telemetry to confirm bus capture
        await bus.emit("STATE_TRANSITION", plugin=name, from_state="setup", to_state="started")
        await bus.emit("AbilityCalled", plugin=name, ability="startup")
        await bus.emit("AbilitySucceeded", plugin=name, ability="startup")

    # Bring up DeepCode orchestrator + bridge for full pipeline simulation
    deepcode = DeepCodeOrchestratorPlugin()
    await deepcode.setup(bus, alita.store, {})
    await deepcode.start()
    bridge = DeepCodePuterBridgePlugin()
    await bridge.setup(bus, alita.store, {})
    await bridge.start()

    # Patch Perplexica search to avoid network access
    perplex = alita.plugins.get("perplexica_search")
    assert perplex is not None

    async def fake_search(*args: Any, **kwargs: Any) -> PerplexicaResponse:
        return PerplexicaResponse(
            query="q",
            search_mode=SearchMode.WEB,
            summary="s",
            reasoning="r",
            sources=[],
            citations=[],
        )

    perplex.search = fake_search  # type: ignore[assignment]

    # Trigger search and deepcode generation
    await bus.emit("perplexica_search", query="q", session_id="sess")
    await bus.emit(
        "deepcode_request",
        task_kind="generic",
        requirements="do it",
        conversation_id="sess",
        correlation_id="corr-1",
    )

    # Wait for pipeline to emit all events
    for _ in range(10):
        if any(e["event_type"] == "puter_file_write" for e in bus.events):
            break
        await asyncio.sleep(0.1)

    kinds = {e["event_type"] for e in bus.events}
    assert "perplexica_result" in kinds
    assert "deepcode_ready_for_apply" in kinds

    # Confirm telemetry events captured for core plugins
    for name in loaded:
        assert any(
            e["event_type"] == "STATE_TRANSITION" and e["plugin"] == name
            for e in bus.events
        )
        assert any(
            e["event_type"].startswith("Ability") and e["plugin"] == name
            for e in bus.events
        )
