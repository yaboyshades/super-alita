import asyncio
import json
from typing import Any

from reug_runtime.loop import Orchestrator
from reug_runtime.tools.service import ToolCatalogService
from tests.runtime.fakes import FakeEventBus


class SpyRegistry:
    """Registry spy that records ensure/execute ordering."""

    def __init__(self) -> None:
        self.call_log: list[tuple[str, str, dict[str, Any]]] = []
        self.executions: list[tuple[str, dict[str, Any]]] = []

    def knows(self, tool_name: str) -> bool:  # pragma: no cover - exercised indirectly
        return False

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.call_log.append(("execute", tool_name, dict(args)))
        self.executions.append((tool_name, dict(args)))
        return {"ok": True, "args": dict(args)}


class SpyToolService:
    def __init__(self, call_log: list[tuple[str, str, dict[str, Any]]]) -> None:
        self.call_log = call_log
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def ensure_tool_registered(
        self, tool_name: str, tool_args: dict[str, Any], registry: Any
    ) -> bool:
        self.calls.append((tool_name, dict(tool_args)))
        self.call_log.append(("ensure", tool_name, dict(tool_args)))
        return True


def test_acting_step_invokes_catalog_before_execution() -> None:
    event_bus = FakeEventBus()
    registry = SpyRegistry()
    orchestrator = Orchestrator(event_bus, registry, model=None, correlation_id="cid-1")
    spy_service = SpyToolService(registry.call_log)
    orchestrator._tool_service = spy_service  # type: ignore[attr-defined]

    tool_calls = [
        {
            "id": "call-1",
            "function": {
                "name": "echo",
                "arguments": json.dumps({"payload": "hi"}),
            },
        }
    ]

    streamed_events: list[dict[str, Any]] = []

    async def run_acting_step() -> None:
        async for event in orchestrator._acting_step(tool_calls):
            streamed_events.append(event)

    asyncio.run(run_acting_step())

    assert spy_service.calls == [("echo", {"payload": "hi"})]
    assert [entry[0] for entry in registry.call_log] == ["ensure", "execute"]
    assert registry.executions == [("echo", {"payload": "hi"})]

    # Ability telemetry should be preserved and emitted in order
    event_types = [e["type"] for e in streamed_events]
    assert event_types == ["AbilityCalled", "AbilitySucceeded"]
    assert event_bus.events == streamed_events
    span_id = streamed_events[0]["span_id"]
    assert all(event["span_id"] == span_id for event in streamed_events)


class RecordingRegistry:
    def __init__(self) -> None:
        self.register_tool_calls: list[dict[str, Any]] = []
        self.executors: dict[str, Any] = {}

    def knows(self, tool_name: str) -> bool:
        return tool_name in self.executors

    def register_tool(self, *, contract: dict[str, Any], executor: Any) -> None:
        self.register_tool_calls.append(contract)
        self.executors[contract["tool_id"]] = executor


def test_ensure_tool_registered_triggers_auto_registration(tmp_path) -> None:
    service = ToolCatalogService(mcp_box_dir=str(tmp_path))
    registry = RecordingRegistry()

    tool_name = "fetch_url_text"
    args = {"url": "https://example.com"}

    assert service.ensure_tool_registered(tool_name, args, registry) is True

    assert len(registry.register_tool_calls) == 1
    contract = registry.register_tool_calls[0]
    assert contract["tool_id"] == tool_name
    assert contract["input_schema"]["required"] == ["url"]

    executor = registry.executors[tool_name]
    assert asyncio.iscoroutinefunction(executor)

    persisted = tmp_path / f"{tool_name}.json"
    assert persisted.exists()
    persisted_data = json.loads(persisted.read_text(encoding="utf-8"))
    assert persisted_data["tool_id"] == tool_name
    assert "input_schema" in persisted_data

