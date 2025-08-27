import pytest

from src.mcp_local import registry as registry_module

import pytest

from src.mcp_local import registry as registry_module


class TelemetryRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, event: str, **data: object) -> None:
        self.events.append((event, data))


@pytest.mark.asyncio
async def test_ainvoke_emits_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = registry_module.ToolRegistry()

    async def plus_one(x: int) -> int:
        return x + 1

    registry.register("plus_one", plus_one)
    recorder = TelemetryRecorder()
    monkeypatch.setattr(registry_module.telemetry, "emit", recorder.emit)

    result = await registry.ainvoke("plus_one", {"x": 1})
    assert result == 2

    assert recorder.events
    name, data = recorder.events[0]
    assert name == "tool_invocation"
    assert data["tool"] == "plus_one"
    assert data["success"] is True
    assert isinstance(data["duration_ms"], float)


@pytest.mark.asyncio
async def test_ainvoke_failure_emits_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = registry_module.ToolRegistry()
    recorder = TelemetryRecorder()
    monkeypatch.setattr(registry_module.telemetry, "emit", recorder.emit)

    with pytest.raises(KeyError):
        await registry.ainvoke("missing", {})

    assert recorder.events
    name, data = recorder.events[0]
    assert name == "tool_invocation"
    assert data["tool"] == "missing"
    assert data["success"] is False


def test_invoke_sync_emits_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = registry_module.ToolRegistry()

    async def foo() -> str:
        return "ok"

    registry.register("foo", foo)
    recorder = TelemetryRecorder()
    monkeypatch.setattr(registry_module.telemetry, "emit", recorder.emit)

    result = registry.invoke("foo", {})
    assert result == "ok"

    assert recorder.events
    name, data = recorder.events[0]
    assert name == "tool_invocation"
    assert data["tool"] == "foo"
    assert data["success"] is True
