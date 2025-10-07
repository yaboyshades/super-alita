from __future__ import annotations

import asyncio
import json
from pathlib import Path
import urllib.request

import pytest

from src.reug_runtime.router import Orchestrator, ToolContractError


class _FakeEventBus:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def emit(self, event: dict[str, object]) -> None:
        self.events.append(event)


class _FakeRegistry:
    def __init__(self, known: set[str] | None = None) -> None:
        self._known = known or set()
        self.registered: list[tuple[dict[str, object], object]] = []
        self.executed: list[tuple[str, dict[str, object]]] = []

    def knows(self, name: str) -> bool:
        return name in self._known

    def register_tool(self, contract: dict[str, object], executor: object) -> None:
        self.registered.append((contract, executor))

    async def execute(self, name: str, args: dict[str, object]) -> dict[str, object]:
        self.executed.append((name, args))
        return {"ok": True}


def _make_orchestrator(tmp_path: Path, registry: _FakeRegistry) -> Orchestrator:
    event_bus = _FakeEventBus()

    class _DummyModel:
        async def stream_chat(self, *args, **kwargs):  # pragma: no cover - unused
            yield {}

    return Orchestrator(event_bus, registry, _DummyModel(), "corr-123")


def test_alias_resolution_uses_canonical_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_BOX_DIR", str(tmp_path))
    index = {"aliases": {"fetch_github_raw": ["GitHubProxy"]}}
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")

    registry = _FakeRegistry(known={"fetch_github_raw"})
    orchestrator = _make_orchestrator(tmp_path, registry)

    ensured = asyncio.run(
        orchestrator._ensure_tool("GitHubProxy", {"owner": "o", "repo": "r", "path": "p"})
    )

    assert ensured is True
    assert registry.registered == []


def test_validate_json_schema_guards(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MCP_BOX_DIR", str(tmp_path))
    orchestrator = _make_orchestrator(tmp_path, _FakeRegistry())

    with pytest.raises(ToolContractError):
        orchestrator._validate_json_schema({"type": "array"}, role="input")

    with pytest.raises(ToolContractError):
        orchestrator._validate_json_schema(
            {"type": "object", "required": ["missing"], "properties": {}}, role="output"
        )


def test_url_fetcher_enforces_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_BOX_DIR", str(tmp_path))
    monkeypatch.setenv("REUG_URL_FETCH_MAX_BYTES", "8")
    monkeypatch.setenv("REUG_URL_FETCH_MAX_CHARS", "5")
    monkeypatch.setenv("REUG_URL_FETCH_TIMEOUT_S", "2.0")

    registry = _FakeRegistry()
    orchestrator = _make_orchestrator(tmp_path, registry)

    captured_timeout: dict[str, float] = {}

    class _FakeResponse:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def read(self, size: int | None = None) -> bytes:
            return self._data

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def _fake_urlopen(url: str, timeout: float) -> _FakeResponse:  # type: ignore[override]
        captured_timeout["value"] = timeout
        return _FakeResponse(b"abcdefghi")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    ensured = asyncio.run(
        orchestrator._ensure_tool("url_fetch_tool", {"url": "http://example.com"})
    )
    assert ensured is True
    assert registry.registered, "expected url tool to be registered"

    contract, executor = registry.registered[-1]
    result = asyncio.run(executor({"url": "http://example.com", "truncate": 20}))

    assert result["truncated"] is True
    assert len(result["content"]) == 5
    assert pytest.approx(captured_timeout["value"], rel=0.0, abs=0.001) == 2.0

    spec_path = tmp_path / "url_fetch_tool.json"
    persisted = json.loads(spec_path.read_text(encoding="utf-8"))
    assert persisted["metadata"]["rate_limit"]["burst"] >= 0
    assert persisted["metadata"]["transport"]["timeout_s"] <= 2.0


def test_fallback_contract_failure_surfaces_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_BOX_DIR", str(tmp_path))
    registry = _FakeRegistry()
    event_bus = _FakeEventBus()

    class _DummyModel:
        async def stream_chat(self, *args, **kwargs):  # pragma: no cover - unused
            yield {}

    orchestrator = Orchestrator(event_bus, registry, _DummyModel(), "corr-err")

    def _raise_fallback(name: str, args: dict[str, object]):
        raise ToolContractError(f"unable to create fallback for {name}")

    monkeypatch.setattr(orchestrator, "_build_fallback_contract", _raise_fallback)

    tool_call = {
        "id": "tc1",
        "function": {"name": "plan_helper", "arguments": "{}"},
    }

    def _run() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        async def _inner() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
            events: list[dict[str, object]] = []
            async for event in orchestrator._acting_step([tool_call]):
                events.append(event)
            return events, event_bus.events

        return asyncio.run(_inner())

    events, emitted = _run()

    ability_events = [ev for ev in events if ev.get("type") == "AbilityFailed"]
    assert ability_events, "expected contract failure event"
    failure = ability_events[0]
    assert failure.get("stage") == "contract"
    assert "unable to create fallback" in str(failure.get("error"))
    assert any(ev.get("type") == "AbilityFailed" for ev in emitted)
