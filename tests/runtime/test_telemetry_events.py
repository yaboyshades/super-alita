import time
from fastapi import FastAPI
from fastapi.testclient import TestClient
from reug_runtime.router import router

from tests.runtime import prefix_path
from tests.runtime.fakes import FakeAbilityRegistry, FakeEventBus, FakeKG, FakeLLM


class TimedFakeEventBus(FakeEventBus):
    async def emit(self, event: dict) -> None:  # type: ignore[override]
        event["timestamp_ms"] = time.time() * 1000
        await super().emit(event)


def _make_app(model) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.event_bus = TimedFakeEventBus()
    app.state.ability_registry = FakeAbilityRegistry()
    app.state.kg = FakeKG()
    app.state.llm_model = model
    return app


def test_success_turn_emits_required_fields() -> None:
    app = _make_app(FakeLLM())
    client = TestClient(app)
    resp = client.post(
        prefix_path("/v1/chat/stream"), json={"message": "hi", "session_id": "ok"}
    )
    assert resp.status_code == 200
    events = app.state.event_bus.events
    kinds = {e["type"] for e in events}
    assert {"TaskStarted", "AbilityCalled", "AbilitySucceeded", "TaskSucceeded"} <= kinds
    for evt in events:
        assert "correlation_id" in evt
        assert "timestamp_ms" in evt
        if evt["type"].startswith("Ability"):
            assert "span_id" in evt


class NoFinalLLM:
    async def stream_chat(self, messages, timeout):
        if any(m["role"] == "assistant" and "<tool_result" in m["content"] for m in messages):
            yield {"content": "still thinking"}
        else:
            yield {"content": '<tool_call>{"tool":"echo","args":{"payload":"hi"}}</tool_call>'}


def test_failing_turn_emits_required_fields(monkeypatch) -> None:
    app = _make_app(NoFinalLLM())
    from reug_runtime import config

    monkeypatch.setattr(config.SETTINGS, "max_tool_calls", 1)
    client = TestClient(app)
    resp = client.post(
        prefix_path("/v1/chat/stream"), json={"message": "hi", "session_id": "fail"}
    )
    assert resp.status_code == 200
    events = app.state.event_bus.events
    kinds = {e["type"] for e in events}
    assert {"TaskStarted", "AbilityCalled", "AbilitySucceeded", "TaskFailed"} <= kinds
    for evt in events:
        assert "correlation_id" in evt
        assert "timestamp_ms" in evt
        if evt["type"].startswith("Ability"):
            assert "span_id" in evt
