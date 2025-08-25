from fastapi import FastAPI
from fastapi.testclient import TestClient
from reug_runtime.router import router

from tests.runtime.fakes import FakeAbilityRegistry, FakeEventBus, FakeKG, FakeLLM


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.event_bus = FakeEventBus()
    app.state.ability_registry = FakeAbilityRegistry()
    app.state.kg = FakeKG()
    app.state.llm_model = FakeLLM()
    return app


def test_streaming_single_turn_smoke() -> None:
    app = _make_app()
    client = TestClient(app)
    resp = client.post("/v1/chat/stream", json={"message": "hello", "session_id": "s1"})
    assert resp.status_code == 200
    text = resp.text
    # sanity: router streamed tokens and produced final answer
    assert "Thinking..." in text
    assert "<final_answer>" in text
    assert "done: hi" in text
    # events emitted as the loop progressed
    evts = app.state.event_bus.events
    kinds = {e["type"] for e in evts}
    assert "TaskStarted" in kinds
    assert "AbilityCalled" in kinds
    assert "AbilitySucceeded" in kinds
    assert "TaskSucceeded" in kinds
