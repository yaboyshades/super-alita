from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch

from reug_runtime.router import router
import reug_runtime.config as config
from tests.runtime.fakes import FakeAbilityRegistry, FakeEventBus, FakeKG, FakeLLM


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.event_bus = FakeEventBus()
    app.state.ability_registry = FakeAbilityRegistry()
    app.state.kg = FakeKG()
    app.state.llm_model = FakeLLM()
    return app


def test_copilot_context_emits_events(monkeypatch) -> None:
    app = _make_app()
    monkeypatch.setattr(config.SETTINGS, "copilot_context", True)
    with patch("reug_runtime.router.build_copilot_context", return_value="ctx"):
        client = TestClient(app)
        resp = client.post("/v1/chat/stream", json={"message": "hi", "session_id": "s1"})
        assert resp.status_code == 200
    events = app.state.event_bus.events
    kinds = [e["type"] for e in events if e.get("tool") == "build_copilot_context"]
    assert "AbilityCalled" in kinds
    assert "AbilitySucceeded" in kinds
