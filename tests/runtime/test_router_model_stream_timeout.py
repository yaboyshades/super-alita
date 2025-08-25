import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient
from reug_runtime.router import router

from tests.runtime import prefix_path
from tests.runtime.fakes import FakeAbilityRegistry, FakeEventBus, FakeKG


class TimeoutLLM:
    """Model stub that immediately raises a timeout."""

    async def stream_chat(self, messages, timeout):
        await asyncio.sleep(0)
        raise TimeoutError("model stalled")
        yield {"content": "never"}


def _mk_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.event_bus = FakeEventBus()
    app.state.ability_registry = FakeAbilityRegistry()
    app.state.kg = FakeKG()
    app.state.llm_model = TimeoutLLM()
    return app


def test_model_stream_timeout() -> None:
    app = _mk_app()
    client = TestClient(app)
    resp = client.post(
        prefix_path("/v1/chat/stream"), json={"message": "hi", "session_id": "mst"}
    )
    assert resp.status_code == 200
    assert "[ERROR: model_stream_timeout]" in resp.text
    events = app.state.event_bus.events
    assert any(
        e["type"] == "TaskFailed" and e.get("reason") == "tool_cap_or_abort" for e in events
    )
    assert not any(e["type"] == "TaskSucceeded" for e in events)
