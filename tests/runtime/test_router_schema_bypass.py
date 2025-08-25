from fastapi import FastAPI
from fastapi.testclient import TestClient
from reug_runtime import config
from reug_runtime.router import router

from tests.runtime.fakes import FakeAbilityRegistry, FakeEventBus, FakeKG


class BadArgLLM:
    async def stream_chat(self, messages, timeout):
        if any(m["role"] == "assistant" and "<tool_result" in m["content"] for m in messages):
            yield {
                "content": '<final_answer>{"content":"ok bad arg","citations":[]}</final_answer>'
            }
            return
        yield {"content": '<tool_call>{"tool":"echo","args":{"payload":123}}</tool_call>'}


def _mk_app():
    app = FastAPI()
    app.include_router(router)
    app.state.event_bus = FakeEventBus()
    app.state.ability_registry = FakeAbilityRegistry()
    app.state.kg = FakeKG()
    app.state.llm_model = BadArgLLM()
    return app


def test_schema_bypass(monkeypatch):
    app = _mk_app()
    old_enforce = config.SETTINGS.schema_enforce
    config.SETTINGS.schema_enforce = False
    client = TestClient(app)
    resp = client.post("/v1/chat/stream", json={"message": "hi", "session_id": "sb"})
    text = resp.text
    config.SETTINGS.schema_enforce = old_enforce
    assert "ok bad arg" in text
    evts = app.state.event_bus.events
    assert any(e["type"] == "SchemaBypass" for e in evts)
    terminals = [e for e in evts if e["type"] in {"TaskSucceeded", "TaskFailed"}]
    assert len(terminals) == 1
    assert terminals[0]["type"] == "TaskSucceeded"
    calls = [e for e in evts if e["type"] == "AbilityCalled"]
    succ = {e["span_id"] for e in evts if e["type"] == "AbilitySucceeded"}
    fail = {e["span_id"] for e in evts if e["type"] == "AbilityFailed"}
    assert all((c["span_id"] in succ) ^ (c["span_id"] in fail) for c in calls)
