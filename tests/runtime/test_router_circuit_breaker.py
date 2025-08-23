from fastapi import FastAPI
from fastapi.testclient import TestClient
from reug_runtime import config
from reug_runtime.router import breaker, router

from tests.runtime.fakes import FakeEventBus, FakeKG


class FlakyRegistry:
    def __init__(self):
        self.calls = 0

    def get_available_tools_schema(self):
        return [
            {
                "tool_id": "flaky",
                "description": "always fails",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            }
        ]

    def knows(self, name):
        return name == "flaky"

    def validate_args(self, name, args):
        return True

    async def execute(self, name, args):
        self.calls += 1
        raise RuntimeError("boom")


class LoopLLM:
    async def stream_chat(self, messages, timeout):
        errors = sum(
            1 for m in messages if m["role"] == "assistant" and "<tool_error" in m["content"]
        )
        if errors >= 4:
            yield {
                "content": '<final_answer>{"content":"breaker done","citations":[]}</final_answer>'
            }
            return
        yield {"content": '<tool_call>{"tool":"flaky","args":{}}</tool_call>'}


def _mk_app():
    breaker.failures.clear()
    app = FastAPI()
    app.include_router(router)
    app.state.event_bus = FakeEventBus()
    app.state.ability_registry = FlakyRegistry()
    app.state.kg = FakeKG()
    app.state.llm_model = LoopLLM()
    return app


def test_circuit_breaker(monkeypatch):
    app = _mk_app()
    old_retries = config.SETTINGS.max_retries
    config.SETTINGS.max_retries = 0
    client = TestClient(app)
    resp = client.post("/v1/chat/stream", json={"message": "hi", "session_id": "cb"})
    text = resp.text
    config.SETTINGS.max_retries = old_retries
    assert "breaker done" in text
    evts = app.state.event_bus.events
    assert any(e["type"] == "ToolCircuitOpen" for e in evts)
    calls = [e for e in evts if e["type"] == "AbilityCalled"]
    fails = [e for e in evts if e["type"] == "AbilityFailed"]
    assert len(calls) == 3
    assert len(fails) == 3
    terminals = [e for e in evts if e["type"] in {"TaskSucceeded", "TaskFailed"}]
    assert len(terminals) == 1
    assert terminals[0]["type"] == "TaskSucceeded"
    succ = {e["span_id"] for e in evts if e["type"] == "AbilitySucceeded"}
    fail = {e["span_id"] for e in evts if e["type"] == "AbilityFailed"}
    assert all((c["span_id"] in succ) ^ (c["span_id"] in fail) for c in calls)
