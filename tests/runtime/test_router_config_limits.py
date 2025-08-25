import asyncio
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from reug_runtime import config
from reug_runtime.router import breaker, router

from tests.runtime.fakes import FakeAbilityRegistry, FakeEventBus, FakeKG


class GreedyLLM:
    async def stream_chat(self, messages, timeout):
        yield {"content": '<tool_call>{"tool":"echo","args":{"payload":"hi"}}</tool_call>'}


def _mk_app_max_calls(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    app.state.event_bus = FakeEventBus()
    app.state.ability_registry = FakeAbilityRegistry()
    app.state.kg = FakeKG()
    app.state.llm_model = GreedyLLM()
    monkeypatch.setattr(config.SETTINGS, "max_tool_calls", 1)
    return app


def test_max_tool_calls_abort(monkeypatch):
    app = _mk_app_max_calls(monkeypatch)
    client = TestClient(app)
    resp = client.post("/v1/chat/stream", json={"message": "go", "session_id": "cap"})
    text = resp.text
    assert "[ERROR: Agent unable to complete request]" in text
    evts = app.state.event_bus.events
    calls = [e for e in evts if e["type"] == "AbilityCalled"]
    assert len(calls) == 1
    terminals = [e for e in evts if e["type"] in {"TaskSucceeded", "TaskFailed"}]
    assert len(terminals) == 1 and terminals[0]["type"] == "TaskFailed"


class SlowRegistry:
    def get_available_tools_schema(self):
        return [
            {
                "tool_id": "slow",
                "description": "slow tool",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            }
        ]

    def knows(self, name):
        return name == "slow"

    def validate_args(self, name, args):
        return True

    async def execute(self, name, args):
        await asyncio.sleep(0.05)
        return {"ok": True}


class TimeoutLLM:
    async def stream_chat(self, messages, timeout):
        if any(m["role"] == "assistant" and "<tool_error" in m["content"] for m in messages):
            yield {"content": '<final_answer>{"content":"gave up","citations":[]}</final_answer>'}
        else:
            yield {"content": '<tool_call>{"tool":"slow","args":{}}</tool_call>'}


def _mk_app_timeout(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    app.state.event_bus = FakeEventBus()
    app.state.ability_registry = SlowRegistry()
    app.state.kg = FakeKG()
    app.state.llm_model = TimeoutLLM()
    monkeypatch.setattr(config.SETTINGS, "tool_timeout_s", 0.01)
    monkeypatch.setattr(config.SETTINGS, "max_retries", 0)
    return app


def test_execution_timeout(monkeypatch):
    app = _mk_app_timeout(monkeypatch)
    client = TestClient(app)
    resp = client.post("/v1/chat/stream", json={"message": "go", "session_id": "to"})
    assert "gave up" in resp.text
    evts = app.state.event_bus.events
    fails = [e for e in evts if e["type"] == "AbilityFailed"]
    assert len(fails) == 1
    assert "timeout_0.01s" in fails[0]["error"]


class FailingRegistry:
    def __init__(self):
        self.call_times: list[float] = []

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
        self.call_times.append(time.time())
        raise RuntimeError("boom")


class BackoffLLM:
    async def stream_chat(self, messages, timeout):
        if any(m["role"] == "assistant" and "<tool_error" in m["content"] for m in messages):
            yield {"content": '<final_answer>{"content":"done","citations":[]}</final_answer>'}
        else:
            yield {"content": '<tool_call>{"tool":"flaky","args":{}}</tool_call>'}


def _mk_app_backoff(monkeypatch):
    breaker.failures.clear()
    app = FastAPI()
    app.include_router(router)
    registry = FailingRegistry()
    app.state.event_bus = FakeEventBus()
    app.state.ability_registry = registry
    app.state.kg = FakeKG()
    app.state.llm_model = BackoffLLM()
    monkeypatch.setattr(config.SETTINGS, "max_retries", 1)
    monkeypatch.setattr(config.SETTINGS, "retry_base_ms", 100)
    return app, registry


def test_retry_backoff(monkeypatch):
    app, registry = _mk_app_backoff(monkeypatch)
    client = TestClient(app)
    resp = client.post("/v1/chat/stream", json={"message": "hi", "session_id": "rb"})
    assert "done" in resp.text
    assert len(registry.call_times) == 2
    delta = registry.call_times[1] - registry.call_times[0]
    assert delta >= 0.1
