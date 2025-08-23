import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from reug_runtime.router import router
from tests.runtime.fakes import FakeEventBus, FakeKG


class FlakyRegistry:
    def __init__(self) -> None:
        self.calls = 0

    def get_available_tools_schema(self):
        return [
            {
                "tool_id": "slow_echo",
                "description": "Echo after delay",
                "input_schema": {"type": "object", "properties": {"payload": {"type": "string"}}},
                "output_schema": {"type": "object"},
            }
        ]

    def knows(self, name: str) -> bool:
        return name == "slow_echo"

    def validate_args(self, name: str, args: dict) -> bool:
        return name == "slow_echo" and isinstance(args.get("payload"), str)

    async def execute(self, name: str, args: dict):
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(999)
        return {"echo": args["payload"]}


class RetryLLM:
    async def stream_chat(self, messages, timeout):
        if not any(m["role"] == "assistant" and "<tool_result" in m["content"] for m in messages):
            yield {"content": '<tool_call>{"tool":"slow_echo","args":{"payload":"hi"}}</tool_call>'}
        else:
            yield {
                "content": '<final_answer>{"content":"ok after retry","citations":[]}</final_answer>'
            }


def _mk_app(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    app.state.event_bus = FakeEventBus()
    app.state.ability_registry = FlakyRegistry()
    app.state.kg = FakeKG()
    app.state.llm_model = RetryLLM()
    from reug_runtime import config

    monkeypatch.setattr(config.SETTINGS, "tool_timeout_s", 0.01)
    monkeypatch.setattr(config.SETTINGS, "max_retries", 1)
    monkeypatch.setattr(config.SETTINGS, "retry_base_ms", 1)
    return app


def test_timeout_then_retry(monkeypatch):
    app = _mk_app(monkeypatch)
    client = TestClient(app)
    resp = client.post("/v1/chat/stream", json={"message": "go", "session_id": "rt"})
    text = resp.text
    assert "ok after retry" in text
    evts = app.state.event_bus.events
    failures = [e for e in evts if e["type"] == "AbilityFailed"]
    successes = [e for e in evts if e["type"] == "AbilitySucceeded"]
    assert len(failures) == 1
    assert len(successes) == 1
