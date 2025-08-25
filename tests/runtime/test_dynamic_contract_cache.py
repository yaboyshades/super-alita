from fastapi import FastAPI
from fastapi.testclient import TestClient
from reug_runtime.router import router

from tests.runtime import prefix_path
from tests.runtime.fakes import FakeEventBus, FakeKG, FakeLLM


class CachingRegistry:
    """Registry that records contracts and executions."""

    def __init__(self) -> None:
        self._known: set[str] = set()
        self.contracts: list[dict] = []
        self.exec_calls: list[dict] = []
        self.health_checks = 0
        self.registers = 0

    def get_available_tools_schema(self):
        return []

    def knows(self, tool_name: str) -> bool:
        return tool_name in self._known

    def validate_args(self, tool_name: str, args: dict) -> bool:
        return tool_name in self._known

    async def health_check(self, contract: dict) -> bool:
        self.contracts.append(contract)
        self.health_checks += 1
        return True

    async def register(self, contract: dict):
        self._known.add(contract["tool_id"])
        self.registers += 1

    async def execute(self, tool_name: str, args: dict):
        self.exec_calls.append({"tool": tool_name, "args": dict(args)})
        return {"ok": True, "echoed": args}


class TwoCallLLM(FakeLLM):
    """Calls the same unknown tool twice before finishing."""

    async def stream_chat(self, messages, timeout):
        tool_results = [
            m
            for m in messages
            if m["role"] == "assistant" and "<tool_result" in m["content"]
        ]
        if len(tool_results) == 0:
            yield {"content": "step1 "}
            yield {
                "content": '<tool_call>{"tool":"brand_new","args":{"x":1}}</tool_call>'
            }
        elif len(tool_results) == 1:
            yield {"content": "step2 "}
            yield {
                "content": '<tool_call>{"tool":"brand_new","args":{"x":2}}</tool_call>'
            }
        else:
            yield {
                "content": '<final_answer>{"content":"done","citations":[]}</final_answer>'
            }


def _mk_app():
    app = FastAPI()
    app.include_router(router)
    app.state.event_bus = FakeEventBus()
    app.state.ability_registry = CachingRegistry()
    app.state.kg = FakeKG()
    app.state.llm_model = TwoCallLLM()
    return app


def test_contract_cached_and_reused():
    app = _mk_app()
    client = TestClient(app)
    resp = client.post(
        prefix_path("/v1/chat/stream"), json={"message": "hi", "session_id": "cache"}
    )
    text = resp.text
    assert "done" in text

    reg = app.state.ability_registry
    # Contract synthesized only once
    assert reg.health_checks == 1
    assert reg.registers == 1
    assert len(reg.contracts) == 1

    expected_contract = {
        "tool_id": "brand_new",
        "version": "0.0.1",
        "description": "Synthesized tool for brand_new",
        "input_schema": {
            "type": "object",
            "properties": {"x": {"type": "number"}},
            "additionalProperties": True,
        },
        "output_schema": {"type": "object", "additionalProperties": True},
        "binding": {"type": "mcp_or_sdk", "endpoint": "brand_new"},
        "guard": {"pii_allowed": False},
    }
    assert reg.contracts[0] == expected_contract

    # Tool executed twice with different args
    assert reg.exec_calls == [
        {"tool": "brand_new", "args": {"x": 1}},
        {"tool": "brand_new", "args": {"x": 2}},
    ]
