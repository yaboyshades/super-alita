from fastapi import FastAPI
from fastapi.testclient import TestClient
from reug_runtime.router import router

from tests.runtime.fakes import FakeEventBus, FakeKG, FakeLLM


class DynamicRegistry:
    """Registry with no predefined tools; supports dynamic registration."""

    def __init__(self) -> None:
        self._known: set[str] = set()
        self._contracts = []
        self._exec_calls = []

    def get_available_tools_schema(self):
        return []

    def knows(self, tool_name: str) -> bool:
        return tool_name in self._known

    def validate_args(self, tool_name: str, args: dict) -> bool:
        return tool_name in self._known

    async def health_check(self, contract: dict) -> bool:
        self._contracts.append(contract)
        return True

    async def register(self, contract: dict):
        self._known.add(contract["tool_id"])

    async def execute(self, tool_name: str, args: dict):
        self._exec_calls.append({"tool": tool_name, "args": args})
        return {"ok": True, "echoed": args}


class DynamicLLM(FakeLLM):
    """Requests a brand new tool."""

    async def stream_chat(self, messages, timeout):
        if any(m["role"] == "assistant" and "<tool_result" in m["content"] for m in messages):
            yield {
                "content": '<final_answer>{"content":"done dynamic","citations":[]}</final_answer>'
            }
            return
        yield {"content": "Thinking... "}
        yield {"content": '<tool_call>{"tool":"brand_new","args":{"x":1}}</tool_call>'}


def _mk_app():
    app = FastAPI()
    app.include_router(router)
    app.state.event_bus = FakeEventBus()
    app.state.ability_registry = DynamicRegistry()
    app.state.kg = FakeKG()
    app.state.llm_model = DynamicLLM()
    return app


def test_dynamic_registration_flow():
    app = _mk_app()
    client = TestClient(app)
    resp = client.post("/v1/chat/stream", json={"message": "hi", "session_id": "dyn"})
    text = resp.text
    assert "done dynamic" in text
    evts = app.state.event_bus.events
    kinds = {e["type"] for e in evts}
    assert {
        "STATE_TRANSITION",
        "TOOL_REGISTERED",
        "AbilityCalled",
        "AbilitySucceeded",
        "TaskSucceeded",
    } <= kinds
