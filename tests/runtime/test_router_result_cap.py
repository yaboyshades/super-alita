from fastapi import FastAPI
from fastapi.testclient import TestClient
from reug_runtime.router import router

from tests.runtime.fakes import FakeEventBus, FakeKG


class BigResultRegistry:
    def get_available_tools_schema(self):
        return [
            {
                "tool_id": "big",
                "description": "big result",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            }
        ]

    def knows(self, name):
        return name == "big"

    def validate_args(self, name, args):
        return True

    async def execute(self, name, args):
        return {"blob": "x" * 300000}


class BigLLM:
    async def stream_chat(self, messages, timeout):
        tr = next(
            (
                m["content"]
                for m in messages
                if m["role"] == "assistant" and "<tool_result" in m["content"]
            ),
            None,
        )
        if tr is not None:
            yield {"content": f'<final_answer>{{"content":"{tr}","citations":[]}}</final_answer>'}
            return
        yield {"content": '<tool_call>{"tool":"big","args":{}}</tool_call>'}


def _mk_app():
    app = FastAPI()
    app.include_router(router)
    app.state.event_bus = FakeEventBus()
    app.state.ability_registry = BigResultRegistry()
    app.state.kg = FakeKG()
    app.state.llm_model = BigLLM()
    return app


def test_result_capping():
    app = _mk_app()
    client = TestClient(app)
    resp = client.post("/v1/chat/stream", json={"message": "hi", "session_id": "big"})
    text = resp.text
    assert "_artifact" in text
    evts = app.state.event_bus.events
    assert any(e["type"] == "ArtifactCreated" for e in evts)
    terminals = [e for e in evts if e["type"] in {"TaskSucceeded", "TaskFailed"}]
    assert len(terminals) == 1
    assert terminals[0]["type"] == "TaskSucceeded"
    calls = [e for e in evts if e["type"] == "AbilityCalled"]
    succ = {e["span_id"] for e in evts if e["type"] == "AbilitySucceeded"}
    fail = {e["span_id"] for e in evts if e["type"] == "AbilityFailed"}
    assert all((c["span_id"] in succ) ^ (c["span_id"] in fail) for c in calls)
