from __future__ import annotations

import json
import re
from fastapi import FastAPI
from fastapi.testclient import TestClient
from reug_runtime.router import router

from tests.runtime.fakes import FakeAbilityRegistry, FakeEventBus, FakeKG


class TaskLLM:
    async def stream_chat(self, messages, timeout):  # type: ignore[override]
        user_msg = messages[-1]["content"]
        if "2+2" in user_msg:
            answer = "4"
        elif user_msg.strip().startswith("reverse"):
            to_rev = user_msg.split("reverse", 1)[1].strip()
            answer = to_rev[::-1]
        else:
            answer = "unknown"
        payload = json.dumps({"content": answer, "citations": []})
        yield {"content": f"<final_answer>{payload}</final_answer>"}


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.event_bus = FakeEventBus()
    app.state.ability_registry = FakeAbilityRegistry()
    app.state.kg = FakeKG()
    app.state.llm_model = TaskLLM()
    return app


def _extract_final_answer(text: str) -> str:
    match = re.search(r"<final_answer>(.*?)</final_answer>", text)
    assert match, "no final answer found"
    data = json.loads(match.group(1))
    return data["content"]


def test_agent_can_complete_simple_tasks() -> None:
    app = _make_app()
    client = TestClient(app)

    tasks = [
        {"prompt": "what is 2+2?", "grader": lambda ans: int(ans) == 4},
        {
            "prompt": "reverse hello",
            "grader": lambda ans: ans == "olleh",
        },
    ]

    for task in tasks:
        resp = client.post(
            "/v1/chat/stream", json={"message": task["prompt"], "session_id": "s1"}
        )
        assert resp.status_code == 200
        answer = _extract_final_answer(resp.text)
        assert task["grader"](answer)
