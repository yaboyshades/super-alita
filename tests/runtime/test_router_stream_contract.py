import json
from collections.abc import Iterable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reug_runtime.router import router
from tests.runtime import prefix_path
from tests.runtime.fakes import FakeAbilityRegistry, FakeEventBus, FakeKG, FakeLLM


def _make_router_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.event_bus = FakeEventBus()
    app.state.ability_registry = FakeAbilityRegistry()
    app.state.kg = FakeKG()
    app.state.llm_model = FakeLLM()
    return app


def _read_stream_text(chunks: Iterable[str]) -> str:
    return "".join(chunks)


@pytest.mark.parametrize("path", ["/v1/chat/stream"])
def test_chat_stream_emits_tool_frames_in_order_and_alignment(path: str) -> None:
    app = _make_router_app()
    client = TestClient(app)

    with client.stream(
        "POST",
        prefix_path(path),
        json={"message": "integration hello", "session_id": "telemetry-session"},
    ) as response:
        assert response.status_code == 200
        payload = _read_stream_text(response.iter_text())

    frames = [frame for frame in payload.split("\n\n") if frame.strip()]

    cumulative_content = ""
    tool_call_index = None
    tool_result_index = None
    final_answer_index = None
    alignment_frame_index = None

    done_events: list[dict[str, object]] = []

    for idx, frame in enumerate(frames):
        lines = frame.splitlines()
        event_name = next((line.split(": ", 1)[1] for line in lines if line.startswith("event: ")), "")
        data_line = next((line for line in lines if line.startswith("data: ")), "")
        payload_obj: object
        if data_line:
            raw_payload = data_line.split(": ", 1)[1]
            try:
                payload_obj = json.loads(raw_payload)
            except json.JSONDecodeError:
                payload_obj = raw_payload
        else:
            payload_obj = {}

        if event_name == "content" and isinstance(payload_obj, dict):
            piece = payload_obj.get("content", "")
            if isinstance(piece, str):
                cumulative_content += piece
                if "<tool_call>" in cumulative_content and tool_call_index is None:
                    tool_call_index = idx
                if "<final_answer>" in cumulative_content and final_answer_index is None:
                    final_answer_index = idx

        if event_name == "tool_result" and tool_result_index is None:
            tool_result_index = idx

        if isinstance(payload_obj, dict) and payload_obj.get("type") == "LoopAlignmentTelemetry":
            alignment_frame_index = idx

        if event_name == "done" and isinstance(payload_obj, dict):
            done_events.append(payload_obj)

    assert tool_call_index is not None, "expected <tool_call> frame in SSE stream"
    assert tool_result_index is not None, "expected tool_result frame in SSE stream"
    assert final_answer_index is not None, "expected <final_answer> frame in SSE stream"

    assert tool_call_index < tool_result_index < final_answer_index

    assert alignment_frame_index is not None, "LoopAlignmentTelemetry should appear in SSE stream"

    events = app.state.event_bus.events
    alignment_events = [evt for evt in events if evt.get("type") == "LoopAlignmentTelemetry"]
    assert alignment_events, "LoopAlignmentTelemetry event should be emitted"
    telemetry = alignment_events[0]
    assert telemetry["atoms"], "telemetry should include atoms"
    assert telemetry["energy"] > 0
    assert telemetry["bandit"] >= 1

    assert done_events, "expected TaskSucceeded payload in SSE stream"
    final_event = done_events[-1]
    assert final_event.get("goal") == "integration hello"
    assert final_event.get("user_input") == "integration hello"
