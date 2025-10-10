import json

import pytest

from src.reug_runtime.router import execute_turn, sse_transformer
from tests.runtime.fakes import FakeAbilityRegistry, FakeEventBus, FakeKG, FakeLLM


@pytest.mark.asyncio
async def test_execute_turn_emits_loop_alignment_event() -> None:
    bus = FakeEventBus()
    registry = FakeAbilityRegistry()
    kg = FakeKG()
    model = FakeLLM()

    events: list[dict[str, object]] = []
    async for event in execute_turn("loop", "loop-session", bus, registry, kg, model):
        events.append(event)

    loop_events = [evt for evt in events if evt.get("type") == "LoopAlignmentTelemetry"]
    assert loop_events, "expected loop alignment telemetry event"

    telemetry = loop_events[0]
    assert telemetry["atoms"], "atoms should include goal/final identifiers"
    assert telemetry["bonds"], "bonds should preview pending graph connections"
    assert telemetry["bonds"][0]["type"] == "ANSWERED"
    assert telemetry["energy"] > 0
    assert telemetry["todo"] == 0
    assert telemetry["bandit"] >= 1
    reward = telemetry["reward"]
    assert isinstance(reward, dict) and reward.get("success") == 1.0

    event_types = [evt["type"] for evt in bus.events]
    assert "LoopAlignmentTelemetry" in event_types
    assert "KnowledgeBondCreated" in event_types
    assert event_types.index("LoopAlignmentTelemetry") < event_types.index(
        "KnowledgeBondCreated"
    )


@pytest.mark.asyncio
async def test_state_transitions_follow_single_turn_path() -> None:
    bus = FakeEventBus()
    registry = FakeAbilityRegistry()
    kg = FakeKG()
    model = FakeLLM()

    events = [
        event
        async for event in execute_turn(
            "path", "state-session", bus, registry, kg, model
        )
    ]

    transition_pairs = [
        (event["from"], event["to"])
        for event in events
        if event.get("type") == "STATE_TRANSITION"
    ]

    assert transition_pairs == [
        ("AWAITING_INPUT", "DECOMPOSE_TASK"),
        ("DECOMPOSE_TASK", "SELECT_TOOL"),
        ("SELECT_TOOL", "EXECUTE_TOOL"),
        ("EXECUTE_TOOL", "PROCESS_TOOL_RESULT"),
        ("PROCESS_TOOL_RESULT", "SELECT_TOOL"),
        ("SELECT_TOOL", "RESPONDING_SUCCESS"),
    ]

    bus_transition_pairs = [
        (event["from"], event["to"])
        for event in bus.events
        if event.get("type") == "STATE_TRANSITION"
    ]
    assert bus_transition_pairs == transition_pairs


@pytest.mark.asyncio
async def test_sse_stream_includes_loop_alignment_frame() -> None:
    bus = FakeEventBus()
    registry = FakeAbilityRegistry()
    kg = FakeKG()
    model = FakeLLM()

    sse_chunks: list[str] = []
    async for chunk in sse_transformer(
        execute_turn("loop", "sse-session", bus, registry, kg, model)
    ):
        sse_chunks.append(chunk)

    payload = "".join(sse_chunks)
    frames = [frame for frame in payload.split("\n\n") if frame.strip()]
    telemetry_frames: list[dict[str, object]] = []
    for frame in frames:
        data_line = next(
            (line for line in frame.splitlines() if line.startswith("data: ")), None
        )
        if not data_line:
            continue
        try:
            parsed = json.loads(data_line[len("data: ") :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("type") == "LoopAlignmentTelemetry":
            telemetry_frames.append(parsed)

    assert telemetry_frames, "Loop alignment telemetry frame missing from SSE stream"
    telemetry = telemetry_frames[0]
    assert telemetry["atoms"], "SSE telemetry should echo atoms"
    assert telemetry["energy"] > 0
    assert telemetry["bandit"] >= 1
    assert telemetry["todo"] == 0
