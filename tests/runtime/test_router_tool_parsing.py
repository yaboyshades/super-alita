"""Tests for helper utilities in :mod:`src.reug_runtime.router`."""

import json

import pytest

from src.reug_runtime.router import OrchestratorPhase, execute_turn, parse_tool_calls
from tests.runtime.fakes import FakeAbilityRegistry, FakeEventBus, FakeKG, FakeLLM


def test_parse_tool_calls_generates_dict_payload() -> None:
    text = '<tool_call>{"tool":"search","args":{"query":"plan"}}</tool_call>'

    calls = parse_tool_calls(text)

    assert len(calls) == 1
    call = calls[0]
    assert call["name"] == "search"
    fn = call["function"]
    assert isinstance(fn, dict)
    assert fn["name"] == "search"
    assert json.loads(fn["arguments"]) == {"query": "plan"}
    assert isinstance(call["id"], str) and call["id"]


def test_parse_tool_calls_ignores_invalid_payload() -> None:
    text = '<tool_call>{"tool":"search","args":[]}</tool_call>'

    calls = parse_tool_calls(text)

    assert calls == []


@pytest.mark.asyncio
async def test_execute_turn_emits_expected_state_flow() -> None:
    bus = FakeEventBus()
    reg = FakeAbilityRegistry()
    kg = FakeKG()
    model = FakeLLM()

    events = [
        event
        async for event in execute_turn(
            "state flow check", "sess-1", bus, reg, kg, model
        )
    ]

    transitions = [evt for evt in events if evt["type"] == "STATE_TRANSITION"]
    assert transitions, "expected state transition events to be emitted"

    path = [(evt["from_state"], evt["to_state"]) for evt in transitions]
    assert (OrchestratorPhase.IDLE.value, OrchestratorPhase.REASONING.value) in path
    assert (OrchestratorPhase.REASONING.value, OrchestratorPhase.ACTING.value) in path
    assert (OrchestratorPhase.ACTING.value, OrchestratorPhase.REASONING.value) in path
    assert path[-1][1] == OrchestratorPhase.COMPLETED.value

    ability_called_idx = [
        idx for idx, evt in enumerate(events) if evt["type"] == "AbilityCalled"
    ]
    assert ability_called_idx, "AbilityCalled event should be present"
    for idx in ability_called_idx:
        prior_transitions = [
            evt for evt in events[: idx + 1] if evt["type"] == "STATE_TRANSITION"
        ]
        assert prior_transitions, "Ability event missing preceding state transition"
        assert prior_transitions[-1]["to_state"] == OrchestratorPhase.ACTING.value
