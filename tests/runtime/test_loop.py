"""Tests for core orchestration loop functionality."""

import asyncio

from reug_runtime.loop import (
    Orchestrator,
    execute_turn,
    parse_tool_calls,
)
from tests.runtime.fakes import FakeAbilityRegistry, FakeEventBus, FakeKG, FakeLLM


def test_parse_tool_calls() -> None:
    """Test tool call parsing from streamed text."""
    text = (
        'Some text <tool_call>{"tool":"echo","args":{"payload":"test"}}</tool_call> '
        "more text"
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "echo"
    assert "payload" in calls[0]["function"]["arguments"]


def test_parse_tool_calls_malformed() -> None:
    """Test that malformed tool calls are ignored."""
    text = "<tool_call>invalid json</tool_call>"
    calls = parse_tool_calls(text)
    assert len(calls) == 0


def test_orchestrator_init() -> None:
    """Test orchestrator initialization."""
    event_bus = FakeEventBus()
    registry = FakeAbilityRegistry()
    model = FakeLLM()
    correlation_id = "test-123"

    orchestrator = Orchestrator(event_bus, registry, model, correlation_id)

    assert orchestrator.event_bus is event_bus
    assert orchestrator.registry is registry
    assert orchestrator.model is model
    assert orchestrator.correlation_id == correlation_id
    assert orchestrator._tool_service is not None


def test_execute_turn_basic() -> None:
    """Test basic execute_turn functionality."""
    event_bus = FakeEventBus()
    registry = FakeAbilityRegistry()
    kg = FakeKG()
    model = FakeLLM()

    async def _run() -> list[dict[str, str]]:
        events: list[dict[str, str]] = []
        async for event in execute_turn(
            "test message",
            "session1",
            event_bus,
            registry,
            kg,
            model,
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    event_types = [e["type"] for e in events]
    assert event_types[0] == "TaskStarted"
    assert event_types[-1] == "TaskSucceeded"
    assert "LLMChunk" in event_types
    ability_called_index = event_types.index("AbilityCalled")
    ability_succeeded_index = event_types.index("AbilitySucceeded")
    assert ability_called_index < ability_succeeded_index
    loop_alignment_index = event_types.index("LoopAlignmentTelemetry")
    assert ability_succeeded_index < loop_alignment_index < len(event_types) - 1


def test_execute_turn_without_kg() -> None:
    """Test execute_turn without knowledge graph."""
    event_bus = FakeEventBus()
    registry = FakeAbilityRegistry()
    model = FakeLLM()

    async def _run() -> list[dict[str, str]]:
        events: list[dict[str, str]] = []
        async for event in execute_turn(
            "test message",
            "session1",
            event_bus,
            registry,
            None,
            model,
        ):
            events.append(event)
        return events

    events = asyncio.run(_run())

    event_types = [e["type"] for e in events]
    assert event_types[0] == "TaskStarted"
    assert event_types[-1] == "TaskSucceeded"


def test_execute_turn_correlation_ids() -> None:
    """Correlation IDs should be stable across emitted events."""
    event_bus = FakeEventBus()
    registry = FakeAbilityRegistry()
    kg = FakeKG()
    model = FakeLLM()

    async def _run() -> list[dict[str, str]]:
        return [
            event
            async for event in execute_turn(
                "test message",
                "session42",
                event_bus,
                registry,
                kg,
                model,
            )
        ]

    events = asyncio.run(_run())

    correlation_ids = {
        event["correlation_id"] for event in events if "correlation_id" in event
    }
    assert len(correlation_ids) == 1

    corr_id = correlation_ids.pop()
    assert corr_id.startswith("session42-")
