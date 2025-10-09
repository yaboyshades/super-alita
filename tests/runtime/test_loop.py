"""Tests for core orchestration loop functionality."""

import pytest

from reug_runtime.loop import Orchestrator, execute_turn, parse_tool_calls
from tests.runtime.fakes import FakeAbilityRegistry, FakeEventBus, FakeKG, FakeLLM


def test_parse_tool_calls():
    """Test tool call parsing from streamed text."""
    text = 'Some text <tool_call>{"tool":"echo","args":{"payload":"test"}}</tool_call> more text'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "echo"
    assert "payload" in calls[0]["function"]["arguments"]


def test_parse_tool_calls_malformed():
    """Test that malformed tool calls are ignored."""
    text = '<tool_call>invalid json</tool_call>'
    calls = parse_tool_calls(text)
    assert len(calls) == 0


def test_orchestrator_init():
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


@pytest.mark.asyncio
async def test_execute_turn_basic():
    """Test basic execute_turn functionality."""
    event_bus = FakeEventBus()
    registry = FakeAbilityRegistry()
    kg = FakeKG()
    model = FakeLLM()
    
    events = []
    async for event in execute_turn(
        "test message", "session1", event_bus, registry, kg, model
    ):
        events.append(event)
    
    # Check that we get the expected event sequence
    event_types = [e["type"] for e in events]
    assert "TaskStarted" in event_types
    assert "LLMChunk" in event_types
    assert "AbilityCalled" in event_types
    assert "AbilitySucceeded" in event_types
    assert "TaskSucceeded" in event_types


@pytest.mark.asyncio
async def test_execute_turn_without_kg():
    """Test execute_turn without knowledge graph."""
    event_bus = FakeEventBus()
    registry = FakeAbilityRegistry()
    model = FakeLLM()
    
    events = []
    async for event in execute_turn(
        "test message", "session1", event_bus, registry, None, model
    ):
        events.append(event)
    
    # Should still work without KG
    event_types = [e["type"] for e in events]
    assert "TaskStarted" in event_types
    assert "TaskSucceeded" in event_types
