"""Tests for telemetry pipeline."""
from __future__ import annotations

from typing import Any

import pytest

from src.plugins.telemetry_pipeline import TelemetryPipelinePlugin
from src.plugins.telemetry_pipeline.orchestrator import (
    TelemetryPipelineOrchestrator,
)


class MockLLMProvider:
    """Mock LLM provider for testing."""

    async def generate(self, prompt: str) -> str:
        """Generate mock response."""
        if "relevance" in prompt.lower():
            return (
                '{"id":"E001","keep":true,"relevance":0.9,"reason":"latency spike"}\n'
                '{"id":"E002","keep":true,"relevance":0.8,"reason":"connection error"}'
            )
        if "cluster" in prompt.lower():
            return (
                '[{"cluster_id":"C1","topic":"streaming_issues",'
                '"members":[{"id":"E001"},{"id":"E002"}],"conflicts":[],'
                '"summary":"Chat streaming experiencing latency issues"}]'
            )
        return "mock response"


@pytest.fixture
def sample_telemetry() -> list[dict[str, Any]]:
    """Sample telemetry data for testing."""
    return [
        {
            "id": "E001",
            "timestamp": "2024-01-01T12:00:00Z",
            "type": "latency_spike",
            "message": "p95 latency increased to 1.8s",
            "session_id": "sess_123",
            "tool_name": "chat_stream",
            "latency_ms": 1800,
        },
        {
            "id": "E002",
            "timestamp": "2024-01-01T12:05:00Z",
            "type": "error",
            "message": "SSE connection dropped",
            "session_id": "sess_123",
            "error_count": 5,
        },
    ]


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Mock LLM provider fixture."""
    return MockLLMProvider()


async def test_orchestrator_end_to_end(
    sample_telemetry: list[dict[str, Any]], mock_llm: MockLLMProvider
) -> None:
    """Test full pipeline execution."""
    orchestrator = TelemetryPipelineOrchestrator(mock_llm)

    result = await orchestrator.process_telemetry(
        task="Diagnose chat streaming latency",
        telemetry_items=sample_telemetry,
        constraints=["p95 < 500ms", "cost delta < 10%"],
        token_budget=1000,
    )

    assert "# Task" in result
    assert "# Critical facts" in result
    assert "Diagnose chat streaming latency" in result


async def test_orchestrator_without_llm(
    sample_telemetry: list[dict[str, Any]]
) -> None:
    """Test pipeline works without LLM provider."""
    orchestrator = TelemetryPipelineOrchestrator(None)

    result = await orchestrator.process_telemetry(
        task="Test task",
        telemetry_items=sample_telemetry,
        token_budget=1000,
    )

    assert "# Task" in result
    assert "Test task" in result


async def test_plugin_integration() -> None:
    """Test plugin initialization and processing."""
    plugin = TelemetryPipelinePlugin()

    # Setup with mock LLM provider in config
    config = {"llm_provider": MockLLMProvider()}
    await plugin.setup(event_bus=None, store=None, config=config)
    await plugin.start()
    assert plugin.is_running

    info = plugin.get_tools()
    assert len(info) == 1
    assert info[0]["name"] == "process_telemetry"

    sample_data = [
        {
            "id": "E001",
            "message": "test message",
            "timestamp": "2024-01-01T12:00:00Z",
        }
    ]

    result = await plugin.process_telemetry(
        task="Test task", telemetry_items=sample_data
    )

    assert "# Task" in result
    assert "Test task" in result

    await plugin.stop()
    assert not plugin.is_running


async def test_empty_telemetry() -> None:
    """Test pipeline with empty telemetry."""
    orchestrator = TelemetryPipelineOrchestrator(None)

    result = await orchestrator.process_telemetry(
        task="Empty test", telemetry_items=[], token_budget=1000
    )

    assert "# Task" in result
    assert "Empty test" in result
    assert "- None" in result  # No critical facts