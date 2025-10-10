"""Tests for telemetry pipeline"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.plugins.telemetry_pipeline.orchestrator import (
    TelemetryPipelineOrchestrator,
)


@pytest.fixture
def sample_telemetry() -> list[dict[str, Any]]:
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


async def test_pipeline_end_to_end(
    sample_telemetry: list[dict[str, Any]]
) -> None:
    """Test full pipeline execution"""
    # Mock dependencies
    mock_registry = MagicMock()  # Mock AbilityRegistry
    mock_llm = MagicMock()  # Mock LLMProvider

    orchestrator = TelemetryPipelineOrchestrator(mock_registry, mock_llm)

    result = await orchestrator.process_telemetry(
        task="Diagnose chat streaming latency",
        telemetry_items=sample_telemetry,
        constraints=["p95 < 500ms", "cost delta < 10%"],
        token_budget=1000,
    )

    assert "# Task" in result
    assert "# Critical facts" in result
    assert "[E001" in result  # Source attribution
