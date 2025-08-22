"""Metrics tests for NeuralAtom safe_execute latency tracking."""

import asyncio

import pytest
from src.core.neural_atom import NeuralAtom, NeuralAtomMetadata


class TestNeuralAtom(NeuralAtom):
    """Test implementation of NeuralAtom for metrics testing."""

    def __init__(self, name: str = "test_atom", delay: float = 0.01):
        metadata = NeuralAtomMetadata(
            name=name,
            description="Test neural atom for metrics validation",
            capabilities=["test", "metrics"],
        )
        super().__init__(metadata)
        self.delay = delay

    async def execute(self, input_data):
        """Test execute with configurable delay."""
        await asyncio.sleep(self.delay)
        return f"Processed: {input_data}"

    def get_embedding(self):
        """Return dummy embedding."""
        return [0.1] * 1024

    def can_handle(self, task_description: str) -> float:
        """Return confidence score with realistic heuristics."""
        if "test" in task_description.lower():
            return 0.9
        if "metrics" in task_description.lower():
            return 0.8
        return 0.3


@pytest.mark.asyncio
async def test_average_latency_ms_property():
    """Test that average_latency_ms property works correctly."""

    # Create test atom with 10ms delay
    atom = TestNeuralAtom(delay=0.01)  # 10ms

    # Initially should be 0
    assert atom.average_latency_ms == 0.0

    # Execute once
    result = await atom.safe_execute("test input 1")
    assert result["success"] is True

    # Should now have some latency recorded
    first_latency = atom.average_latency_ms
    assert first_latency > 0.0

    # Execute again
    result = await atom.safe_execute("test input 2")
    assert result["success"] is True

    # Latency should be updated (exponential moving average)
    second_latency = atom.average_latency_ms
    assert second_latency > 0.0


@pytest.mark.asyncio
async def test_metrics_after_20_executions():
    """Fire 20 dummy executions and assert average_latency_ms > 0."""

    # Create test atom with small delay for faster testing
    atom = TestNeuralAtom(delay=0.005)  # 5ms

    # Fire 20 executions
    for i in range(20):
        result = await atom.safe_execute(f"test input {i}")
        assert result["success"] is True

    # Assert metrics are properly tracked
    assert atom.metadata.usage_count == 20
    assert atom.metadata.success_rate == 1.0  # All successful
    assert atom.metadata.avg_execution_time > 0.0
    assert atom.average_latency_ms > 0.0
    # Key assertion from the requirement
    assert (
        atom.average_latency_ms > 0
    ), "Average latency must be greater than 0 after executions"


__all__ = ["TestNeuralAtom"]
