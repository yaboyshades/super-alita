"""Tests for SSE streaming functionality."""

import asyncio
import json
import pytest

from reug_runtime.streaming import sse_transformer


def _make_event_generator():
    """Create a simple async generator for testing."""
    async def _gen():
        yield {"type": "TaskStarted", "correlation_id": "test-123", "goal": "test"}
        yield {"type": "LLMChunk", "data": {"text": "Hello"}}
        yield {"type": "AbilityCalled", "tool": "echo", "correlation_id": "test-123"}
        yield {"type": "AbilitySucceeded", "tool": "echo", "correlation_id": "test-123", "result": {"echo": "hi"}}
        yield {"type": "TaskSucceeded", "correlation_id": "test-123", "data": {"content": "done"}}
    
    return _gen()


@pytest.mark.asyncio
async def test_sse_transformer_basic():
    """Test basic SSE transformation."""
    event_gen = _make_event_generator()
    
    chunks = []
    async for chunk in sse_transformer(event_gen):
        chunks.append(chunk)
    
    # Join all chunks to get the full SSE stream
    sse_stream = "".join(chunks)
    
    # Check that we get proper SSE format
    assert "event: start\n" in sse_stream
    assert "event: content\n" in sse_stream
    assert "event: tool_start\n" in sse_stream
    assert "event: tool_result\n" in sse_stream
    assert "event: done\n" in sse_stream
    
    # Check that data is properly JSON encoded
    assert 'data: {"content": "Hello"}' in sse_stream
    assert '"type": "TaskStarted"' in sse_stream


@pytest.mark.asyncio
async def test_sse_transformer_llm_chunk():
    """Test LLM chunk transformation specifically."""
    async def _gen():
        yield {"type": "LLMChunk", "data": {"text": "Test chunk"}}
    
    chunks = []
    async for chunk in sse_transformer(_gen()):
        chunks.append(chunk)
    
    sse_stream = "".join(chunks)
    
    # LLM chunks should be transformed to content events with specific format
    assert "event: content\n" in sse_stream
    assert 'data: {"content": "Test chunk"}' in sse_stream


@pytest.mark.asyncio
async def test_sse_transformer_empty():
    """Test SSE transformer with empty generator."""
    async def _empty_gen():
        return
        yield  # pragma: no cover
    
    chunks = []
    async for chunk in sse_transformer(_empty_gen()):
        chunks.append(chunk)
    
    # Should handle empty generators gracefully
    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_sse_transformer_unknown_event():
    """Test SSE transformer with unknown event types."""
    async def _gen():
        yield {"type": "UnknownEvent", "data": "test"}
    
    chunks = []
    async for chunk in sse_transformer(_gen()):
        chunks.append(chunk)
    
    sse_stream = "".join(chunks)
    
    # Unknown events should be mapped to "message"
    assert "event: message\n" in sse_stream
    assert '"type": "UnknownEvent"' in sse_stream
