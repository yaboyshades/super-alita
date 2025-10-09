"""Tests for the SSE streaming transformer."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from reug_runtime.streaming import sse_transformer


def _make_event_generator() -> AsyncGenerator[dict[str, object], None]:
    """Create a simple async generator for testing."""

    async def _gen() -> AsyncGenerator[dict[str, object], None]:
        yield {"type": "TaskStarted", "correlation_id": "test-123", "goal": "test"}
        yield {"type": "LLMChunk", "data": {"text": "Hello"}}
        yield {"type": "AbilityCalled", "tool": "echo", "correlation_id": "test-123"}
        yield {
            "type": "AbilitySucceeded",
            "tool": "echo",
            "correlation_id": "test-123",
            "result": {"echo": "hi"},
        }
        yield {
            "type": "TaskSucceeded",
            "correlation_id": "test-123",
            "data": {"content": "done"},
        }

    return _gen()


def test_sse_transformer_basic() -> None:
    """Test basic SSE transformation."""

    async def _run() -> str:
        chunks = []
        async for chunk in sse_transformer(_make_event_generator()):
            chunks.append(chunk)
        return "".join(chunks)

    sse_stream = asyncio.run(_run())

    # Check that we get proper SSE format
    assert "event: start\n" in sse_stream
    assert "event: content\n" in sse_stream
    assert "event: tool_start\n" in sse_stream
    assert "event: tool_result\n" in sse_stream
    assert "event: done\n" in sse_stream

    # Check that data is properly JSON encoded
    assert 'data: {"content": "Hello"}' in sse_stream
    assert '"type": "TaskStarted"' in sse_stream


def test_sse_transformer_llm_chunk() -> None:
    """Test LLM chunk transformation specifically."""

    async def _run() -> str:
        async def _gen():
            yield {"type": "LLMChunk", "data": {"text": "Test chunk"}}

        chunks = []
        async for chunk in sse_transformer(_gen()):
            chunks.append(chunk)
        return "".join(chunks)

    sse_stream = asyncio.run(_run())

    # LLM chunks should be transformed to content events with specific format
    assert "event: content\n" in sse_stream
    assert 'data: {"content": "Test chunk"}' in sse_stream


def test_sse_transformer_empty() -> None:
    """Test SSE transformer with empty generator."""

    async def _run() -> int:
        async def _empty_gen():
            return
            yield  # pragma: no cover

        chunks = []
        async for chunk in sse_transformer(_empty_gen()):
            chunks.append(chunk)
        return len(chunks)

    chunk_count = asyncio.run(_run())

    # Should handle empty generators gracefully
    assert chunk_count == 0


def test_sse_transformer_unknown_event() -> None:
    """Test SSE transformer with unknown event types."""

    async def _run() -> str:
        async def _gen():
            yield {"type": "UnknownEvent", "data": "test"}

        chunks = []
        async for chunk in sse_transformer(_gen()):
            chunks.append(chunk)
        return "".join(chunks)

    sse_stream = asyncio.run(_run())

    # Unknown events should be mapped to "message"
    assert "event: message\n" in sse_stream
    assert '"type": "UnknownEvent"' in sse_stream


def test_sse_transformer_heartbeat(monkeypatch: Any) -> None:
    """Heartbeat configuration should emit ping events."""
    monkeypatch.setenv("ALITA_SSE_HEARTBEAT", "1")
    monkeypatch.setenv("ALITA_SSE_HEARTBEAT_INTERVAL", "1")

    async def _run() -> list[str]:
        async def _gen():
            yield {"type": "TaskStarted", "correlation_id": "hb-1", "goal": "ping"}
            await asyncio.sleep(1.05)

        gen = sse_transformer(_gen())
        chunks: list[str] = []
        try:
            for _ in range(5):
                chunk = await anext(gen)
                chunks.append(chunk)
                if "event: ping" in chunk:
                    break
        finally:
            await gen.aclose()
        return chunks

    chunks = asyncio.run(_run())

    assert any("event: ping" in chunk for chunk in chunks)
