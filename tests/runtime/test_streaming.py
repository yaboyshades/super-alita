"""Tests for the SSE streaming transformer."""

import asyncio


import pytest

from collections.abc import AsyncGenerator
from typing import Any


from reug_runtime.streaming import sse_transformer


def _make_event_generator() -> AsyncGenerator[dict[str, object], None]:
    """Create a simple async generator for testing."""

    async def _gen() -> AsyncGenerator[dict[str, object], None]:
        yield {"type": "TaskStarted", "correlation_id": "test-123", "goal": "test"}
        yield {"type": "LLMChunk", "data": {"text": "Hello"}}
        yield {"type": "AbilityCalled", "tool": "echo", "correlation_id": "test-123"}

        yield {"type": "AbilitySucceeded", "tool": "echo", "correlation_id": "test-123", "result": {"echo": "hi"}}
        yield {"type": "TaskSucceeded", "correlation_id": "test-123", "data": {"content": "done"}}

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



def _collect_sse_chunks(event_gen):
    async def _collect():
        chunks: list[str] = []
        async for chunk in sse_transformer(event_gen):
            chunks.append(chunk)
        return chunks

    return asyncio.run(_collect())


def test_sse_transformer_basic():
    """Test basic SSE transformation."""
    event_gen = _make_event_generator()
    chunks = _collect_sse_chunks(event_gen)

    # Join all chunks to get the full SSE stream
    sse_stream = "".join(chunks)

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


def test_sse_transformer_llm_chunk():
    """Test LLM chunk transformation specifically."""
    async def _gen():
        yield {"type": "LLMChunk", "data": {"text": "Test chunk"}}

    chunks = _collect_sse_chunks(_gen())

    sse_stream = "".join(chunks)

    # LLM chunks should be transformed to content events with specific format
    assert "event: content\n" in sse_stream
    assert 'data: {"content": "Test chunk"}' in sse_stream


def test_sse_transformer_empty():
    """Test SSE transformer with empty generator."""
    async def _empty_gen():
        return
        yield  # pragma: no cover

    chunks = _collect_sse_chunks(_empty_gen())

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


def test_sse_transformer_event_name_mapping():
    """Ensure every orchestrator event maps to the expected SSE alias."""

    events = [
        (
            {"type": "TaskStarted", "correlation_id": "cid", "goal": "g", "session_id": "s"},
            "start",
        ),
        (
            {"type": "LLMChunk", "data": {"text": "hello"}},
            "content",
        ),
        (
            {
                "type": "AbilityCalled",
                "tool": "echo",
                "correlation_id": "cid",
                "span_id": "span-1",
            },
            "tool_start",
        ),
        (
            {
                "type": "AbilitySucceeded",
                "tool": "echo",
                "correlation_id": "cid",
                "span_id": "span-1",
                "result": {"echo": "hi"},
            },
            "tool_result",
        ),
        (
            {
                "type": "AbilityFailed",
                "tool": "echo",
                "correlation_id": "cid",
                "span_id": "span-2",
                "error": "boom",
            },
            "tool_error",
        ),
        (
            {
                "type": "KnowledgeContextRetrieved",
                "correlation_id": "cid",
                "session_id": "s",
                "snippet": "ctx",
                "goal_id": "goal-1",
            },
            "message",
        ),
        (
            {
                "type": "KnowledgeAtomCreated",
                "correlation_id": "cid",
                "session_id": "s",
                "atom_id": "atom-1",
                "atom_type": "final_answer",
            },
            "message",
        ),
        (
            {
                "type": "LoopAlignmentTelemetry",
                "correlation_id": "cid",
                "session_id": "s",
                "atoms": ["a"],
                "bonds": [],
                "energy": 1.0,
                "todo": 0,
                "bandit": 0,
                "reward": {"success": 1.0},
            },
            "message",
        ),
        (
            {
                "type": "KnowledgeBondCreated",
                "correlation_id": "cid",
                "session_id": "s",
                "bond_type": "ANSWERED",
                "source_atom_id": "goal-1",
                "target_atom_id": "atom-1",
            },
            "message",
        ),
        (
            {
                "type": "TaskSucceeded",
                "correlation_id": "cid",
                "session_id": "s",
                "data": {"content": "done"},
            },
            "done",
        ),
    ]

    async def _gen():
        for ev, _ in events:
            yield ev

    chunks = _collect_sse_chunks(_gen())

    event_names = [
        chunk.strip().split(": ", 1)[1]
        for chunk in chunks
        if chunk.startswith("event: ")
    ]

    assert event_names == [expected for _, expected in events]
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
