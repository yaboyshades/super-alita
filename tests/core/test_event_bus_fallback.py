import asyncio

import pytest

from src.core.event_bus import EventBus


@pytest.mark.asyncio
async def test_event_bus_initialize_falls_back_to_memory(monkeypatch):
    # Use an invalid port to force connection failure
    bus = EventBus(redis_url="redis://127.0.0.1:0/0")

    # Initialize should not raise and should mark running
    ok = await bus.initialize()
    assert ok is True
    assert bus.is_running is True
    assert bus.backend == "memory"

    # If redis isn't available, backend will be memory and local delivery should work
    received = []
    done = asyncio.Event()

    async def handler(evt):
        received.append(getattr(evt, "event_type", "unknown"))
        done.set()

    await bus.subscribe("test_fallback", handler)
    await bus.emit("test_fallback", source_plugin="test", payload={"x": 1})

    await asyncio.wait_for(done.wait(), timeout=2.0)
    assert received == ["test_fallback"]

    await bus.shutdown()
