import asyncio
import json
from typing import Any

import pytest

from reug_runtime.event_bus import (
    FileEventBus,
    ProductionEventBus,
    RedisEventBus,
)


@pytest.mark.asyncio
async def test_file_event_bus_writes(tmp_path):
    bus = FileEventBus(str(tmp_path))
    event = {"event_type": "PING"}
    await bus.emit(event)

    log = tmp_path / "events.jsonl"
    assert log.exists()
    data = json.loads(log.read_text().strip())
    assert data["event_type"] == "PING"
    assert "timestamp" in data


@pytest.mark.asyncio
@pytest.mark.integration_redis()
async def test_redis_event_bus_publishes():
    import redis

    try:
        client = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True)
        client.ping()
    except Exception:
        pytest.skip("Redis server not available")

    channel = "test.reug_runtime.events"
    pubsub = client.pubsub()
    pubsub.subscribe(channel)
    pubsub.get_message(timeout=0.1)

    bus = RedisEventBus(url="redis://127.0.0.1:6379/0", channel=channel)
    await bus.emit({"event_type": "PING"})

    message = None
    for _ in range(10):
        message = pubsub.get_message(timeout=1.0)
        if message and message.get("type") == "message":
            break
    assert message is not None
    data = json.loads(message["data"])
    assert data["event_type"] == "PING"


@pytest.mark.asyncio
async def test_production_event_bus_dispatches_and_tracks_metrics(tmp_path):
    bus = ProductionEventBus(str(tmp_path), queue_max_sizes={"high": 5, "medium": 5, "low": 5})
    handled = asyncio.Event()
    events: list[dict[str, Any]] = []

    async def handler(event: dict[str, Any]) -> None:
        events.append(event)
        handled.set()

    await bus.subscribe("PING", handler)
    payload = {"event_type": "PING", "data": {"value": 1}}
    await bus.publish(payload, priority="high")

    await asyncio.wait_for(handled.wait(), timeout=1.0)

    snapshot = bus.metrics.snapshot()
    assert snapshot["success"] == 1
    assert events and events[0]["event_type"] == "PING"


@pytest.mark.asyncio
async def test_production_event_bus_backpressure_routes_to_dead_letter(tmp_path):
    bus = ProductionEventBus(str(tmp_path), queue_max_sizes={"medium": 1, "high": 1, "low": 1})
    # Prevent the worker from draining the queue so the second publish hits backpressure
    bus._ensure_worker = lambda priority: None  # type: ignore[attr-defined]
    await bus.publish({"event_type": "PING", "priority": "low"})
    await bus.publish({"event_type": "PING", "priority": "low", "importance": 0.1})

    assert bus.dead_letter_queue.qsize() == 1
    dead_letter_event = await bus.dead_letter_queue.get()
    assert dead_letter_event["dead_letter_reason"] == "backpressure"


@pytest.mark.asyncio
async def test_production_event_bus_degraded_publish_uses_fallback(tmp_path):
    captured: list[dict[str, Any]] = []

    class DummyFallback(FileEventBus):
        async def emit(self, event: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]
            captured.append(event)
            return await super().emit(event)

    fallback = DummyFallback(str(tmp_path / "fallback"))
    bus = ProductionEventBus(str(tmp_path), fallback_bus=fallback)
    # Simulate high error rate
    bus.metrics.record_publish_failure()
    bus.metrics.record_publish_failure()

    event = {"event_type": "PING", "data": {}}
    await bus.publish(event)

    assert captured == [event]
