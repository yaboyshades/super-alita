import asyncio
import gc
import json
import tracemalloc
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


@pytest.mark.asyncio
async def test_production_event_bus_memory_usage_under_sustained_load(tmp_path):
    bus = ProductionEventBus(
        str(tmp_path), queue_max_sizes={"medium": 2048, "high": 128, "low": 2048}
    )
    total_events = 2000
    processed = 0
    processed_lock = asyncio.Lock()
    done = asyncio.Event()

    async def handler(event: dict[str, Any]) -> None:
        nonlocal processed
        async with processed_lock:
            processed += 1
            if processed >= total_events:
                done.set()

    await bus.subscribe("PING", handler)

    tracemalloc.start()
    try:
        gc.collect()
        baseline_current, _ = tracemalloc.get_traced_memory()

        for idx in range(total_events):
            payload = {"event_type": "PING", "seq": idx}
            await bus.publish(payload, priority="medium")

        await asyncio.wait_for(done.wait(), timeout=10.0)
        await asyncio.wait_for(
            asyncio.gather(*(queue.join() for queue in bus.queues.values())),
            timeout=10.0,
        )
        gc.collect()
        current_after, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert processed == total_events
    # Memory usage should not grow unbounded during sustained load.
    assert current_after - baseline_current < 2 * 1024 * 1024


@pytest.mark.asyncio
async def test_production_event_bus_thread_safety_under_high_concurrency(tmp_path):
    bus = ProductionEventBus(
        str(tmp_path), queue_max_sizes={"medium": 2048, "high": 256, "low": 256}
    )
    publishers = 20
    events_per_publisher = 50
    total_events = publishers * events_per_publisher
    processed = 0
    processed_lock = asyncio.Lock()
    done = asyncio.Event()

    async def handler(event: dict[str, Any]) -> None:
        nonlocal processed
        async with processed_lock:
            processed += 1
            if processed >= total_events:
                done.set()

    await bus.subscribe("PING", handler)

    async def publish_events(publisher_id: int) -> None:
        for idx in range(events_per_publisher):
            await bus.publish(
                {
                    "event_type": "PING",
                    "publisher": publisher_id,
                    "seq": idx,
                },
                priority="medium",
            )

    await asyncio.gather(*(publish_events(i) for i in range(publishers)))

    await asyncio.wait_for(done.wait(), timeout=15.0)
    await asyncio.wait_for(
        asyncio.gather(*(queue.join() for queue in bus.queues.values())),
        timeout=15.0,
    )

    snapshot = bus.metrics.snapshot()
    assert processed == total_events
    assert snapshot["success"] == total_events
    assert snapshot["failure"] == 0
    assert snapshot["dropped"] == 0


@pytest.mark.asyncio
async def test_production_event_bus_circuit_breaker_recovers(tmp_path):
    captured: list[dict[str, Any]] = []

    class CapturingFallback(FileEventBus):
        async def emit(self, event: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]
            captured.append(event)
            return await super().emit(event)

    fallback = CapturingFallback(str(tmp_path / "fallback"))
    bus = ProductionEventBus(
        str(tmp_path),
        degraded_error_rate=0.5,
        queue_max_sizes={"medium": 32, "high": 32, "low": 32},
        fallback_bus=fallback,
    )

    for _ in range(3):
        bus.metrics.record_publish_failure()

    degraded_event = {"event_type": "PING", "id": "degraded"}
    await bus.publish(degraded_event)

    assert captured == [degraded_event]

    for _ in range(7):
        bus.metrics.record_publish_success()

    processed = asyncio.Event()

    async def handler(event: dict[str, Any]) -> None:
        if event.get("id") == "recovered":
            processed.set()

    await bus.subscribe("PING", handler)

    recovered_event = {"event_type": "PING", "id": "recovered"}
    await bus.publish(recovered_event, priority="high")

    await asyncio.wait_for(processed.wait(), timeout=5.0)
    await asyncio.wait_for(
        asyncio.gather(*(queue.join() for queue in bus.queues.values())),
        timeout=5.0,
    )

    assert captured == [degraded_event]
