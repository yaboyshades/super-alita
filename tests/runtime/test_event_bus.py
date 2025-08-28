import json

import pytest

from reug_runtime.event_bus import FileEventBus, RedisEventBus


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
async def test_event_bus_stress_throughput_and_backpressure(monkeypatch):
    """High-volume publish test with throughput metrics and backpressure."""
    import asyncio

    import fakeredis.aioredis
    import redis.asyncio as redis

    from src.core.event_bus_clean import EventBus
    from src.core.events import BaseEvent

    fake = fakeredis.aioredis.FakeRedis()
    monkeypatch.setattr(redis, "Redis", lambda *a, **k: fake)

    EventBus._instance = None
    bus = EventBus()

    max_queue = 50
    queue: asyncio.Queue[BaseEvent] = asyncio.Queue(maxsize=max_queue)
    dropped = 0

    async def handler(evt: BaseEvent) -> None:
        nonlocal dropped
        try:
            queue.put_nowait(evt)
        except asyncio.QueueFull:
            dropped += 1

    await bus.subscribe("test_event", handler)
    await bus.start()

    total = 200
    for _ in range(total):
        await bus.publish(BaseEvent(event_type="test_event", source_plugin="tester"))

    await asyncio.sleep(1.1)
    await bus.publish(BaseEvent(event_type="test_event", source_plugin="tester"))
    await asyncio.sleep(0.1)

    metrics = bus.get_metrics()
    assert metrics["recv_count"] == total + 1
    assert metrics["eps"] > 0

    processed = queue.qsize()
    assert processed <= max_queue
    assert processed + dropped == total + 1

    await bus.shutdown()
