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
