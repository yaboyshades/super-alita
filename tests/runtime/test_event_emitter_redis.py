import json

import pytest
import redis

from reug.events import EventEmitter


@pytest.mark.integration_redis()
def test_emit_redis_publishes(monkeypatch):
    client = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True)
    try:
        client.ping()
    except Exception:
        pytest.skip("Redis server not available")

    channel = "test.reug.events"
    pubsub = client.pubsub()
    pubsub.subscribe(channel)
    # Drain subscription confirmation message
    pubsub.get_message(timeout=0.1)

    monkeypatch.setenv("REUG_EVENTBUS_CHANNEL", channel)
    emitter = EventEmitter(sink="redis")
    emitter.emit(event_type="PING")

    message = None
    for _ in range(10):
        message = pubsub.get_message(timeout=1.0)
        if message and message.get("type") == "message":
            break
    assert message is not None
    assert message.get("type") == "message"
    data = json.loads(message["data"])
    assert data["event_type"] == "PING"
