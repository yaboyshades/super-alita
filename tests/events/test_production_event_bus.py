import time

import pytest

from src.events.production_event_bus import ProductionEventBus


class FakeRedis:
    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict[bytes, bytes]]]] = {}
        self.groups: set[tuple[str, str]] = set()

    async def xadd(self, stream: str, data: dict[str, str], maxlen: int | None = None):  # type: ignore[override]
        payload = {key.encode(): value.encode() if isinstance(value, str) else value for key, value in data.items()}
        entries = self.streams.setdefault(stream, [])
        entry_id = f"{int(time.time() * 1000)}-{len(entries)}"
        entries.append((entry_id, payload))
        if maxlen and len(entries) > maxlen:
            del entries[0]
        return entry_id

    async def xlen(self, stream: str):  # type: ignore[override]
        return len(self.streams.get(stream, []))

    async def xgroup_create(self, stream: str, groupname: str, id: str = "0", mkstream: bool = False):  # type: ignore[override]
        self.groups.add((stream, groupname))
        self.streams.setdefault(stream, [])

    async def xrevrange(self, stream: str, count: int = 100):  # type: ignore[override]
        entries = list(reversed(self.streams.get(stream, [])))
        return entries[:count]

    async def xrange(self, stream: str, min: str = "-", max: str = "+"):  # type: ignore[override]
        return list(self.streams.get(stream, []))

    async def xack(self, stream: str, group: str, *ids):  # type: ignore[override]
        return len(ids)

    async def xreadgroup(self, *_, **__):  # type: ignore[override]
        return []


@pytest.mark.asyncio
async def test_emit_publishes_event():
    fake = FakeRedis()
    bus = ProductionEventBus("redis://localhost", redis_client=fake)

    event_data = {
        "type": "user.action",
        "user_id": "user-1",
        "action_type": "open",
        "timestamp": "2024-01-01T00:00:00",
        "data": {"details": "opened"},
    }

    result = await bus.emit(event_data)

    assert result["event_id"].startswith("event_")
    assert await fake.xlen("events") == 1


@pytest.mark.asyncio
async def test_process_event_with_retry_moves_to_dlq():
    fake = FakeRedis()
    bus = ProductionEventBus("redis://localhost", redis_client=fake)

    async def failing_handler(_event):
        raise RuntimeError("boom")

    event = {
        "id": "event_test",
        "type": "system.alert",
        "correlation_id": "corr",
        "timestamp": "2024-01-01T00:00:00",
        "source": "unit",
        "data": {"alert_type": "system", "severity": "high", "message": "boom"},
        "metadata": {},
        "alert_type": "system",
        "severity": "high",
        "message": "boom",
    }

    success = await bus._process_event_with_retry(
        failing_handler,
        event,
        "1-0",
        "default",
        "consumer",
        max_retries=1,
    )

    assert success is False
    dlq_events = await bus.dead_letter_queue.get_failed_events()
    assert len(dlq_events) == 1
    assert dlq_events[0]["original_event"]["id"] == "event_test"


@pytest.mark.asyncio
async def test_get_metrics_reflects_activity():
    fake = FakeRedis()
    bus = ProductionEventBus("redis://localhost", redis_client=fake)

    await bus.emit(
        {
            "type": "user.action",
            "user_id": "user-2",
            "action_type": "close",
            "timestamp": "2024-01-01T00:01:00",
            "data": {},
        }
    )

    metrics = await bus.get_metrics()
    assert metrics["events_in_stream"] == 1
    assert metrics["emitted_count"] == 1
