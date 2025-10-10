from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from fakeredis import aioredis

from src.orchestrator.eventbus import RedisEventBus, UnifiedEvent


@pytest.fixture
async def fakeredis_async() -> AsyncIterator[aioredis.FakeRedis]:
    client = aioredis.FakeRedis()
    try:
        yield client
    finally:
        await client.flushall()
        await client.close()


@pytest.mark.asyncio
async def test_publish_persists_event_payload(
    fakeredis_async: aioredis.FakeRedis,
) -> None:
    bus = RedisEventBus(channel="unified.event.stream", redis=fakeredis_async)
    event = UnifiedEvent(
        topic="unified.event.sample",
        payload={"message": "hello"},
        correlation_id=str(uuid4()),
        causation_id=None,
        metadata={"source": "contract-test"},
    )

    await bus.publish(event)

    stored = await fakeredis_async.lpop("unified.event.stream")
    assert stored is not None, "publish must push payload to redis list"
    assert event.correlation_id in stored.decode("utf-8")


@pytest.mark.asyncio
async def test_publish_emits_metrics_hook(
    fakeredis_async: aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics_calls: list[dict[str, int]] = []

    async def fake_emit(metrics: dict[str, int]) -> None:
        metrics_calls.append(metrics)

    bus = RedisEventBus(channel="unified.event.stream", redis=fakeredis_async)
    monkeypatch.setattr(bus, "emit_metrics", fake_emit)

    await bus.publish(
        UnifiedEvent(
            topic="unified.event.sample",
            payload={"count": 1},
            correlation_id="corr-123",
            causation_id=None,
            metadata={"source": "contract-test"},
        )
    )

    assert metrics_calls, "publish must emit metrics after persisting"
    assert metrics_calls[-1]["events_in"] >= 1
