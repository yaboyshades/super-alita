import asyncio
from collections.abc import AsyncIterator

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
async def test_subscribe_delivers_published_event(
    fakeredis_async: aioredis.FakeRedis,
) -> None:
    bus = RedisEventBus(channel="unified.event.stream", redis=fakeredis_async)

    async def producer() -> None:
        await bus.publish(
            UnifiedEvent(
                topic="unified.event.sample",
                payload={"message": "world"},
                correlation_id="corr-456",
                causation_id=None,
                metadata={"source": "contract-test"},
            )
        )

    consume_task = asyncio.create_task(bus.subscribe())
    await producer()

    delivered = await consume_task
    assert delivered.topic == "unified.event.sample"
    assert delivered.payload["message"] == "world"
    assert delivered.correlation_id == "corr-456"


@pytest.mark.asyncio
async def test_subscribe_respects_consumer_timeout(
    fakeredis_async: aioredis.FakeRedis,
) -> None:
    bus = RedisEventBus(channel="unified.event.stream", redis=fakeredis_async)

    with pytest.raises(TimeoutError):
        await bus.subscribe(timeout=0.01)
