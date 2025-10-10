import time
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
async def test_eventbus_latency_under_target(
    fakeredis_async: aioredis.FakeRedis,
) -> None:
    bus = RedisEventBus(channel="unified.event.stream", redis=fakeredis_async)

    event = UnifiedEvent(
        topic="unified.performance.echo",
        payload={"message": "ping"},
        correlation_id=str(uuid4()),
        causation_id=None,
        metadata={"source": "performance-test"},
    )

    start = time.perf_counter()
    await bus.publish(event)
    delivered = await bus.subscribe(timeout=0.5)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert delivered.correlation_id == event.correlation_id
    assert elapsed_ms < 50, f"latency budget exceeded: {elapsed_ms:.2f}ms"
