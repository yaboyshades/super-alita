import asyncio

import pytest

from src.reug_runtime.event_bus import ProductionEventBus


@pytest.mark.asyncio
async def test_clarification_emitted_when_patterns_align(tmp_path):
    bus = ProductionEventBus(str(tmp_path))
    captured: list[dict] = []
    trigger = asyncio.Event()

    async def handler(event: dict) -> None:
        captured.append(event)
        trigger.set()

    await bus.subscribe("clarification_opportunity", handler)

    events = [
        {
            "type": "user_message",
            "session_id": "session-1",
            "timestamp": 1.0,
            "data": {"text": "The workflow is still broken."},
        },
        {
            "type": "user_message",
            "session_id": "session-1",
            "timestamp": 2.0,
            "data": {"text": "The workflow is still broken."},
        },
    ]

    for event in events:
        await bus.publish(event)

    await asyncio.wait_for(bus.queues["medium"].join(), timeout=1.0)
    await asyncio.wait_for(trigger.wait(), timeout=1.0)

    assert len(captured) == 1
    clarification = captured[0]
    assert clarification["event_type"] == "clarification_opportunity"
    context = clarification["context"]
    assert context["repeated_text"].lower() == "the workflow is still broken."
    assert context["frustration_reason"].startswith("keyword:")
    assert len(context["history"]) == 2


@pytest.mark.asyncio
async def test_no_clarification_when_only_frustration(tmp_path):
    bus = ProductionEventBus(str(tmp_path))
    captured: list[dict] = []

    async def handler(event: dict) -> None:
        captured.append(event)

    await bus.subscribe("clarification_opportunity", handler)

    events = [
        {
            "type": "user_message",
            "session_id": "session-2",
            "timestamp": 1.0,
            "data": {"text": "This is so frustrating!"},
        },
        {
            "type": "user_message",
            "session_id": "session-2",
            "timestamp": 2.0,
            "data": {"text": "Can we fix it?"},
        },
    ]

    for event in events:
        await bus.publish(event)

    await asyncio.wait_for(bus.queues["medium"].join(), timeout=1.0)
    await asyncio.sleep(0.05)

    assert captured == []


@pytest.mark.asyncio
async def test_no_clarification_when_only_repetition(tmp_path):
    bus = ProductionEventBus(str(tmp_path))
    captured: list[dict] = []

    async def handler(event: dict) -> None:
        captured.append(event)

    await bus.subscribe("clarification_opportunity", handler)

    events = [
        {
            "type": "user_message",
            "session_id": "session-3",
            "timestamp": 1.0,
            "data": {"text": "Please rerun the build."},
        },
        {
            "type": "user_message",
            "session_id": "session-3",
            "timestamp": 2.0,
            "data": {"text": "Please rerun the build."},
        },
        {
            "type": "user_message",
            "session_id": "session-3",
            "timestamp": 3.0,
            "data": {"text": "Please rerun the build."},
        },
    ]

    for event in events:
        await bus.publish(event)

    await asyncio.wait_for(bus.queues["medium"].join(), timeout=1.0)
    await asyncio.sleep(0.05)

    assert captured == []
