"""Test EventBus robust wildcard subscription - TRUE unsubscribed channels.

This test validates that wildcard pattern subscriptions receive events from
channels that have NO specific subscribers at all.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from src.core.event_bus import EventBus
from src.core.events import BaseEvent


@pytest.mark.asyncio
async def test_wildcard_unsubscribed_channels_only():
    """Test wildcard receives events from completely unsubscribed channels.

    This is the true test of robust wildcard support - receiving events
    from channels that have NO specific subscribers.
    """

    # Create a fresh EventBus instance
    event_bus = EventBus(host="localhost", port=6379)

    # Only subscribe to wildcard - NO specific channel subscriptions
    wildcard_handler = AsyncMock()
    await event_bus.subscribe("*", wildcard_handler)

    try:
        # Start the EventBus
        await event_bus.start()
        await asyncio.sleep(0.2)

        # Publish events to channels that have NO subscribers
        unsubscribed_events = [
            BaseEvent(
                event_type="completely_unknown_channel_1",
                source_plugin="test",
                metadata={"id": "unsubscribed_1"},
            ),
            BaseEvent(
                event_type="never_subscribed_channel_2",
                source_plugin="test",
                metadata={"id": "unsubscribed_2"},
            ),
            BaseEvent(
                event_type="random_channel_xyz",
                source_plugin="test",
                metadata={"id": "unsubscribed_3"},
            ),
        ]

        # Publish to completely unsubscribed channels
        for event in unsubscribed_events:
            await event_bus.publish(event)
            await asyncio.sleep(0.1)

        # Wait for message processing
        await asyncio.sleep(0.4)

        # Wildcard handler should receive ALL events from unsubscribed channels
        assert (
            wildcard_handler.call_count == 3
        ), f"Wildcard should receive 3 events, got {wildcard_handler.call_count}"

        # Verify event content
        received_events = [
            call_args[0][0] for call_args in wildcard_handler.call_args_list
        ]
        received_ids = [event.metadata["id"] for event in received_events]

        assert "unsubscribed_1" in received_ids
        assert "unsubscribed_2" in received_ids
        assert "unsubscribed_3" in received_ids

        print(
            f"✅ Robust wildcard working: received events from {len(received_ids)} unsubscribed channels"
        )

    finally:
        await event_bus.shutdown()


@pytest.mark.asyncio
async def test_mixed_subscribed_and_unsubscribed():
    """Test wildcard with mix of subscribed and unsubscribed channels."""

    event_bus = EventBus(host="localhost", port=6379)

    # Handlers
    wildcard_handler = AsyncMock()
    specific_handler = AsyncMock()

    # Subscribe wildcard and one specific channel
    await event_bus.subscribe("*", wildcard_handler)
    await event_bus.subscribe("known_channel", specific_handler)

    try:
        await event_bus.start()
        await asyncio.sleep(0.2)

        # Mix of subscribed and unsubscribed channels
        events = [
            BaseEvent(
                event_type="known_channel",
                source_plugin="test",
                metadata={"type": "subscribed"},
            ),
            BaseEvent(
                event_type="unknown_channel_1",
                source_plugin="test",
                metadata={"type": "unsubscribed"},
            ),
            BaseEvent(
                event_type="unknown_channel_2",
                source_plugin="test",
                metadata={"type": "unsubscribed"},
            ),
        ]

        for event in events:
            await event_bus.publish(event)
            await asyncio.sleep(0.1)

        await asyncio.sleep(0.4)

        # Wildcard should receive ALL 3 events
        assert (
            wildcard_handler.call_count == 3
        ), f"Wildcard should receive all 3 events, got {wildcard_handler.call_count}"

        # Specific handler should receive only 1 event (its channel)
        assert (
            specific_handler.call_count == 1
        ), f"Specific handler should receive 1 event, got {specific_handler.call_count}"

        # Verify wildcard received both subscribed and unsubscribed events
        wildcard_events = [
            call_args[0][0] for call_args in wildcard_handler.call_args_list
        ]
        wildcard_types = [event.metadata["type"] for event in wildcard_events]

        assert wildcard_types.count("subscribed") == 1
        assert wildcard_types.count("unsubscribed") == 2

        print("✅ Mixed subscribed/unsubscribed channels working correctly")

    finally:
        await event_bus.shutdown()


if __name__ == "__main__":
    asyncio.run(test_wildcard_unsubscribed_channels_only())
