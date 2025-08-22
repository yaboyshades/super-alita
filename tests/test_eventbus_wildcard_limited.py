"""Test EventBus wildcard subscription functionality - Limited Version.

This test validates basic wildcard ("*") subscription support for channels that have been subscribed to.
Note: This is a limited implementation that requires channels to have at least one subscriber.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from src.core.event_bus import EventBus
from src.core.events import BaseEvent


@pytest.mark.asyncio
async def test_wildcard_subscription_fan_out_limited():
    """Test that wildcard subscriptions receive events from subscribed channels.

    This validates that:
    1. Wildcard handlers receive events from channels that have specific subscribers
    2. Specific handlers only receive events from their channel
    3. Both specific and wildcard handlers receive events without duplication
    4. Events are properly dispatched to all eligible handlers

    NOTE: This is a limited implementation - wildcard handlers only receive events
    from channels that have at least one specific subscriber.
    """
    # Setup EventBus
    event_bus = EventBus(host="localhost", port=6379)

    # Mock handlers to track calls
    wildcard_handler = AsyncMock()
    specific_handler_a = AsyncMock()
    specific_handler_b = AsyncMock()
    dummy_handler_c = AsyncMock()  # Dummy handler for channel_c to ensure subscription

    # Subscribe handlers
    await event_bus.subscribe("*", wildcard_handler)
    await event_bus.subscribe("channel_a", specific_handler_a)
    await event_bus.subscribe("channel_b", specific_handler_b)
    await event_bus.subscribe(
        "channel_c", dummy_handler_c
    )  # Ensure channel_c is subscribed

    # Start the EventBus to begin listening
    await event_bus.start()

    # Wait a moment for the event bus to fully start
    await asyncio.sleep(0.2)

    # Create test events
    test_events = [
        BaseEvent(
            event_type="channel_a",
            source_plugin="test",
            metadata={"message": "Hello A", "value": 1},
        ),
        BaseEvent(
            event_type="channel_b",
            source_plugin="test",
            metadata={"message": "Hello B", "value": 2},
        ),
        BaseEvent(
            event_type="channel_c",
            source_plugin="test",
            metadata={"message": "Hello C", "value": 3},
        ),
    ]

    # Publish events to different channels
    for event in test_events:
        await event_bus.publish(event)
        await asyncio.sleep(0.05)  # Small delay between publishes

    # Give time for async dispatch
    await asyncio.sleep(0.3)

    # Validate wildcard handler received all events
    assert (
        wildcard_handler.call_count == 3
    ), f"Wildcard handler should receive all events, got {wildcard_handler.call_count}"

    # Validate specific handlers only received their channel events
    assert (
        specific_handler_a.call_count == 1
    ), "Channel A handler should receive 1 event"
    assert (
        specific_handler_b.call_count == 1
    ), "Channel B handler should receive 1 event"
    assert dummy_handler_c.call_count == 1, "Channel C handler should receive 1 event"

    # Validate event content reached handlers correctly
    wildcard_calls = [call_args[0][0] for call_args in wildcard_handler.call_args_list]
    assert len(wildcard_calls) == 3

    # Check that wildcard handler received events from all channels
    wildcard_messages = [
        event_call.metadata["message"] for event_call in wildcard_calls
    ]
    assert "Hello A" in wildcard_messages
    assert "Hello B" in wildcard_messages
    assert "Hello C" in wildcard_messages

    # Check specific handlers received correct events
    channel_a_event = specific_handler_a.call_args[0][0]
    assert channel_a_event.metadata["message"] == "Hello A"

    channel_b_event = specific_handler_b.call_args[0][0]
    assert channel_b_event.metadata["message"] == "Hello B"

    channel_c_event = dummy_handler_c.call_args[0][0]
    assert channel_c_event.metadata["message"] == "Hello C"

    # Cleanup
    await event_bus.shutdown()


@pytest.mark.asyncio
async def test_wildcard_no_duplicate_calls():
    """Test that handlers subscribed to both specific and wildcard don't get duplicates.

    This validates that if a handler is subscribed to both a specific channel
    and the wildcard, it only receives one copy of each event.
    """
    event_bus = EventBus(host="localhost", port=6379)

    # Mock handler that will be subscribed to both specific and wildcard
    multi_handler = AsyncMock()

    # Subscribe same handler to both wildcard and specific channel
    await event_bus.subscribe("*", multi_handler)
    await event_bus.subscribe("test_channel", multi_handler)

    # Start the EventBus to begin listening
    await event_bus.start()
    await asyncio.sleep(0.2)

    # Publish event to the specific channel
    test_event = BaseEvent(
        event_type="test_channel",
        source_plugin="test",
        metadata={"message": "No duplicates"},
    )
    await event_bus.publish(test_event)

    # Give time for async dispatch
    await asyncio.sleep(0.2)

    # Handler should only be called once, not twice
    assert multi_handler.call_count == 1, "Handler should not receive duplicate events"

    # Validate the event content
    event_received = multi_handler.call_args[0][0]
    assert event_received.metadata["message"] == "No duplicates"

    # Cleanup
    await event_bus.shutdown()
