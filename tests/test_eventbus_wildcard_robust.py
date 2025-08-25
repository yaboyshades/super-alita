"""Test EventBus robust wildcard subscription functionality.

This test validates full wildcard ("*") subscription support using Redis/Memurai pattern subscriptions.
This robust implementation receives ALL published events, even from unsubscribed channels.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from src.core.event_bus import EventBus
from src.core.events import BaseEvent


@pytest.mark.asyncio
async def test_wildcard_subscription_robust_all_channels():
    """Test that wildcard subscriptions receive events from ALL channels, including unsubscribed ones.

    This validates that:
    1. Wildcard handlers receive events from ALL channels (subscribed and unsubscribed)
    2. Pattern subscription (psubscribe "*") captures all events
    3. Events are properly dispatched using both 'message' and 'pmessage' types
    4. Robust wildcard support works for any channel name
    """
    # Setup EventBus
    event_bus = EventBus(host="localhost", port=6379)

    # Mock handlers to track calls
    wildcard_handler = AsyncMock()
    specific_handler_a = AsyncMock()  # Only subscribe to one specific channel

    # Subscribe handlers
    await event_bus.subscribe("*", wildcard_handler)  # Wildcard pattern subscription
    await event_bus.subscribe(
        "known_channel", specific_handler_a
    )  # One specific channel

    # Start the EventBus to begin listening
    await event_bus.start()

    # Wait a moment for the event bus to fully start
    await asyncio.sleep(0.3)

    # Create test events for various channels (some subscribed, some not)
    test_events = [
        BaseEvent(
            event_type="known_channel",  # This channel has a specific subscriber
            source_plugin="test",
            metadata={"message": "Known Channel Event", "value": 1},
        ),
        BaseEvent(
            event_type="unknown_channel_1",  # This channel has NO specific subscribers
            source_plugin="test",
            metadata={"message": "Unknown Channel 1 Event", "value": 2},
        ),
        BaseEvent(
            event_type="unknown_channel_2",  # This channel has NO specific subscribers
            source_plugin="test",
            metadata={"message": "Unknown Channel 2 Event", "value": 3},
        ),
        BaseEvent(
            event_type="dynamic_channel",  # This channel has NO specific subscribers
            source_plugin="test",
            metadata={"message": "Dynamic Channel Event", "value": 4},
        ),
    ]

    # Publish events to different channels (some subscribed, some not)
    for event in test_events:
        await event_bus.publish(event)
        await asyncio.sleep(0.1)  # Small delay between publishes

    # Give time for async dispatch
    await asyncio.sleep(0.5)

    # Validate wildcard handler received ALL events (even from unsubscribed channels)
    assert (
        wildcard_handler.call_count == 4
    ), f"Wildcard handler should receive ALL events, got {wildcard_handler.call_count}"

    # Validate specific handler only received its channel event
    assert (
        specific_handler_a.call_count == 1
    ), "Known channel handler should receive 1 event"

    # Validate event content reached wildcard handler correctly
    wildcard_calls = [call_args[0][0] for call_args in wildcard_handler.call_args_list]
    assert len(wildcard_calls) == 4

    # Check that wildcard handler received events from all channels
    wildcard_messages = [
        event_call.metadata["message"] for event_call in wildcard_calls
    ]
    expected_messages = [
        "Known Channel Event",
        "Unknown Channel 1 Event",
        "Unknown Channel 2 Event",
        "Dynamic Channel Event",
    ]

    for expected_msg in expected_messages:
        assert (
            expected_msg in wildcard_messages
        ), f"Wildcard handler should receive '{expected_msg}'"

    # Check specific handler received correct event
    specific_event = specific_handler_a.call_args[0][0]
    assert specific_event.metadata["message"] == "Known Channel Event"

    # Cleanup
    await event_bus.shutdown()


@pytest.mark.asyncio
async def test_wildcard_mixed_with_specific_subscriptions():
    """Test wildcard and specific subscriptions working together robustly.

    This validates that:
    1. Wildcard receives events from ALL channels
    2. Specific handlers only receive their channel events
    3. No duplicate handling for channels with both wildcard and specific subscriptions
    4. Pattern subscription works alongside regular subscriptions
    """
    event_bus = EventBus(host="localhost", port=6379)

    # Mock handlers
    wildcard_handler = AsyncMock()
    specific_handler_alpha = AsyncMock()
    specific_handler_beta = AsyncMock()

    # Subscribe handlers
    await event_bus.subscribe("*", wildcard_handler)  # Pattern subscription
    await event_bus.subscribe("alpha_channel", specific_handler_alpha)
    await event_bus.subscribe("beta_channel", specific_handler_beta)

    # Start the EventBus
    await event_bus.start()
    await asyncio.sleep(0.3)

    # Create test events
    test_events = [
        BaseEvent(
            event_type="alpha_channel", source_plugin="test", metadata={"id": "alpha"}
        ),
        BaseEvent(
            event_type="beta_channel", source_plugin="test", metadata={"id": "beta"}
        ),
        BaseEvent(
            event_type="gamma_channel", source_plugin="test", metadata={"id": "gamma"}
        ),  # No specific subscriber
        BaseEvent(
            event_type="delta_channel", source_plugin="test", metadata={"id": "delta"}
        ),  # No specific subscriber
    ]

    # Publish all events
    for event in test_events:
        await event_bus.publish(event)
        await asyncio.sleep(0.1)

    # Give time for async dispatch
    await asyncio.sleep(0.5)

    # Validate wildcard received ALL events
    assert wildcard_handler.call_count == 4, "Wildcard should receive all 4 events"

    # Validate specific handlers received only their events
    assert (
        specific_handler_alpha.call_count == 1
    ), "Alpha handler should receive 1 event"
    assert specific_handler_beta.call_count == 1, "Beta handler should receive 1 event"

    # Validate event content
    alpha_event = specific_handler_alpha.call_args[0][0]
    assert alpha_event.metadata["id"] == "alpha"

    beta_event = specific_handler_beta.call_args[0][0]
    assert beta_event.metadata["id"] == "beta"

    # Validate wildcard received all events including unsubscribed channels
    wildcard_calls = [call_args[0][0] for call_args in wildcard_handler.call_args_list]
    wildcard_ids = [event_call.metadata["id"] for event_call in wildcard_calls]

    expected_ids = ["alpha", "beta", "gamma", "delta"]
    for expected_id in expected_ids:
        assert (
            expected_id in wildcard_ids
        ), f"Wildcard should receive event with id '{expected_id}'"

    # Cleanup
    await event_bus.shutdown()


@pytest.mark.asyncio
async def test_wildcard_only_no_specific_subscriptions():
    """Test wildcard subscription working without any specific channel subscriptions.

    This validates that:
    1. Pure wildcard subscription works (no specific channels)
    2. Pattern subscription receives events from any channel
    3. Events to completely unsubscribed channels are captured
    """
    event_bus = EventBus(host="localhost", port=6379)

    # Mock handler - only wildcard, no specific subscriptions
    wildcard_only_handler = AsyncMock()

    # Subscribe only to wildcard
    await event_bus.subscribe("*", wildcard_only_handler)

    # Start the EventBus
    await event_bus.start()
    await asyncio.sleep(0.3)

    # Create events for completely unsubscribed channels
    test_events = [
        BaseEvent(
            event_type="random_channel_1", source_plugin="test", metadata={"test": "1"}
        ),
        BaseEvent(
            event_type="random_channel_2", source_plugin="test", metadata={"test": "2"}
        ),
        BaseEvent(
            event_type="random_channel_3", source_plugin="test", metadata={"test": "3"}
        ),
    ]

    # Publish events
    for event in test_events:
        await event_bus.publish(event)
        await asyncio.sleep(0.1)

    # Give time for async dispatch
    await asyncio.sleep(0.5)

    # Validate wildcard received all events from unsubscribed channels
    assert (
        wildcard_only_handler.call_count == 3
    ), "Wildcard-only should receive all 3 events"

    # Validate event content
    wildcard_calls = [
        call_args[0][0] for call_args in wildcard_only_handler.call_args_list
    ]
    test_values = [event_call.metadata["test"] for event_call in wildcard_calls]

    assert "1" in test_values
    assert "2" in test_values
    assert "3" in test_values

    # Cleanup
    await event_bus.shutdown()
