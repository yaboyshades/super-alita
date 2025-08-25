"""Test EventBus pattern subscription functionality - focused test.

This test validates that our pattern subscription implementation works correctly.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from src.core.event_bus import EventBus
from src.core.events import BaseEvent


@pytest.mark.asyncio
async def test_pattern_subscription_basic():
    """Test basic pattern subscription functionality."""

    # Create a fresh EventBus instance
    event_bus = EventBus(host="localhost", port=6379)

    # Mock handler
    wildcard_handler = AsyncMock()

    try:
        # Subscribe to wildcard pattern
        await event_bus.subscribe("*", wildcard_handler)

        # Start the EventBus
        await event_bus.start()

        # Wait for startup
        await asyncio.sleep(0.2)

        # Create and publish a test event
        test_event = BaseEvent(
            event_type="test_pattern_channel",
            source_plugin="test",
            metadata={"message": "Pattern test"},
        )

        await event_bus.publish(test_event)

        # Wait for message processing
        await asyncio.sleep(0.3)

        # Verify the wildcard handler was called
        assert (
            wildcard_handler.call_count >= 1
        ), f"Wildcard handler should be called, got {wildcard_handler.call_count} calls"

        # Verify the event content
        if wildcard_handler.call_count > 0:
            received_event = wildcard_handler.call_args_list[0][0][0]
            assert received_event.metadata["message"] == "Pattern test"
            print(
                f"✅ Pattern subscription working: received {wildcard_handler.call_count} calls"
            )

    finally:
        # Always cleanup
        try:
            await event_bus.shutdown()
        except Exception as e:
            print(f"Cleanup error (expected): {e}")


if __name__ == "__main__":
    asyncio.run(test_pattern_subscription_basic())
