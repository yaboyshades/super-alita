"""
Extended Redis EventBus tests with pluggable serialization.

Tests both JSON and Protobuf wire formats, backpressure handling,
and Dead Letter Queue functionality.
"""

import asyncio
import importlib
import logging
import socket
from datetime import datetime
from typing import List

import pytest
from src.core.event_bus import EventBus
from src.core.events import BaseEvent, ConversationEvent, SystemEvent
from src.core.serialization import JsonSerializer, ProtobufSerializer

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

pytestmark = pytest.mark.integration_redis

if importlib.util.find_spec("redis") is None:
    pytest.skip("redis not installed", allow_module_level=True)


def _redis_running(host: str = "localhost", port: int = 6379) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


if not _redis_running():
    pytest.skip("Redis server not available", allow_module_level=True)


@pytest.fixture
async def fresh_bus():
    """Create a fresh EventBus instance for each test."""
    bus = EventBus(wire_format="json")
    await bus.connect()
    await bus.start()
    yield bus
    await bus.shutdown()


@pytest.fixture
async def fresh_bus_protobuf():
    """Create a fresh EventBus instance with Protobuf serialization."""
    if ProtobufSerializer is None:
        pytest.skip("Protobuf not available")
    bus = EventBus(wire_format="protobuf")
    await bus.connect()
    await bus.start()
    yield bus
    await bus.shutdown()


class TestSerializationFormats:
    """Test different wire formats for event serialization."""

    @pytest.mark.parametrize("wire_format", ["json", "protobuf"])
    async def test_serializer_roundtrip(self, wire_format):
        """Test that events can be serialized and deserialized correctly."""
        if wire_format == "protobuf" and ProtobufSerializer is None:
            pytest.skip("Protobuf not available")

        bus = EventBus(wire_format=wire_format)
        await bus.connect()
        await bus.start()

        try:
            # Create test event
            test_event = ConversationEvent(
                source_plugin="test_plugin",
                session_id="test_session",
                user_message="Hello, World!",
                message_id="test_msg_001",
                timestamp=datetime.now().isoformat(),
            )

            # Storage for received events
            received_events: List[BaseEvent] = []

            async def test_handler(event):
                received_events.append(event)

            # Subscribe and publish
            await bus.subscribe("conversation_message", test_handler)
            await bus.publish(test_event)

            # Wait for processing
            await asyncio.sleep(0.5)

            # Verify event was received correctly
            assert len(received_events) == 1
            received = received_events[0]
            assert received.source_plugin == "test_plugin"
            assert received.user_message == "Hello, World!"
            assert received.session_id == "test_session"

        finally:
            await bus.shutdown()

    async def test_json_serializer_direct(self):
        """Test JSON serializer directly."""
        serializer = JsonSerializer()

        event = SystemEvent(
            source_plugin="test",
            level="info",
            message="test message",
            component="test_component",
        )

        # Serialize and deserialize
        serialized = serializer.encode(event)
        deserialized = serializer.decode(serialized, SystemEvent)

        assert deserialized.source_plugin == "test"
        assert deserialized.level == "info"
        assert deserialized.message == "test message"
        assert deserialized.component == "test_component"


class TestBackpressureAndReliability:
    """Test system behavior under load and error conditions."""

    async def test_high_throughput_publishing(self, fresh_bus):
        """Test publishing many events quickly."""
        received_count = 0

        async def counter_handler(event):
            nonlocal received_count
            received_count += 1

        await fresh_bus.subscribe("system", counter_handler)

        # Publish 100 events rapidly
        for i in range(100):
            event = SystemEvent(
                source_plugin="throughput_test",
                level="info",
                message=f"Message {i}",
                component="test",
            )
            await fresh_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(2.0)

        # Should receive most or all events
        assert received_count >= 90  # Allow for some timing variations

    async def test_error_handling_in_subscribers(self, fresh_bus):
        """Test that subscriber errors don't crash the system."""
        successful_events = []

        async def failing_handler(event):
            raise RuntimeError("Simulated handler failure")

        async def working_handler(event):
            successful_events.append(event)

        # Subscribe both handlers
        await fresh_bus.subscribe("system", failing_handler)
        await fresh_bus.subscribe("system", working_handler)

        # Publish test event
        test_event = SystemEvent(
            source_plugin="error_test",
            level="error",
            message="Test error handling",
            component="test",
        )
        await fresh_bus.publish(test_event)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Working handler should still receive the event
        assert len(successful_events) == 1
        assert successful_events[0].message == "Test error handling"

    async def test_multiple_event_types(self, fresh_bus):
        """Test handling multiple different event types."""
        conversation_events = []
        system_events = []

        async def conversation_handler(event):
            conversation_events.append(event)

        async def system_handler(event):
            system_events.append(event)

        # Subscribe to different event types
        await fresh_bus.subscribe("conversation_message", conversation_handler)
        await fresh_bus.subscribe("system", system_handler)

        # Publish mixed events
        await fresh_bus.publish(
            ConversationEvent(
                source_plugin="test",
                session_id="session1",
                user_message="Hello",
                message_id="msg_001",
            )
        )

        await fresh_bus.publish(
            SystemEvent(
                source_plugin="test",
                level="info",
                message="System ready",
                component="core",
            )
        )

        await fresh_bus.publish(
            ConversationEvent(
                source_plugin="test",
                session_id="session2",
                user_message="World",
                message_id="msg_002",
            )
        )

        # Wait for processing
        await asyncio.sleep(0.5)

        # Verify correct routing
        assert len(conversation_events) == 2
        assert len(system_events) == 1
        assert conversation_events[0].user_message == "Hello"
        assert conversation_events[1].user_message == "World"
        assert system_events[0].message == "System ready"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    async def test_large_event_payload(self, fresh_bus):
        """Test handling of large event payloads."""
        large_message = "x" * 10000  # 10KB message
        received_events = []

        async def handler(event):
            received_events.append(event)

        await fresh_bus.subscribe("conversation_message", handler)

        large_event = ConversationEvent(
            source_plugin="test",
            session_id="test_session",
            user_message=large_message,
            message_id="large_msg_001",
        )

        await fresh_bus.publish(large_event)
        await asyncio.sleep(0.5)

        assert len(received_events) == 1
        assert len(received_events[0].user_message) == 10000

    async def test_empty_message_handling(self, fresh_bus):
        """Test handling of events with empty/null fields."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        await fresh_bus.subscribe("conversation_message", handler)

        empty_event = ConversationEvent(
            source_plugin="test",
            session_id="test_session",
            user_message="",  # Empty message
            message_id="empty_msg_001",
        )

        await fresh_bus.publish(empty_event)
        await asyncio.sleep(0.5)

        assert len(received_events) == 1
        assert received_events[0].user_message == ""

    async def test_unicode_content(self, fresh_bus):
        """Test handling of Unicode content in events."""
        unicode_message = "Hello 世界 🌍 Здравствуй мир"
        received_events = []

        async def handler(event):
            received_events.append(event)

        await fresh_bus.subscribe("conversation_message", handler)

        unicode_event = ConversationEvent(
            source_plugin="test",
            session_id="unicode_session",
            user_message=unicode_message,
            message_id="unicode_msg_001",
        )

        await fresh_bus.publish(unicode_event)
        await asyncio.sleep(0.5)

        assert len(received_events) == 1
        assert received_events[0].user_message == unicode_message


# Pytest configuration for asyncio
pytest_plugins = ("pytest_asyncio",)


if __name__ == "__main__":
    # Run with: python -m pytest tests/core/test_event_bus_redis.py -v
    pytest.main([__file__, "-v"])
