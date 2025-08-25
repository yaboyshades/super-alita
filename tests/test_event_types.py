"""
Tests for event type registry functionality.
"""

from unittest.mock import patch

from src.core.event_types import (
    EVENT_REGISTRY,
    EventDescriptor,
    deprecate_event,
    get_event,
    list_events,
    register_event,
    validate_event_payload,
)


class TestEventDescriptor:
    """Test EventDescriptor functionality."""

    def test_create_minimal_descriptor(self):
        """Test creating a minimal event descriptor."""
        descriptor = EventDescriptor(name="test_event", version=1)

        assert descriptor.name == "test_event"
        assert descriptor.version == 1
        assert descriptor.schema == {}
        assert descriptor.description == ""
        assert descriptor.deprecated is False
        assert descriptor.successor is None

    def test_create_complete_descriptor(self):
        """Test creating a complete event descriptor."""
        schema = {"type": "object", "required": ["field1"]}
        descriptor = EventDescriptor(
            name="test_event",
            version=2,
            schema=schema,
            description="Test event description",
            deprecated=True,
            successor="new_event",
        )

        assert descriptor.name == "test_event"
        assert descriptor.version == 2
        assert descriptor.schema == schema
        assert descriptor.description == "Test event description"
        assert descriptor.deprecated is True
        assert descriptor.successor == "new_event"


class TestEventRegistration:
    """Test event registration functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        EVENT_REGISTRY.clear()

    def test_register_new_event(self):
        """Test registering a new event type."""
        descriptor = EventDescriptor(name="new_event", version=1)

        result = register_event(descriptor)

        assert result is True
        assert "new_event" in EVENT_REGISTRY
        assert EVENT_REGISTRY["new_event"] == descriptor

    def test_register_upgrade_event(self):
        """Test upgrading an existing event type."""
        old_descriptor = EventDescriptor(name="test_event", version=1)
        new_descriptor = EventDescriptor(name="test_event", version=2)

        register_event(old_descriptor)
        result = register_event(new_descriptor)

        assert result is True
        assert EVENT_REGISTRY["test_event"] == new_descriptor
        assert EVENT_REGISTRY["test_event"].version == 2

    def test_register_same_version(self):
        """Test registering same version (idempotent)."""
        descriptor1 = EventDescriptor(name="test_event", version=1, description="First")
        descriptor2 = EventDescriptor(
            name="test_event", version=1, description="Second"
        )

        register_event(descriptor1)
        result = register_event(descriptor2)

        assert result is True  # Updated because different
        assert EVENT_REGISTRY["test_event"].description == "Second"

    def test_register_same_descriptor(self):
        """Test registering identical descriptor (no-op)."""
        descriptor = EventDescriptor(name="test_event", version=1)

        register_event(descriptor)
        result = register_event(descriptor)

        assert result is False  # No change
        assert EVENT_REGISTRY["test_event"] == descriptor

    def test_register_downgrade_ignored(self):
        """Test that downgrade attempts are ignored."""
        new_descriptor = EventDescriptor(name="test_event", version=2)
        old_descriptor = EventDescriptor(name="test_event", version=1)

        register_event(new_descriptor)

        with patch("src.core.event_types.logger") as mock_logger:
            result = register_event(old_descriptor)

            assert result is False
            assert EVENT_REGISTRY["test_event"] == new_descriptor
            mock_logger.warning.assert_called_once()


class TestEventRetrieval:
    """Test event retrieval functionality."""

    def setup_method(self):
        """Clear registry and add test events."""
        EVENT_REGISTRY.clear()

        register_event(EventDescriptor(name="event1", version=1))
        register_event(EventDescriptor(name="event2", version=2))

    def test_get_existing_event(self):
        """Test getting an existing event."""
        descriptor = get_event("event1")

        assert descriptor is not None
        assert descriptor.name == "event1"
        assert descriptor.version == 1

    def test_get_nonexistent_event(self):
        """Test getting a non-existent event."""
        descriptor = get_event("nonexistent")

        assert descriptor is None

    def test_list_all_events(self):
        """Test listing all events."""
        events = list_events()

        assert len(events) == 2
        assert "event1" in events
        assert "event2" in events
        assert events["event1"].version == 1
        assert events["event2"].version == 2

    def test_list_events_is_copy(self):
        """Test that list_events returns a copy."""
        events = list_events()
        events["new_event"] = EventDescriptor(name="new_event", version=1)

        # Should not affect original registry
        assert "new_event" not in EVENT_REGISTRY


class TestEventValidation:
    """Test event payload validation."""

    def setup_method(self):
        """Set up test events with schemas."""
        EVENT_REGISTRY.clear()

        # Event with schema
        schema = {
            "type": "object",
            "required": ["field1", "field2"],
            "properties": {"field1": {"type": "string"}, "field2": {"type": "number"}},
        }
        register_event(EventDescriptor(name="with_schema", version=1, schema=schema))

        # Event without schema
        register_event(EventDescriptor(name="no_schema", version=1))

    def test_validate_valid_payload(self):
        """Test validating a valid payload."""
        payload = {"field1": "test", "field2": 42, "extra": "allowed"}

        result = validate_event_payload("with_schema", payload)

        assert result is True

    def test_validate_missing_required_field(self):
        """Test validating payload with missing required field."""
        payload = {"field1": "test"}  # Missing field2

        with patch("src.core.event_types.logger") as mock_logger:
            result = validate_event_payload("with_schema", payload)

            assert result is False
            mock_logger.warning.assert_called_once()

    def test_validate_no_schema(self):
        """Test validating payload for event without schema."""
        payload = {"anything": "goes"}

        result = validate_event_payload("no_schema", payload)

        assert result is True

    def test_validate_nonexistent_event(self):
        """Test validating payload for non-existent event."""
        payload = {"field": "value"}

        result = validate_event_payload("nonexistent", payload)

        assert result is True  # No schema to validate against


class TestEventDeprecation:
    """Test event deprecation functionality."""

    def setup_method(self):
        """Set up test events."""
        EVENT_REGISTRY.clear()
        register_event(EventDescriptor(name="old_event", version=1))

    def test_deprecate_existing_event(self):
        """Test deprecating an existing event."""
        result = deprecate_event("old_event", "new_event")

        assert result is True

        descriptor = get_event("old_event")
        assert descriptor.deprecated is True
        assert descriptor.successor == "new_event"

    def test_deprecate_without_successor(self):
        """Test deprecating an event without successor."""
        result = deprecate_event("old_event")

        assert result is True

        descriptor = get_event("old_event")
        assert descriptor.deprecated is True
        assert descriptor.successor is None

    def test_deprecate_nonexistent_event(self):
        """Test deprecating a non-existent event."""
        result = deprecate_event("nonexistent")

        assert result is False


class TestCoreSeedEvents:
    """Test that core events are properly seeded."""

    def test_core_events_present(self):
        """Test that core events are registered on import."""
        # Import should trigger seeding
        from src.core.event_types import EVENT_REGISTRY

        expected_events = [
            "tool_call",
            "tool_result",
            "goal_received",
            "plan_created",
            "memory_store",
            "plugin_loaded",
            "error_occurred",
        ]

        for event_name in expected_events:
            assert event_name in EVENT_REGISTRY, f"Core event {event_name} not found"

    def test_tool_call_schema(self):
        """Test tool_call event has proper schema."""
        from src.core.event_types import EVENT_REGISTRY

        tool_call = EVENT_REGISTRY["tool_call"]
        assert tool_call.version == 2
        assert "required" in tool_call.schema

        required_fields = tool_call.schema["required"]
        expected_required = [
            "tool_call_id",
            "tool_name",
            "arguments",
            "conversation_id",
        ]

        for req_field in expected_required:
            assert req_field in required_fields

    def test_tool_result_schema(self):
        """Test tool_result event has proper schema."""
        from src.core.event_types import EVENT_REGISTRY

        tool_result = EVENT_REGISTRY["tool_result"]
        assert tool_result.version == 2
        assert "required" in tool_result.schema

        required_fields = tool_result.schema["required"]
        expected_required = ["tool_call_id", "success"]

        for req_field in expected_required:
            assert req_field in required_fields
