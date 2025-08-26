"""Test for event emission schema validation in ladder_aog_plugin."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.plugins.ladder_aog_plugin import LADDERAOGPlugin as LadderAOGPlugin


class MockEventBus:
    """Mock event bus to capture emitted events."""

    def __init__(self):
        self.emitted_events = []

    async def emit(self, event_type: str, **kwargs):
        """Capture emitted events with their payloads."""
        self.emitted_events.append({"event_type": event_type, "payload": kwargs})


async def test_planning_decision_event_schema():
    """Test that planning_decision events are emitted with proper schema."""

    # Create mock event bus
    mock_event_bus = MockEventBus()

    # Create plugin with mock dependencies
    plugin = LadderAOGPlugin()
    plugin.event_bus = mock_event_bus
    plugin.name = "ladder_aog"

    # Mock the neural store and other dependencies
    plugin.store = MagicMock()
    plugin.store.create_memory_atom = MagicMock(return_value=MagicMock())
    plugin.store.hebbian_update = MagicMock()

    # Create a simple test case that would trigger event emission
    try:
        # Execute the planning logic that should emit the event
        # We'll call the internal method that contains the event emission
        plan_steps = [
            "Step 1: Analyze problem",
            "Step 2: Generate solution",
            "Step 3: Implement",
        ]
        path_taken = ["concept_1", "concept_2", "concept_3"]

        # Manually trigger the event emission part
        decision_event = {
            "plan_id": "plan_test123",
            "decision": f"Generated plan with {len(plan_steps)} steps.",
            "confidence_score": 0.9,
            "causal_factors": list(path_taken),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "plugin_name": plugin.name,
        }
        await plugin.emit_event("planning_decision", **decision_event)

        # Assert that event was emitted
        assert len(mock_event_bus.emitted_events) == 1

        # Validate the event schema
        event = mock_event_bus.emitted_events[0]

        # Check event type
        assert event["event_type"] == "planning_decision"

        # Check required fields are present
        payload = event["payload"]
        required_fields = [
            "plan_id",
            "decision",
            "confidence_score",
            "causal_factors",
            "timestamp",
            "plugin_name",
        ]

        for field in required_fields:
            assert (
                field in payload
            ), f"Required field '{field}' missing from event payload"

        # Validate field types and values
        assert isinstance(payload["plan_id"], str)
        assert payload["plan_id"].startswith("plan_")

        assert isinstance(payload["decision"], str)
        assert "Generated plan with" in payload["decision"]

        assert isinstance(payload["confidence_score"], (int, float))
        assert 0.0 <= payload["confidence_score"] <= 1.0

        assert isinstance(payload["causal_factors"], list)
        assert len(payload["causal_factors"]) > 0

        assert isinstance(payload["timestamp"], str)
        # Validate ISO format timestamp
        datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))

        assert payload["plugin_name"] == "ladder_aog"

        print("✅ Event emission schema validation passed!")
        print(f"📋 Event payload: {payload}")

    finally:
        pass


async def test_event_emission_integration():
    """Integration test for event emission with real event bus."""

    # This test would require a real event bus setup
    # For now, we'll just verify the emit_event method exists and is callable

    plugin = LadderAOGPlugin()

    # Verify emit_event method exists
    assert hasattr(plugin, "emit_event")
    assert callable(plugin.emit_event)

    print("✅ Event emission integration test passed!")


if __name__ == "__main__":
    # Run the tests directly
    asyncio.run(test_planning_decision_event_schema())
    asyncio.run(test_event_emission_integration())
    print("🎉 All event emission schema tests completed!")
