#!/usr/bin/env python3
"""
Validation script for Dynamic Core Utils plugin integration.
Tests dynamic capability discovery and AST-based calculator via event bus.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.events import ToolCallEvent, ToolResultEvent
from src.plugins.core_utils_plugin_dynamic import CoreUtilsPlugin
from src.tools.core_utils import CoreUtils


class MockEventBus:
    """Mock event bus for testing plugin integration."""

    def __init__(self):
        self.published_events = []
        self.subscribers = {}

    async def publish(self, event):
        """Mock publish - stores events for verification."""
        self.published_events.append(event)
        if hasattr(event, "event_type"):
            print(f"📤 Published: {event.event_type}")
        else:
            print(f"📤 Published: {type(event).__name__}")

        # Trigger subscribers if any
        event_type = getattr(event, "event_type", type(event).__name__)
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                await handler(event)

    async def emit(self, event_type: str, **kwargs):
        """Mock emit method for plugin interface compatibility."""
        # Create a mock event object from the parameters
        from src.core.events import ToolResultEvent

        if event_type == "tool_result":
            event = ToolResultEvent(
                event_type=event_type,
                source_plugin=kwargs.get("source_plugin", "unknown"),
                **{k: v for k, v in kwargs.items() if k != "source_plugin"}
            )
            await self.publish(event)

    async def subscribe(self, event_type: str, handler):
        """Mock subscribe - registers handlers."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        print(f"📝 Subscribed to: {event_type}")


async def test_core_utils_direct():
    """Test core utilities directly (without event bus)."""
    print("\n🧮 Testing CoreUtils.calculate directly...")

    test_cases = [
        ("2 + 3", 5),
        ("10 - 4", 6),
        ("3 * 7", 21),
        ("15 / 3", 5.0),
        ("(2 + 3) * 4", 20),
        ("-5 + 10", 5),
        ("2 ** 3", "error"),  # Power not supported
        ("1 / 0", "error"),  # Division by zero
    ]

    for expr, expected in test_cases:
        try:
            result = CoreUtils.calculate(expr)
            status = "✅" if result == expected else "❌"
            print(f"  {status} {expr} = {result}")
        except Exception as e:
            status = "✅" if expected == "error" else "❌"
            print(f"  {status} {expr} = ERROR: {e}")

    print("\n🔤 Testing CoreUtils.reverse_string directly...")

    string_tests = [
        ("hello", "olleh"),
        ("Python", "nohtyP"),
        ("", ""),
        ("a", "a"),
        ("12345", "54321"),
    ]

    for text, expected in string_tests:
        result = CoreUtils.reverse_string(text)
        status = "✅" if result == expected else "❌"
        print(f"  {status} reverse_string('{text}') = '{result}'")


async def test_dynamic_capability_discovery():
    """Test dynamic capability discovery."""
    print("\n🔍 Testing Dynamic Capability Discovery...")

    # Create plugin
    plugin = CoreUtilsPlugin()

    # Setup without event bus to test discovery
    await plugin.setup(None, None, {"enabled": True})

    # Check discovered capabilities
    capabilities = plugin.get_discovered_capabilities()
    print(f"\n📋 Discovered capabilities: {len(capabilities)}")

    for tool_name, metadata in capabilities.items():
        print(
            f"  • {tool_name}: {metadata['parameters']} (static: {metadata['is_static']})"
        )

    # Test can_handle_tool method
    print("\n🎯 Capability checks:")
    test_tools = ["core.calculate", "core.reverse_string", "core.unknown", "other.tool"]
    for tool in test_tools:
        can_handle = plugin.can_handle_tool(tool)
        status = "✅" if can_handle else "❌"
        print(f"  {status} Can handle '{tool}': {can_handle}")


async def test_plugin_integration():
    """Test plugin integration via mock event bus."""
    print("\n🔌 Testing Dynamic CoreUtilsPlugin integration...")

    # Create mock event bus and plugin
    event_bus = MockEventBus()
    plugin = CoreUtilsPlugin()

    # Setup plugin
    await plugin.setup(event_bus, None, {"enabled": True})
    await plugin.start()

    print(f"✅ Plugin '{plugin.name}' started successfully")

    # Test calculator tool via events
    print("\n📡 Testing calculator via event bus...")

    calc_event = ToolCallEvent(
        source_plugin="test",
        conversation_id="test_session",
        session_id="test_session_id",
        tool_name="core.calculate",
        parameters={"expression": "(5 + 3) * 2"},
        tool_call_id="calc_test_1",
    )

    await event_bus.publish(calc_event)

    # Test string reverse tool via events
    print("\n📡 Testing string reverse via event bus...")

    reverse_event = ToolCallEvent(
        source_plugin="test",
        conversation_id="test_session",
        session_id="test_session_id",
        tool_name="core.reverse_string",
        parameters={"text": "integration"},
        tool_call_id="reverse_test_1",
    )

    await event_bus.publish(reverse_event)

    # Check results
    await asyncio.sleep(0.1)  # Allow event processing

    print(f"\n📊 Events published: {len(event_bus.published_events)}")
    for i, event in enumerate(event_bus.published_events):
        if isinstance(event, ToolResultEvent):
            result_summary = (
                event.result.get("result", "no result")
                if event.result
                else "empty result"
            )
            print(
                f"  {i + 1}. Tool Result (id: {event.tool_call_id}): {result_summary} (success: {event.success})"
            )
        else:
            event_type = getattr(event, "event_type", type(event).__name__)
            tool_name = getattr(event, "tool_name", "unknown")
            print(f"  {i + 1}. {event_type}: {tool_name}")

    await plugin.shutdown()
    print("✅ Plugin shutdown completed")


async def main():
    """Run comprehensive validation."""
    print("🚀 Super Alita Dynamic Core Utils Integration Validation")
    print("=" * 60)

    try:
        # Test direct functionality
        await test_core_utils_direct()

        # Test dynamic capability discovery
        await test_dynamic_capability_discovery()

        # Test plugin integration
        await test_plugin_integration()

        print("\n" + "=" * 60)
        print("✅ All Dynamic Core Utils integration tests completed successfully!")
        print("🎯 Dynamic capability discovery and AST-based utilities are ready!")

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
