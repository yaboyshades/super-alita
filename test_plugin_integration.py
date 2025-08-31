#!/usr/bin/env python3
"""Test AutogenCreatorPlugin integration."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from reug_runtime.event_bus import make_event_bus  # noqa: E402
from src.plugins.autogen_creator_plugin import AutogenCreatorPlugin  # noqa: E402


async def test_plugin_setup():
    """Test that our plugin can be set up correctly."""
    try:
        plugin = AutogenCreatorPlugin()
        bus = make_event_bus()

        # Test setup
        await plugin.setup(bus, None, {})
        print("✓ Plugin setup successful")

        # Test start
        await plugin.start()
        print("✓ Plugin start successful")

        # Test that it's running
        assert plugin.is_running
        print("✓ Plugin is running")

        # Test stop
        await plugin.stop()
        print("✓ Plugin stop successful")

        return True
    except Exception as e:
        print(f"✗ Plugin test failed: {e}")
        return False


async def main():
    """Main test."""
    print("Testing AutogenCreatorPlugin integration...")
    success = await test_plugin_setup()
    print(f"Test result: {'PASS' if success else 'FAIL'}")


if __name__ == "__main__":
    asyncio.run(main())
