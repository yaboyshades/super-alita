#!/usr/bin/env python3
"""
Test the refactored autogen pipeline with native DeepCode integration.
"""

import asyncio

from src.core.event_bus import EventBus
from src.native_deepcode_api import get_native_deepcode_api
from src.pipelines.autogen_pipeline import autogen_any


async def test_native_autogen():
    """Test autogen pipeline with native DeepCode API."""
    print("Testing native autogen pipeline...")

    # Create native API and event bus
    api = get_native_deepcode_api()
    bus = EventBus()

    # Test a simple request that should trigger api_client capability
    result = await autogen_any(
        description="I need an API client to call an external REST API",
        repo_path=".",
        iterations=1,
        event_bus=bus,
        api=api,
    )

    print(f"Result: {result}")
    return result


if __name__ == "__main__":
    asyncio.run(test_native_autogen())
