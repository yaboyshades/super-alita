#!/usr/bin/env python3
"""Simple test of Paper2Code capability"""

import asyncio

from src.core.plugin_registry import register_plugin
from src.pipelines.autogen_pipeline import autogen_any
from src.plugins.native_deepcode_plugin import NativeDeepCodePlugin


async def test_simple_paper2code():
    """Test Paper2Code capability"""

    # Setup plugin registry
    plugin = NativeDeepCodePlugin()
    await plugin.start()
    register_plugin("native_deepcode", plugin)
    print("Plugin registered: native_deepcode")

    # Test Paper2Code request - shorter description
    paper_request = "Implement the Transformer attention mechanism"

    # Debug the need detector response
    from src.policies.need_detector import NeedDetector

    detector = NeedDetector()
    needs = detector.detect(paper_request)
    print(f"\nNeed detector results: {needs}")

    # Test full pipeline
    print("\nRunning autogen pipeline...")
    result = await autogen_any(paper_request)
    print(f"Pipeline result: {result}")

    await plugin.stop()


if __name__ == "__main__":
    asyncio.run(test_simple_paper2code())
