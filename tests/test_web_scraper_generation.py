#!/usr/bin/env python3
"""Test web scraper generation specifically"""

import asyncio
import sys
from pathlib import Path

sys.path.append("./src")
from core.plugin_registry import register_plugin
from native_deepcode_api import set_native_plugin
from pipelines.autogen_pipeline import autogen_any
from plugins.native_deepcode_plugin import NativeDeepCodePlugin


async def test_web_scraper_generation():
    print("Testing web scraper generation with corrected paths...")

    # Setup plugin
    plugin = NativeDeepCodePlugin()
    await plugin.setup(None, None, {})
    await plugin.start()
    set_native_plugin(plugin)
    register_plugin("native_deepcode", plugin)

    # Test web scraper specifically
    test_request = (
        "Build a web scraper for extracting product data from e-commerce sites"
    )

    print(f"Request: {test_request}")
    result = await autogen_any(test_request)

    print(f"\nResult: {result}")

    if result.get("status") == "complete":
        applied = result.get("applied", [])
        failed = result.get("failed", [])

        print(f"Applied capabilities: {applied}")
        print(f"Failed capabilities: {failed}")

        if applied:
            print(
                "\n✓ SUCCESS! Native DeepCode plugin generated and applied code!"
            )
            # Check if files were actually created
            test_files = [
                "src/abilities/web_scraper.py",
                "tests/abilities/test_web_scraper.py",
                "docs/abilities/web_scraper.md",
            ]

            for file_path in test_files:
                if Path(file_path).exists():
                    print(f"  ✓ Created: {file_path}")
                    # Show file size
                    size = Path(file_path).stat().st_size
                    print(f"    Size: {size} bytes")
                else:
                    print(f"  ✗ Missing: {file_path}")
        else:
            print("\n⚠ No capabilities were applied")

    await plugin.stop()
    print("\nTest completed!")


if __name__ == "__main__":
    asyncio.run(test_web_scraper_generation())
