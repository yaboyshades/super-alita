#!/usr/bin/env python3
"""Test plugin direct invocation"""

import asyncio

from src.core.plugin_registry import register_plugin
from src.plugins.native_deepcode_plugin import NativeDeepCodePlugin


async def test():
    # Register plugin
    plugin = NativeDeepCodePlugin()
    await plugin.start()
    register_plugin("native_deepcode", plugin)

    print(f"Registered plugin: {plugin}")

    # Test plugin directly with invoke_tool
    result = await plugin.invoke_tool(
        "deepcode_request",
        {
            "task_kind": "paper2code",
            "requirements": "Implement ResNet architecture",
            "repo_path": ".",
        },
    )
    print(f'Direct plugin result status: {result.get("status")}')
    print(f"Result keys: {list(result.keys())}")

    if "diffs" in result:
        print(f'Generated {len(result["diffs"])} files')
        for diff in result["diffs"]:
            print(f'  - {diff["path"]} ({diff["change_type"]})')

    await plugin.stop()


if __name__ == "__main__":
    asyncio.run(test())
