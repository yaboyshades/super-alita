#!/usr/bin/env python3
"""Test Alita direct plugin invocation"""

import asyncio

from src.core.plugin_registry import register_plugin
from src.plugins.native_deepcode_plugin import NativeDeepCodePlugin


async def test():
    # Register plugin
    plugin = NativeDeepCodePlugin()
    await plugin.start()
    register_plugin('native_deepcode', plugin)
    
    print('Testing direct Alita plugin invocation...')
    
    # Test with Alita-specific request
    result = await plugin.invoke_tool('deepcode_request', {
        'task_kind': 'paper2code',
        'requirements': 'Implement Alita neural architecture with multi-modal fusion',
        'repo_path': '.'
    })
    print(f'Plugin result status: {result.get("status")}')
    
    # Get latest results
    latest = await plugin.invoke_tool('deepcode_latest', {})
    print(f'Latest result keys: {list(latest.keys())}')
    
    if 'diffs' in latest:
        print(f'Generated {len(latest["diffs"])} files:')
        for diff in latest['diffs']:
            print(f'  - {diff["path"]} ({diff["change_type"]})')
            # Check if it mentions Alita
            if 'alita' in diff['path'].lower():
                print('    ✅ Found Alita-specific file!')
    else:
        print('No diffs in latest results')
    
    await plugin.stop()

if __name__ == "__main__":
    asyncio.run(test())