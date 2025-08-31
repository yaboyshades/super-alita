#!/usr/bin/env python3
"""Test native DeepCode capability directly"""

import asyncio
import sys

sys.path.append('./src')
from core.plugin_registry import register_plugin
from native_deepcode_api import get_native_deepcode_api, set_native_plugin
from plugins.native_deepcode_plugin import NativeDeepCodePlugin


async def test_capability_directly():
    print('Testing native DeepCode capability directly...')
    
    # Setup plugin
    plugin = NativeDeepCodePlugin()
    await plugin.setup(None, None, {})
    await plugin.start()
    set_native_plugin(plugin)
    register_plugin('native_deepcode', plugin)
    
    api = get_native_deepcode_api()
    
    # Test the API directly
    print('\n1. Making deepcode_request...')
    result = await api.deepcode_request(
        task_kind='web_scraper',
        requirements='Simple web scraper for product data extraction',
        repo_path='.'
    )
    print(f'Request result: {result}')
    
    print('\n2. Getting latest results...')
    latest = await api.deepcode_latest()
    print(f'Latest results keys: {list(latest.keys()) if latest else "None"}')
    
    if 'diffs' in latest:
        print(f'Generated diffs: {len(latest["diffs"])}')
        for diff in latest['diffs']:
            print(f'  - {diff["path"]} (confidence: {diff.get("confidence", "N/A")})')
    
    print('\n3. Testing gate validation...')
    # Test what the gate is looking for
    from src.contracts.gates.common_gates import (
        CombinedGate,
        PytestGate,
        RequiredPathsGate,
        SafetyGate,
    )
    
    gate = CombinedGate(
        RequiredPathsGate(
            required_paths=["src/capabilities/web_scraper.py"],
            required_docs=["docs/capabilities/web_scraper.md"],
        ),
        SafetyGate(api),
        PytestGate(api),
    )
    
    ok, info = gate.validate_latest(latest)
    print(f'Gate validation result: {ok}')
    print(f'Gate info: {info}')
    
    await plugin.stop()


if __name__ == "__main__":
    asyncio.run(test_capability_directly())