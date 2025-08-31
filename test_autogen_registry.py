#!/usr/bin/env python3
"""Test the autogen pipeline with global plugin registry"""

import asyncio
import sys
from pathlib import Path

sys.path.append('./src')
from core.plugin_registry import register_plugin
from native_deepcode_api import set_native_plugin
from pipelines.autogen_pipeline import autogen_any
from plugins.native_deepcode_plugin import NativeDeepCodePlugin


async def test_autogen_with_registry():
    print('Testing autogen pipeline with global plugin registry...')
    
    # Setup the plugin and register it globally
    plugin = NativeDeepCodePlugin()
    await plugin.setup(None, None, {})
    await plugin.start()
    
    # Register the plugin globally so autogen can find it
    set_native_plugin(plugin)
    register_plugin("native_deepcode", plugin)
    
    print(f'Plugin registered: {plugin.name}')
    
    # Test the autogen pipeline
    test_request = '''
I need to implement a RESTful API client for a social media platform.
This should include authentication, rate limiting, error handling, and methods for:
- Getting user profiles
- Posting messages
- Following/unfollowing users
- Retrieving timeline feeds
- Uploading media
'''

    print('\nRunning autogen pipeline...')
    result = await autogen_any(test_request)
    print(f'Autogen result: {result}')
    
    # Check if actual code was generated
    if result.get('status') == 'complete' and result.get('applied'):
        print('\n✓ Success! Code was generated and applied.')
        for applied_item in result.get('applied', []):
            if isinstance(applied_item, dict) and 'paths' in applied_item:
                for file_path in applied_item['paths']:
                    if Path(file_path).exists():
                        print(f'  Generated: {file_path}')
            elif isinstance(applied_item, str):
                if Path(applied_item).exists():
                    print(f'  Generated: {applied_item}')
    elif 'failed' in result:
        print(f'\n⚠ Pipeline completed but some capabilities failed: {result["failed"]}')
    else:
        print('\n✗ Pipeline did not generate code')
    
    await plugin.stop()
    print('\nTest completed!')


if __name__ == "__main__":
    asyncio.run(test_autogen_with_registry())