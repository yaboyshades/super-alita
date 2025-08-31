#!/usr/bin/env python3
"""Test script for native DeepCode pipeline"""

import asyncio
import sys
from pathlib import Path

sys.path.append('./src')
from native_deepcode_api import get_native_deepcode_api, set_native_plugin
from plugins.native_deepcode_plugin import NativeDeepCodePlugin


async def test_full_pipeline():
    print('Testing complete native DeepCode pipeline...')
    
    # Create and setup the plugin
    plugin = NativeDeepCodePlugin()
    await plugin.setup(None, None, {})
    await plugin.start()
    
    # Connect it to the API
    set_native_plugin(plugin)
    api = get_native_deepcode_api()
    
    # Test a web scraper request
    print('\n1. Generating web scraper...')
    result = await api.deepcode_request(
        task_kind='web_scraper',
        requirements='Web scraper for extracting product data from e-commerce sites with rate limiting and error handling',
        repo_path='.'
    )
    print(f'Request result: {result}')
    
    # Get the latest results
    latest = await api.deepcode_latest()
    if 'diffs' in latest:
        print(f'\nGenerated {len(latest["diffs"])} files:')
        for diff in latest['diffs']:
            print(f'  - {diff["path"]} ({diff.get("change_type", "unknown")})')
    
    # Apply the generated code
    print('\n2. Applying generated code...')
    apply_result = await api.deepcode_apply()
    print(f'Apply result: {apply_result}')
    
    # Check if files were actually created
    if apply_result.get('applied_files'):
        print('\n3. Checking generated files:')
        for file_path in apply_result['applied_files']:
            if Path(file_path).exists():
                print(f'  ✓ Created: {file_path}')
                # Show first few lines
                try:
                    content = Path(file_path).read_text(encoding='utf-8')
                    lines = content.split('\n')[:5]
                    print(f'    Preview: {lines[0][:60]}...')
                except Exception as e:
                    print(f'    Error reading: {e}')
            else:
                print(f'  ✗ Missing: {file_path}')
    
    await plugin.stop()
    print('\nFull pipeline test completed!')


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())