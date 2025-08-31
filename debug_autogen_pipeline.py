#!/usr/bin/env python3
"""Debug the autogen pipeline step by step"""

import asyncio
import sys

sys.path.append('./src')
import re

from contracts.gates.common_gates import (
    CombinedGate,
    PytestGate,
    RequiredPathsGate,
    SafetyGate,
)
from core.plugin_registry import register_plugin
from native_deepcode_api import get_native_deepcode_api, set_native_plugin
from plugins.native_deepcode_plugin import NativeDeepCodePlugin
from policies.need_detector import NeedDetector


async def debug_autogen_pipeline():
    print('=== DEBUGGING AUTOGEN PIPELINE ===')
    
    # Setup plugin
    plugin = NativeDeepCodePlugin()
    await plugin.setup(None, None, {})
    await plugin.start()
    set_native_plugin(plugin)
    register_plugin('native_deepcode', plugin)
    api = get_native_deepcode_api()
    
    # Test request
    test_request = 'Build a web scraper for extracting product data from e-commerce sites'
    
    print(f'\n1. REQUEST: {test_request}')
    
    # Step 1: Need detection
    detector = NeedDetector()
    needs = detector.detect(test_request)
    print(f'\n2. DETECTED NEEDS: {needs}')
    
    if not needs:
        print('❌ No needs detected!')
        return
    
    # Step 2: For each detected need, test the pipeline
    for need in needs:
        print(f'\n3. PROCESSING NEED: {need}')
        
        # Step 3: Generate code
        print('   3.1 Making deepcode_request...')
        result = await api.deepcode_request(
            task_kind=need,
            requirements=f'Web scraper for {test_request}',
            repo_path='.'
        )
        print(f'   Request result: {result}')
        
        # Step 4: Get latest results
        print('   3.2 Getting latest results...')
        latest = await api.deepcode_latest()
        if latest:
            print(f'   Latest keys: {list(latest.keys())}')
            if 'diffs' in latest:
                print(f'   Generated files: {[d["path"] for d in latest["diffs"]]}')
        
        # Step 5: Test gate validation
        print('   3.3 Testing gate validation...')
        
        # Create the gate that would be used for web_scraper
        gate = CombinedGate(
            RequiredPathsGate(
                required_paths=[
                    re.compile(r'^src/abilities/.*web_scraper', re.I),
                    re.compile(r'^tests/abilities/.*web_scraper', re.I),
                ],
                required_docs=[re.compile(r'^docs/.*web_scraper', re.I)],
            ),
            SafetyGate(api),
            PytestGate(api),
        )
        
        ok, info = gate.validate_latest(latest)
        print(f'   Gate validation: {ok}')
        print(f'   Gate info: {info}')
        
        if ok:
            print('   ✓ Gate validation passed!')
            # Step 6: Apply changes
            print('   3.4 Applying changes...')
            apply_result = await api.deepcode_apply()
            print(f'   Apply result: {apply_result}')
        else:
            print('   ❌ Gate validation failed!')
            if 'reasons' in info:
                print(f'   Failure reasons: {info["reasons"]}')
    
    await plugin.stop()
    print('\n=== DEBUGGING COMPLETE ===')


if __name__ == "__main__":
    asyncio.run(debug_autogen_pipeline())