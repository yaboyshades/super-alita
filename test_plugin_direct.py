#!/usr/bin/env python3
"""Direct test of native plugin paper2code"""

import asyncio

from src.plugins.native_deepcode_plugin import NativeDeepCodePlugin


async def test_plugin_direct():
    """Test paper2code directly through plugin"""
    
    plugin = NativeDeepCodePlugin()
    await plugin.start()
    
    try:
        # Test direct generation
        result = await plugin._generate_paper2code(
            "Implement Transformer attention mechanism", 
            "."
        )
        print(f"Paper2code generation result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    await plugin.stop()

if __name__ == "__main__":
    asyncio.run(test_plugin_direct())