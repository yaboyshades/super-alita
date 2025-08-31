#!/usr/bin/env python3
"""Test Paper2Code with Real Research Paper - ResNet"""

import asyncio

from src.core.plugin_registry import register_plugin
from src.pipelines.autogen_pipeline import autogen_any
from src.plugins.native_deepcode_plugin import NativeDeepCodePlugin


async def test_resnet_paper():
    """Test Paper2Code with ResNet paper implementation"""
    
    # Setup plugin registry
    plugin = NativeDeepCodePlugin()
    await plugin.start()
    register_plugin("native_deepcode", plugin)
    print("Plugin registered: native_deepcode")
    
    # Real paper request - ResNet paper
    resnet_request = """
    Implement the ResNet architecture from 'Deep Residual Learning for Image Recognition' by He et al.
    
    Key requirements:
    - Implement residual blocks with skip connections (identity mapping)
    - Support both basic blocks (for ResNet-18/34) and bottleneck blocks (for ResNet-50/101/152)
    - Include batch normalization and ReLU activations as specified
    - Implement the full ResNet architecture with configurable depths
    - Add proper weight initialization (Kaiming initialization)
    - Include downsampling layers for feature map size reduction
    - Support different input sizes and number of classes
    
    The core innovation is the residual connection: F(x) + x where F(x) is the residual mapping.
    This solves the degradation problem in very deep networks.
    """
    
    print("\nTesting ResNet paper implementation...")
    print(f"Request: {resnet_request[:100]}...")
    
    # Run the autogen pipeline
    print("\nRunning autogen pipeline...")
    result = await autogen_any(resnet_request)
    print(f"\nPipeline result: {result}")
    
    # Check what files were generated
    if result.get('status') == 'complete' and result.get('applied'):
        print('\n✅ ResNet implementation generated successfully!')
        for applied_item in result.get('applied', []):
            if isinstance(applied_item, dict) and 'paths' in applied_item:
                print(f"\nGenerated files for {applied_item['kind']}:")
                for file_path in applied_item['paths']:
                    print(f"  📄 {file_path}")
    else:
        print('\n❌ ResNet implementation failed')
        if 'failed' in result:
            print(f"Failed capabilities: {result['failed']}")
    
    await plugin.stop()
    print("\nResNet test completed!")

if __name__ == "__main__":
    asyncio.run(test_resnet_paper())