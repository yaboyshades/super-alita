#!/usr/bin/env python3
"""Test Paper2Code with different architectures to show adaptability"""

import asyncio

from src.core.plugin_registry import register_plugin
from src.pipelines.autogen_pipeline import autogen_any
from src.plugins.native_deepcode_plugin import NativeDeepCodePlugin


async def test_adaptive_paper2code():
    """Test Paper2Code adaptability with different architectures"""

    # Setup plugin registry
    plugin = NativeDeepCodePlugin()
    await plugin.start()
    register_plugin("native_deepcode", plugin)
    print("Plugin registered: native_deepcode")

    # Test different types of papers
    test_cases = [
        {
            "name": "Vision Transformer",
            "request": """
            Implement the Vision Transformer (ViT) from 'An Image is Worth 16x16 Words:
            Transformers for Image Recognition at Scale'. Key features include patch
            embedding, positional encoding, multi-head attention, and classification head.
            """,
        },
        {
            "name": "Memory Networks",
            "request": """
            Implement Memory Networks for question answering with episodic memory,
            attention-based retrieval, and working memory components for multi-hop reasoning.
            """,
        },
        {
            "name": "Multimodal Fusion",
            "request": """
            Implement a multimodal fusion architecture for combining text and image features
            using cross-attention mechanisms and adaptive gating for dynamic weighting.
            """,
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}: {test_case['name']}")
        print(f"{'='*50}")

        result = await autogen_any(test_case["request"])

        if result.get("status") == "complete" and result.get("applied"):
            print(f"✅ {test_case['name']} generated successfully!")
            for applied_item in result.get("applied", []):
                if isinstance(applied_item, dict) and "paths" in applied_item:
                    main_file = [
                        p
                        for p in applied_item["paths"]
                        if "src/abilities" in p and p.endswith(".py")
                    ][0]
                    print(f"📄 Generated: {main_file}")
        else:
            print(f"❌ {test_case['name']} failed")

    await plugin.stop()
    print("\n🎉 Adaptive Paper2Code testing completed!")


if __name__ == "__main__":
    asyncio.run(test_adaptive_paper2code())
