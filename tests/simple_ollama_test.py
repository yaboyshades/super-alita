#!/usr/bin/env python3
"""
Simple Super Alita + Ollama Integration Test
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def test_simple_integration():
    """Simple test of Super Alita with Ollama"""
    try:
        from src.abilities.deepconf_ability import (
            ConsensusMode,
            DeepConfAbility,
        )

        print("🚀 Super Alita + Ollama Integration Test")
        print("=" * 50)

        # Simple config for llama3.2:1b
        config = {
            "vllm_base_url": "http://localhost:11434/v1",
            "model_name": "llama3.2:1b",
            "timeout": 30.0,
            "max_retries": 2,
        }

        print(f"📋 Model: {config['model_name']}")
        print(f"📋 Endpoint: {config['vllm_base_url']}")

        # Create ability
        ability = DeepConfAbility(config)
        print("✅ DeepConf ability created")

        # Get plugin info
        info = ability.get_plugin_info()
        print(f"✅ Plugin: {info['name']} v{info['version']}")

        print("\n🧠 Testing consensus sampling...")

        # Simple test with conservative parameters
        response = await ability.sample_consensus(
            prompt="What is 2+2? Answer briefly.",
            num_samples=1,
            temperature=0.1,
            max_tokens=20,
            mode=ConsensusMode.OFFLINE,
        )

        print("✅ Consensus sampling successful!")
        print(f"📝 Response: {response.consensus_text}")
        print(f"🎯 Confidence: {response.consensus_confidence:.3f}")
        print(f"📊 Method: {response.aggregation_method}")

        print("\n✅ Integration test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main function"""
    print("Testing Super Alita with Ollama llama3.2:1b...")
    print()

    success = await test_simple_integration()

    if success:
        print("\n🎉 SUCCESS: Super Alita + Ollama integration works!")
        print("\n💡 You can now use:")
        print("   • Consensus sampling")
        print("   • Multiple sampling modes")
        print("   • Confidence calibration")
    else:
        print("\n⚠️ Integration test failed")


if __name__ == "__main__":
    asyncio.run(main())
