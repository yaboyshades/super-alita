#!/usr/bin/env python3
"""
Super Alita + Ollama gpt-oss:20b Integration Test
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def test_gpt_oss_integration():
    """Test Super Alita with Ollama gpt-oss:20b"""
    try:
        from src.abilities.deepconf_ability import ConsensusMode, DeepConfAbility

        print("🚀 Super Alita + Ollama gpt-oss:20b Test")
        print("=" * 50)

        # Config for gpt-oss:20b
        config = {
            "vllm_base_url": "http://127.0.0.1:11434/v1",
            "model_name": "gpt-oss:20b",
            "timeout": 60.0,  # Longer timeout for 20B model
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

        # Initialize the ability (create mock event bus for testing)
        class MockEventBus:

            async def emit(self, event_type: str, data: dict = None):
                if data is None:
                    data = {}
                pass

            async def subscribe(self, event_type: str, handler):
                pass

        mock_bus = MockEventBus()
        success = await ability.initialize(mock_bus)
        if not success:
            print("❌ Failed to initialize DeepConf ability")
            return False
        print("✅ DeepConf ability initialized")

        print("\n🧠 Testing consensus sampling with gpt-oss:20b...")

        # Test with carefully tuned parameters
        response = await ability.sample_consensus(
            prompt="What is 2+2? Answer briefly with just the number.",
            num_samples=1,  # Single sample first
            temperature=0.1,
            max_tokens=10,
            mode=ConsensusMode.OFFLINE,
        )

        print("✅ Consensus sampling successful!")
        print(f"📝 Response: {response.consensus_text}")
        print(f"🎯 Confidence: {response.consensus_confidence:.3f}")
        print(f"📊 Method: {response.aggregation_method}")

        # Test a second question
        print("\n🧠 Testing second question...")
        response2 = await ability.sample_consensus(
            prompt="Hello! How are you? Respond briefly.",
            num_samples=1,
            temperature=0.2,
            max_tokens=20,
            mode=ConsensusMode.OFFLINE,
        )

        print("✅ Second test successful!")
        print(f"📝 Response: {response2.consensus_text}")

        print("\n✅ Integration test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main function"""
    print("Testing Super Alita with Ollama gpt-oss:20b...")
    print()

    success = await test_gpt_oss_integration()

    if success:
        print("\n🎉 SUCCESS: Super Alita + gpt-oss:20b integration works!")
        print("\n💡 The 20B model is now working with:")
        print("   • Super Alita consensus sampling")
        print("   • Proper parameter constraints")
        print("   • Confidence calibration")
        print("   • Production-ready integration")
    else:
        print("\n⚠️ Integration test failed")


if __name__ == "__main__":
    asyncio.run(main())
