#!/usr/bin/env python3
"""
Demo: Super Alita + Ollama Integ        print("\n🔌 Testing connection to Ollama...")

        # For testing, we'll skip full initialization and test the consensus directly
        # In a real app, you'd pass a proper EventBus

        print("✅ Testing consensus sampling with proper parameters...")

        # Test with conservative parameters to avoid verbose output
        response = await ability.sample_consensus(
            prompt="Hello! Please respond with just 'Hi there!' and nothing else.",
            num_samples=1,  # Single sample for testing
            temperature=0.1,  # Low temperature for focused responses
            max_tokens=10,   # Limit output length
            mode=ConsensusMode.OFFLINE  # Use offline mode for simpler testing
        )ample
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def test_ollama_working():
    """Test the working Ollama integration"""
    try:
        from src.abilities.deepconf_ability import ConsensusMode, DeepConfAbility

        print("🚀 Super Alita + Ollama Integration Test")
        print("=" * 50)

        # Configure for testing both models
        # Use llama3.2:1b as known working, test gpt-oss-20b-split if desired
        test_models = [
            {
                "name": "llama3.2:1b",
                "config": {
                    "vllm_base_url": "http://localhost:11434/v1",
                    "model_name": "llama3.2:1b",
                    "timeout": 30.0,
                    "max_retries": 2,
                },
            },
            {
                "name": "gpt-oss-20b-split",
                "config": {
                    "vllm_base_url": "http://localhost:11434/v1",
                    "model_name": "gpt-oss-20b-split",
                    "timeout": 60.0,  # Longer timeout for larger model
                    "max_retries": 1,
                },
            },
        ]

        # Start with the working model
        config = test_models[0]["config"]

        print(f"📋 Testing with model: {config['model_name']}")
        print(f"📋 Endpoint: {config['vllm_base_url']}")

        # Create ability
        ability = DeepConfAbility(config)
        print("✅ DeepConf ability created")

        # Show capabilities
        info = ability.get_plugin_info()
        print(f"✅ Plugin: {info['name']} v{info['version']}")
        print(f"✅ Capabilities: {', '.join(info['capabilities'][:3])}...")
        print(f"✅ Consensus modes: {info['supported_modes']}")

        print("\n🔌 Testing connection to Ollama...")

        # Test initialization
        success = await ability.initialize(None)
        if success:
            print("✅ Connected to Ollama successfully!")

            print("\n🧠 Testing consensus sampling...")

            # Simple consensus test
            response = await ability.sample_consensus(
                prompt="What is 2+2?",
                num_samples=2,
                temperature=0.5,
                max_tokens=30,
                mode=ConsensusMode.ONLINE,
            )

            print("✅ Consensus sampling successful!")
            print("📝 Question: What is 2+2?")
            print(f"📝 Consensus answer: {response.consensus_text}")
            print(f"🎯 Confidence: {response.consensus_confidence:.3f}")
            print(f"📊 Aggregation method: {response.aggregation_method}")
            print(f"🔢 Individual responses: {len(response.individual_responses)}")

            # Show metadata
            print(f"📊 Metadata: {response.metadata}")

            print("\n✅ Integration test completed successfully!")

            return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main function"""
    print("Testing Super Alita with Ollama llama3.2:1b model...")
    print()

    success = await test_ollama_working()

    if success:
        print("\n🎉 SUCCESS: Super Alita is working with Ollama!")
        print("\n💡 You can now use:")
        print("   • Consensus sampling with multiple modes")
        print("   • Confidence calibration")
        print("   • Batch processing")
        print("   • Caching for improved performance")
    else:
        print("\n⚠️ Integration test failed")
        print("Check Ollama status and model availability")


if __name__ == "__main__":
    asyncio.run(main())
