#!/usr/bin/env python3
"""
Test DeepConf ability registration directly
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def test_deepconf_registration():
    """Test DeepConf ability registration"""
    try:
        print("🔍 Testing DeepConf ability registration...")

        # Import the ability
        from src.abilities.deepconf_ability import ConsensusMode, DeepConfAbility

        print("✅ DeepConf imports successful")

        # Create the ability
        deepconf_ability = DeepConfAbility(
            {
                "vllm_base_url": "http://localhost:11434/v1",
                "model_name": "gpt-oss:20b",
                "timeout": 60.0,
                "max_retries": 2,
            }
        )
        print("✅ DeepConf ability created")

        # Create mock event bus
        class MockEventBus:
            async def emit(self, event_type: str, data: dict = None):
                if data is None:
                    data = {}
                print(f"📡 Event emitted: {event_type}")

            async def subscribe(self, event_type: str, handler):
                print(f"📡 Subscribed to: {event_type}")

        # Initialize the ability
        mock_bus = MockEventBus()
        success = await deepconf_ability.initialize(mock_bus)
        print(f"✅ DeepConf initialization: {success}")

        # Test basic functionality
        if success:
            print("🧪 Testing consensus sampling...")
            response = await deepconf_ability.sample_consensus(
                prompt="Test prompt",
                num_samples=1,
                temperature=0.1,
                max_tokens=10,
                mode=ConsensusMode.OFFLINE,
            )
            print(f"✅ Consensus response: {response.consensus_text[:50]}...")
            print(f"✅ Confidence: {response.consensus_confidence}")

        return True

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    print("Testing DeepConf ability registration...")
    success = await test_deepconf_registration()

    if success:
        print("\n🎉 DeepConf ability test successful!")
    else:
        print("\n⚠️ DeepConf ability test failed")


if __name__ == "__main__":
    asyncio.run(main())
