#!/usr/bin/env python3
"""
Test imports step by step to find the issue
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test imports step by step"""
    try:
        print("🔍 Testing step-by-step imports...")

        print("1. Testing basic imports...")

        print("✅ Basic imports successful")

        print("2. Testing VLLMDeepConfClient import...")

        print("✅ VLLMDeepConfClient import successful")

        print("3. Testing PluginInterface import...")

        print("✅ PluginInterface import successful")

        print("4. Testing EnhancedDeepConfPipeline import...")

        print("✅ EnhancedDeepConfPipeline import successful")

        print("5. Testing full DeepConfAbility import...")

        print("✅ DeepConfAbility import successful")

        return True

    except Exception as e:
        print(f"❌ Import failed at step: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🎉 All imports successful!")
    else:
        print("\n⚠️ Import test failed")
