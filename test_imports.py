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
        import asyncio
        from dataclasses import dataclass
        from enum import Enum
        from typing import Any

        print("✅ Basic imports successful")

        print("2. Testing VLLMDeepConfClient import...")
        from src.clients.deepconf_vllm import VLLMDeepConfClient

        print("✅ VLLMDeepConfClient import successful")

        print("3. Testing PluginInterface import...")
        from src.plugins.plugin_interface import PluginInterface

        print("✅ PluginInterface import successful")

        print("4. Testing EnhancedDeepConfPipeline import...")
        from src.reasoning.deepconf_pipeline import EnhancedDeepConfPipeline

        print("✅ EnhancedDeepConfPipeline import successful")

        print("5. Testing full DeepConfAbility import...")
        from src.abilities.deepconf_ability import DeepConfAbility, ConsensusMode

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
