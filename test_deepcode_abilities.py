#!/usr/bin/env python3
"""
Test script to validate deepcode abilities integration
"""

import asyncio
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_deepcode_abilities():
    """Test the deepcode abilities integration"""
    
    # Test imports
    try:
        from src.abilities.deepcode_analysis_ability import DeepCodeAnalysisAbility
        from src.abilities.deepcode_integration_ability import DeepCodeIntegrationAbility
        from src.main import SimpleAbilityRegistry
        print("✅ Successfully imported deepcode abilities")
    except Exception as e:
        print(f"❌ Failed to import deepcode abilities: {e}")
        return False

    # Test ability creation
    try:
        analysis_ability = DeepCodeAnalysisAbility()
        integration_ability = DeepCodeIntegrationAbility()
        print("✅ Successfully created ability instances")
    except Exception as e:
        print(f"❌ Failed to create ability instances: {e}")
        return False

    # Test registry integration
    try:
        registry = SimpleAbilityRegistry()
        tools = registry.get_available_tools_schema()
        deepcode_tools = [t for t in tools if t["tool_id"].startswith(("analyze_", "detect_", "get_", "check_", "understand_"))]
        print(f"✅ Registry contains {len(deepcode_tools)} deepcode tools out of {len(tools)} total tools")
        
        # Print deepcode tools
        for tool in deepcode_tools:
            print(f"   - {tool['tool_id']}: {tool['description']}")
            
    except Exception as e:
        print(f"❌ Failed to test registry integration: {e}")
        return False

    # Test ability setup
    try:
        await analysis_ability.setup(None, None, {})
        await integration_ability.setup(None, None, {})
        print("✅ Successfully setup abilities")
    except Exception as e:
        print(f"❌ Failed to setup abilities: {e}")
        return False

    # Test tool schema retrieval
    try:
        analysis_tools = analysis_ability.get_available_tools()
        integration_tools = integration_ability.get_available_tools()
        print(f"✅ Analysis ability provides {len(analysis_tools)} tools")
        print(f"✅ Integration ability provides {len(integration_tools)} tools")
        
        for tool in analysis_tools:
            print(f"   Analysis: {tool['name']}")
        for tool in integration_tools:
            print(f"   Integration: {tool['name']}")
            
    except Exception as e:
        print(f"❌ Failed to get tool schemas: {e}")
        return False

    # Test simple tool execution through registry
    try:
        # Test a simple tool that doesn't require actual files
        result = await registry.execute("get_supported_extensions", {})
        print(f"✅ Successfully executed get_supported_extensions: {result}")
    except Exception as e:
        print(f"❌ Failed to execute tool through registry: {e}")
        return False

    print("\n🎉 All deepcode abilities tests passed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_deepcode_abilities())
    exit(0 if success else 1)