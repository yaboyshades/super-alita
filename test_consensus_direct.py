#!/usr/bin/env python3
"""Test the consensus tool registration directly."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_consensus_tool():
    """Test the consensus tool through the ability registry."""
    print("🚀 Starting consensus tool test...")

    try:
        print("📦 Importing main module...")
        from src.main import create_app

        print("🔧 Creating FastAPI app...")
        app = create_app()
        print(f"✅ App created: {type(app)}")

        print("🔧 Getting ability registry...")
        registry = app.state.ability_registry
        print(f"✅ Registry obtained: {type(registry)}")

        print("🔧 Getting available tools...")
        tools = registry.get_available_tools_schema()
        print(f"✅ Tools obtained: {len(tools)} tools found")

        print(f"📊 Total tools: {len(tools)}")
        for tool in tools:
            name = tool.get("tool_id") or tool.get("name")
            description = tool.get("description", "")
            print(f"  - {name}: {description[:60]}...")

        # Look for consensus tool specifically
        consensus_tools = [
            t
            for t in tools
            if "consensus" in str(t.get("tool_id", "")).lower()
            or "deepconf" in str(t.get("tool_id", "")).lower()
        ]

        if consensus_tools:
            print(f"\n✅ Found {len(consensus_tools)} consensus tool(s):")
            for tool in consensus_tools:
                print(f"  * {tool.get('tool_id')}")

            # Test execution of the consensus tool
            print("\n🧪 Testing consensus tool execution...")
            test_args = {
                "prompt": "What is the capital of France?",
                "num_samples": 2,
                "temperature": 0.7,
                "max_tokens": 50,
            }

            result = await registry.execute("deepconf_consensus", test_args)
            print(f"✅ Consensus tool result: {result}")

        else:
            print("\n❌ No consensus tools found!")
            print("Available tool IDs:")
            for tool in tools:
                print(f"  - {tool.get('tool_id', 'NO_ID')}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_consensus_tool())
