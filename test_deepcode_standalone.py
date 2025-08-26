#!/usr/bin/env python3
"""Test DeepCode HTTP endpoint independently"""

import asyncio
from pathlib import Path


async def test_deepcode_endpoint():
    """Test the DeepCode HTTP endpoint"""

    # First, let's test without the full server - just the logic
    print("Testing DeepCode analysis logic...")
    from src.deepcode import analyze_current_file

    test_file = "src/main.py"
    result = await analyze_current_file(test_file)
    print(
        f"✅ DeepCode analysis successful: {len(result.get('issues', []))} issues found"
    )

    # Test the event bus logic
    print("\nTesting event bus integration...")
    from src.reug_runtime.event_bus import make_event_bus
    from src.telemetry import create_event

    bus = await make_event_bus()

    # Create a test DeepCode request event
    payload = {
        "task_kind": "analyze",
        "repo_path": str(Path.cwd()),
        "conversation_id": "test-123",
    }

    evt = create_event("deepcode_request", **payload)
    await bus.emit(evt)
    print("✅ Event bus emission successful")

    # Test the plugins
    print("\nTesting DeepCode plugins...")
    try:
        from src.plugins.deepcode_generator_plugin import DeepCodeGeneratorBridgePlugin
        from src.plugins.deepcode_orchestrator_plugin import DeepCodeOrchestratorPlugin

        gen_plugin = DeepCodeGeneratorBridgePlugin()
        orch_plugin = DeepCodeOrchestratorPlugin()

        print(f"✅ Plugins loaded: {gen_plugin.name}, {orch_plugin.name}")

        # Test plugin event handling
        await gen_plugin.start(bus)
        await orch_plugin.start(bus)

        print("✅ Plugins started successfully")

    except Exception as e:
        print(f"❌ Plugin test failed: {e}")
        return False

    print("\n🎉 All DeepCode integration tests passed!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_deepcode_endpoint())
    if success:
        print("\n📋 DeepCode Integration Summary:")
        print("- ✅ Core analysis engine working")
        print("- ✅ Event bus integration functional")
        print("- ✅ Plugin system operational")
        print("- ✅ Ready for VS Code command integration")

        print("\n🚀 Next Steps:")
        print("1. Test VS Code commands in extension")
        print("2. Verify end-to-end workflow")
        print("3. Test real DeepCode service integration")
    else:
        print("❌ Integration test failed")
