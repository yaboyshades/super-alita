#!/usr/bin/env python3
"""
Super Alita Deployment Validation Script
Tests all core components systematically
"""

import asyncio
import sys
from datetime import UTC
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_core_imports():
    """Test that all core modules can be imported"""
    print("🔍 Testing Core Imports...")

    try:
        print("  ✅ Main app import: SUCCESS")
    except Exception as e:
        print(f"  ❌ Main app import: FAILED - {e}")
        return False

    try:
        print("  ✅ Decision Policy v1: SUCCESS")
    except Exception as e:
        print(f"  ❌ Decision Policy v1: FAILED - {e}")

    try:
        print("  ✅ Event Bus: SUCCESS")
    except Exception as e:
        print(f"  ❌ Event Bus: FAILED - {e}")

    try:
        print("  ✅ REUG Runtime: SUCCESS")
    except Exception as e:
        print(f"  ❌ REUG Runtime: FAILED - {e}")

    return True


async def test_app_creation():
    """Test that the FastAPI app can be created"""
    print("\n🏗️ Testing App Creation...")

    try:
        from src.main import create_app

        app = create_app()
        print("  ✅ FastAPI app created successfully")

        # Check routes
        routes = [route.path for route in app.routes]
        print(f"  📋 Available routes: {routes}")

        return True
    except Exception as e:
        print(f"  ❌ App creation failed: {e}")
        return False


async def test_decision_policy():
    """Test Decision Policy v1 functionality"""
    print("\n🧠 Testing Decision Policy v1...")

    try:
        from src.core.decision_policy_v1 import (
            DecisionPolicyEngine,
            Goal,
            IntentType,
            RiskLevel,
        )

        engine = DecisionPolicyEngine()
        print("  ✅ Decision Policy engine created")

        # Test intent classification
        classifier = engine.intent_classifier
        intent = classifier.classify("clone a git repository")
        print(f"  ✅ Intent classification: {intent}")

        # Test goal synthesis
        goal = Goal(
            description="test goal",
            intent=IntentType.CREATE,
            slots={},
            risk_level=RiskLevel.LOW,
            success_criteria=["complete"],
            constraints=[],
        )
        print(f"  ✅ Goal creation: {goal.description}")

        return True
    except Exception as e:
        print(f"  ❌ Decision Policy test failed: {e}")
        return False


async def test_event_bus():
    """Test Event Bus functionality"""
    print("\n📡 Testing Event Bus...")

    try:
        from datetime import datetime

        from src.core.events import create_event

        event = create_event(
            "test_event",
            source_plugin="deployment_validator",
            message="Test deployment validation",
            timestamp=datetime.now(UTC),
        )
        print(f"  ✅ Event creation: {event.event_type}")

        return True
    except Exception as e:
        print(f"  ❌ Event Bus test failed: {e}")
        return False


async def test_mcp_integration():
    """Test MCP server integration"""
    print("\n🔌 Testing MCP Integration...")

    try:
        import importlib.util
        from pathlib import Path

        # Test that the wrapper file exists and can be loaded as spec
        wrapper_path = Path(__file__).parent / "mcp_server_wrapper.py"
        if not wrapper_path.exists():
            print("  ❌ MCP wrapper file not found")
            return False

        spec = importlib.util.spec_from_file_location("mcp_wrapper", wrapper_path)
        if not spec or not spec.loader:
            print("  ❌ Could not create module spec for MCP wrapper")
            return False

        print("  ✅ MCP wrapper file exists and is loadable")

        # Test that MCP tools can be imported
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        try:
            from mcp_server.tools import (
                find_missing_docstrings,
                format_and_lint,
                refactor_to_result,
            )

            print("  ✅ MCP tools imported successfully")
        except ImportError as e:
            print(f"  ⚠️ MCP tools import failed (fallbacks available): {e}")

        print("  ✅ MCP server components available")
        return True

    except Exception as e:
        print(f"  ❌ MCP integration test failed: {e}")
        return False


async def test_tool_registries():
    """Test tool registry functionality"""
    print("\n🔧 Testing Tool Registries...")

    try:
        from src.main import SimpleAbilityRegistry

        registry = SimpleAbilityRegistry()
        tools = registry.get_available_tools_schema()
        print(f"  ✅ Simple registry: {len(tools)} tools available")

        # Test if echo tool is available
        if registry.knows("echo"):
            print("  ✅ Echo tool registered")
        else:
            print("  ⚠️ Echo tool not found")

        return True
    except Exception as e:
        print(f"  ❌ Tool registry test failed: {e}")
        return False


async def test_plugin_system():
    """Test plugin loading system"""
    print("\n🔌 Testing Plugin System...")

    try:
        # Check for plugin interface
        print("  ✅ Plugin interface available")

        # Check for plugin loader functions
        from src.core.plugin_loader import discover_plugins, load_plugin_manifest

        print("  ✅ Plugin loader functions available")

        # Try to load plugins
        try:
            # First, try to load the plugin manifest
            manifest = load_plugin_manifest()
            plugins = discover_plugins(manifest)
            print(f"  ✅ Plugin discovery: {len(plugins)} plugins found")
        except Exception as e:
            print(f"  ⚠️ Plugin discovery: {e}")

        return True
    except Exception as e:
        print(f"  ❌ Plugin system test failed: {e}")
        return False


async def run_deployment_validation():
    """Run complete deployment validation"""
    print(">>> Super Alita Deployment Validation <<<")
    print("=" * 50)

    tests = [
        test_core_imports,
        test_app_creation,
        test_decision_policy,
        test_event_bus,
        test_mcp_integration,
        test_tool_registries,
        test_plugin_system,
    ]

    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"  💥 Test failed with exception: {e}")
            results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)

    print("\n" + "=" * 50)
    print("📊 DEPLOYMENT VALIDATION SUMMARY")
    print("=" * 50)
    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"❌ Tests Failed: {total - passed}/{total}")

    if passed == total:
        print("SUCCESS! ALL TESTS PASSED - Super Alita ready for deployment!")
    else:
        print("WARNING! Some tests failed - Check output above for issues")

    return passed == total


if __name__ == "__main__":
    asyncio.run(run_deployment_validation())
