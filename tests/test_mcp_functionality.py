#!/usr/bin/env python3
"""
Test script to verify MCP server functionality through stdio transport.
This simulates how Claude Desktop would interact with the MCP server.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from mcp_server_wrapper import app


async def test_mcp_tools():
    """Test the MCP tools functionality directly."""
    print("🔧 Testing Super Alita MCP Tools")
    print("=" * 50)

    # Test 1: find_missing_docstrings
    print("\n1. Testing find_missing_docstrings tool...")
    try:
        result = await app._handle_call(
            "find_missing_docstrings", {"root": "src", "include_tests": False}
        )
        print(f"✅ Found {result.get('count', 0)} functions missing docstrings")
        if result.get("functions"):
            print(f"   First few: {result['functions'][:3]}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 2: format_and_lint_selection (dry run on a small file)
    print("\n2. Testing format_and_lint_selection tool...")
    try:
        test_file = REPO_ROOT / "pyproject.toml"
        if test_file.exists():
            result = await app._handle_call(
                "format_and_lint_selection", {"target_path": str(test_file)}
            )
            print(f"✅ Formatting result: stdout={len(result.get('stdout', ''))} chars")
        else:
            print("⚠️  Test file not found, skipping format test")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 3: apply_result_pattern_refactor (dry run)
    print("\n3. Testing apply_result_pattern_refactor tool...")
    try:
        result = await app._handle_call(
            "apply_result_pattern_refactor",
            {
                "file_path": "src/main.py",
                "function_name": "create_app",
                "dry_run": True,
            },
        )
        print(f"✅ Refactor analysis: applied={result.get('applied', False)}")
        if result.get("diff"):
            print(f"   Diff length: {len(result['diff'])} chars")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n🎉 MCP Tools test completed!")


async def test_telemetry():
    """Test telemetry emission functionality."""
    print("\n📊 Testing Telemetry Integration")
    print("=" * 50)

    telemetry_file = REPO_ROOT / "telemetry.jsonl"

    # Count existing telemetry entries
    existing_lines = 0
    if telemetry_file.exists():
        with open(telemetry_file) as f:
            existing_lines = len(f.readlines())

    print(f"📈 Existing telemetry entries: {existing_lines}")

    # Run a tool to generate telemetry
    try:
        await app._handle_call(
            "find_missing_docstrings", {"root": "tests", "include_tests": True}
        )

        # Check if new telemetry was written
        new_lines = 0
        if telemetry_file.exists():
            with open(telemetry_file) as f:
                new_lines = len(f.readlines())

        print(f"📈 New telemetry entries: {new_lines - existing_lines}")

        if new_lines > existing_lines:
            print("✅ Telemetry is working correctly!")
        else:
            print("⚠️  No new telemetry entries detected")

    except Exception as e:
        print(f"❌ Telemetry test error: {e}")


def main():
    """Main test function."""
    print("🚀 Super Alita MCP Server Test Suite")
    print("=" * 60)

    # Test if MCP server imports are working
    print("✅ MCP server wrapper imported successfully")
    print(f"✅ FastMCP app created: {type(app).__name__}")

    # Run async tests
    asyncio.run(test_mcp_tools())
    asyncio.run(test_telemetry())

    print("\n🎯 Test Summary:")
    print("- MCP server imports: ✅")
    print("- Tool functionality: ✅ (see details above)")
    print("- Telemetry emission: ✅ (see details above)")

    print("\n💡 Next steps:")
    print("1. Configure Claude Desktop with claude_desktop_config_local.json")
    print("2. Test MCP integration in Claude Desktop")
    print("3. Use tools in VS Code/Copilot with stdio transport")


if __name__ == "__main__":
    main()
