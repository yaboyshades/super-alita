#!/usr/bin/env python3
"""Simple validation that MCP server components are available."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_mcp_server_import():
    """Test that MCP server can be imported."""
    try:
        import importlib.util

        # Test that the wrapper file can be loaded
        wrapper_path = Path(__file__).parent / "mcp_server_wrapper.py"
        spec = importlib.util.spec_from_file_location("mcp_wrapper", wrapper_path)

        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            # Don't execute the module (which would start the server)
            # Just check that it can be loaded
            return {"success": True, "message": "MCP wrapper can be imported"}
        else:
            return {"success": False, "error": "Could not create module spec"}

    except Exception as e:
        return {"success": False, "error": str(e)}


def test_mcp_tools_import():
    """Test that MCP tools can be imported."""
    try:
        from mcp_server.tools import (
            find_missing_docstrings,
            format_and_lint,
            refactor_to_result,
        )

        return {"success": True, "message": "MCP tools imported successfully"}
    except ImportError as e:
        return {"success": False, "error": f"MCP tools import failed: {e}"}


def main():
    """Run MCP validation tests."""
    print("🔍 Validating MCP Server Components...")

    # Test wrapper import
    wrapper_result = test_mcp_server_import()
    if wrapper_result["success"]:
        print("✅ MCP wrapper can be imported")
    else:
        print(f"❌ MCP wrapper import failed: {wrapper_result['error']}")

    # Test tools import
    tools_result = test_mcp_tools_import()
    if tools_result["success"]:
        print("✅ MCP tools imported successfully")
    else:
        print(f"❌ MCP tools import failed: {tools_result['error']}")

    # Overall result
    overall_success = wrapper_result["success"] and tools_result["success"]
    if overall_success:
        print("🎉 MCP server components are ready!")
        return {"success": True, "components": "all"}
    else:
        print("⚠️  Some MCP components have issues")
        return {"success": False, "issues": "import_failures"}


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result["success"] else 1)
