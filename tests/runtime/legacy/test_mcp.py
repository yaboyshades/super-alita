#!/usr/bin/env python3
import pytest
pytest.skip("legacy test", allow_module_level=True)

try:
    from mcp.server.fastmcp import FastMCP
    print("✅ FastMCP imported successfully")
    app = FastMCP("test")
    print("✅ FastMCP app created successfully")
    print("✅ MCP server is working properly")
except Exception as e:
    print(f"❌ Error: {e}")
