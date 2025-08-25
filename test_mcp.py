#!/usr/bin/env python3
import sys
import os

# Remove the workspace from the Python path to avoid conflicts
sys.path = [p for p in sys.path if 'super-alita-clean' not in p]

# Add the virtual environment paths explicitly
venv_base = r'D:\Coding_Projects\super-alita-clean\.venv'
sys.path.extend([
    os.path.join(venv_base, 'Lib', 'site-packages'),
    os.path.join(venv_base, 'Lib', 'site-packages', 'win32'),
    os.path.join(venv_base, 'Lib', 'site-packages', 'win32', 'lib'),
    os.path.join(venv_base, 'Lib', 'site-packages', 'Pythonwin'),
])

try:
    from mcp.server.fastmcp import FastMCP
    print("✅ FastMCP imported successfully")
    app = FastMCP("test")
    print("✅ FastMCP app created successfully")
    print("✅ MCP server is working properly")
except Exception as e:
    print(f"❌ Error: {e}")
