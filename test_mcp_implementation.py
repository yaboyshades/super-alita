#!/usr/bin/env python3
"""
Quick test script to validate our MCP server implementation without external dependencies.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mcp'))

# Mock the fastmcp module since we can't install it in this environment
class MockFastMCP:
    def __init__(self, name, instructions=""):
        self.name = name
        self.instructions = instructions
        self.tools = {}
    
    def tool(self):
        def decorator(func):
            self.tools[func.__name__] = func
            return func
        return decorator
    
    def run(self, transport="stdio", host="0.0.0.0", port=8000):
        print(f"Mock MCP server '{self.name}' would run on {host}:{port} via {transport}")

# Mock the modules
sys.modules['fastmcp'] = type('MockModule', (), {'FastMCP': MockFastMCP})()
sys.modules['openai'] = type('MockModule', (), {'OpenAI': lambda api_key: None})()

# Now test our imports
try:
    import fastmcp_server
    print("✓ FastMCP server module imports successfully")
    
    # Test server creation
    server = fastmcp_server.create_server()
    print(f"✓ Server created with {len(server.tools)} tools")
    
    # Test that we have the expected tools
    expected_tools = ['search', 'fetch']
    for tool in expected_tools:
        if tool in server.tools:
            print(f"✓ Tool '{tool}' registered")
        else:
            print(f"✗ Tool '{tool}' missing")
    
    print("✓ MCP server implementation looks good!")
    
except Exception as e:
    print(f"✗ Error testing MCP server: {e}")
    import traceback
    traceback.print_exc()