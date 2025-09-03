#!/usr/bin/env python3
"""Test the MCP registration endpoint with canonical tool resolution.

This script tests that the MCP registration endpoint correctly:
1. Identifies canonical tools based on signature
2. Reuses existing canonical tools instead of creating duplicates
3. Persists new tools when no canonical equivalent exists
"""


import requests

# Base URL for the API
BASE_URL = "http://127.0.0.1:8080"


def test_mcp_registration():
    """Test MCP registration with canonical tool resolution."""
    # First, ensure we have a clean index
    print("Step 1: Creating fresh index...")
    res = requests.post(f"{BASE_URL}/tools/mcp/abstract", json={})
    if res.status_code == 200:
        print(f"✅ Index refreshed with {len(res.json()['tools'])} tools")
    else:
        print(f"❌ Failed to refresh index: {res.status_code}")
        return

    # Now register a new tool spec
    print("\nStep 2: Registering first tool...")
    calculator_spec = {
        "tool_id": "calculator_v1",
        "description": "Simple calculator for basic math",
        "action": "calculate",
        "input_schema": {
            "type": "object",
            "required": ["operation", "a", "b"],
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number"}},
        },
    }

    res = requests.post(f"{BASE_URL}/tools/mcp/register", json=calculator_spec)
    if res.status_code == 200:
        reg_result = res.json()
        print(f"✅ Tool registered: {reg_result}")
    else:
        print(f"❌ Failed to register tool: {res.status_code}")
        return

    # Now register a second tool with same signature but different ID
    print("\nStep 3: Registering second tool with same signature but different ID...")
    calculator_spec_2 = {
        "tool_id": "math_tool",  # Different ID
        "description": "Performs mathematical operations",  # Different description
        "action": "calculate",  # Same action
        "input_schema": {  # Same schema structure
            "type": "object",
            "required": ["operation", "a", "b"],
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number"}},
        },
    }

    res = requests.post(f"{BASE_URL}/tools/mcp/register", json=calculator_spec_2)
    if res.status_code == 200:
        reg_result_2 = res.json()
        print(f"✅ Second tool registered: {reg_result_2}")

        # Check if the canonical ID was used
        if reg_result_2.get("tool_id") == reg_result.get("tool_id"):
            print("✅ SUCCESS: Canonical ID was correctly reused!")
        else:
            print(
                f"❌ FAILED: Canonical ID was not reused. Got {reg_result_2.get('tool_id')}"
            )
    else:
        print(f"❌ Failed to register second tool: {res.status_code}")

    # Check the index to verify there's only one canonical entry
    print("\nStep 4: Checking index for canonical entries...")
    res = requests.post(f"{BASE_URL}/tools/mcp/abstract", json={})
    if res.status_code == 200:
        index_data = res.json()
        tools = index_data.get("tools", [])

        # Find tools with calculate action
        calc_tools = [t for t in tools if t.get("action") == "calculate"]
        print(f"Found {len(calc_tools)} tools with 'calculate' action")

        if len(calc_tools) == 1:
            print("✅ SUCCESS: Only one canonical tool exists!")
            canonical_tool = calc_tools[0]
            print(f"   - Tool ID: {canonical_tool.get('tool_id')}")
            print(f"   - Aliases: {canonical_tool.get('aliases')}")
            print(f"   - Files: {canonical_tool.get('files')}")
        else:
            print(
                f"❌ FAILED: Found {len(calc_tools)} tools instead of 1 canonical tool"
            )
    else:
        print(f"❌ Failed to get index: {res.status_code}")

    # Finally, check the catalog to see if our tool is included
    print("\nStep 5: Checking catalog for canonical tool...")
    res = requests.get(f"{BASE_URL}/tools/mcp/catalog")
    if res.status_code == 200:
        catalog_data = res.json()
        # Find calculator tool by the expected canonical ID from step 4
        canonical_id = reg_result.get("registered")  # This should be the canonical ID
        calc_tools = [t for t in catalog_data if t.get("name") == canonical_id]
        if len(calc_tools) == 1:
            print(f"✅ SUCCESS: Canonical tool '{canonical_id}' found in catalog!")
        else:
            print(f"❌ FAILED: Canonical tool '{canonical_id}' not found in catalog")
            tool_names = [t.get("name") for t in catalog_data]
            print(f"   - Available tools in catalog: {tool_names}")
    else:
        print(f"❌ Failed to get catalog: {res.status_code}")


if __name__ == "__main__":
    test_mcp_registration()
