#!/usr/bin/env python3
"""Test script for MCP Abstractor and Catalog.

This script tests the MCP Abstractor's ability to generate both index.json
and catalog.json from tool specs, and verifies that the MCP catalog endpoint
works correctly.
"""

import asyncio
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, "./src")

from src.reug_runtime.mcp_abstractor import abstract_mcp_box


async def test_mcp_abstraction():
    """Test MCP Abstraction and catalog generation."""
    base_url = "http://127.0.0.1:8080"
    box_dir = ".mcp_box"

    # Create test directory and files
    Path(box_dir).mkdir(exist_ok=True)

    # Create a few test specs
    test_specs = [
        {
            "tool_id": "test_calculator",
            "description": "Simple calculator for basic math operations",
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
        },
        {
            "tool_id": "Basic Math",  # Will be normalized
            "description": "Alternative calculator with same signature",
            "action": "calculate",
            "input_schema": {
                "type": "object",
                "required": ["operation", "a", "b"],
                "properties": {
                    "operation": {"type": "string"},
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
            },
            "output_schema": {
                "type": "object",
                "properties": {"result": {"type": "number"}},
            },
        },
    ]

    # Write test specs
    for i, spec in enumerate(test_specs):
        spec_path = Path(box_dir) / f"test_spec_{i}.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        print(f"Created test spec: {spec_path}")

    # Clean up existing index and catalog
    for f in [Path(box_dir) / "index.json", Path(box_dir) / "catalog.json"]:
        if f.exists():
            f.unlink()

    # Call abstract_mcp_box directly
    print("\nTesting local abstraction...")
    result = abstract_mcp_box(box_dir)
    print(f"Abstract result: {len(result['tools'])} tools indexed")

    # Check that both files were created
    index_path = Path(box_dir) / "index.json"
    catalog_path = Path(box_dir) / "catalog.json"

    if index_path.exists():
        print(f"✅ index.json created: {index_path.stat().st_size} bytes")
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        print(f"Index contains {len(index_data['tools'])} tools")
    else:
        print("❌ index.json not created")

    if catalog_path.exists():
        print(f"✅ catalog.json created: {catalog_path.stat().st_size} bytes")
        catalog_data = json.loads(catalog_path.read_text(encoding="utf-8"))
        print(f"Catalog contains {len(catalog_data)} tools")
    else:
        print("❌ catalog.json not created")

    # Test API endpoints
    print("\nTesting API endpoints...")

    # Test abstract endpoint
    try:
        abstract_resp = requests.post(
            f"{base_url}/tools/mcp/abstract", json={}
        )
        if abstract_resp.status_code == 200:
            abstract_data = abstract_resp.json()
            print(
                f"✅ /tools/mcp/abstract endpoint: {len(abstract_data['tools'])} tools"
            )
        else:
            print(
                f"❌ /tools/mcp/abstract endpoint failed: {abstract_resp.status_code}"
            )
    except Exception as e:
        print(f"❌ /tools/mcp/abstract request error: {e}")

    # Test catalog endpoint
    try:
        catalog_resp = requests.get(f"{base_url}/tools/mcp/catalog")
        if catalog_resp.status_code == 200:
            catalog_api_data = catalog_resp.json()
            print(
                f"✅ /tools/mcp/catalog endpoint: {len(catalog_api_data)} tools"
            )
        else:
            print(
                f"❌ /tools/mcp/catalog endpoint failed: {catalog_resp.status_code}"
            )
    except Exception as e:
        print(f"❌ /tools/mcp/catalog request error: {e}")

    # Test tools catalog endpoint to see if it includes MCP tools
    try:
        tools_catalog_resp = requests.get(f"{base_url}/tools/catalog")
        if tools_catalog_resp.status_code == 200:
            tools_catalog_data = tools_catalog_resp.json()
            tool_names = [t.get("name") for t in tools_catalog_data]

            print(
                f"✅ /tools/catalog endpoint: {len(tools_catalog_data)} total tools"
            )

            # Check if our test tool is included
            if "test_calculator" in tool_names:
                print("✅ Test tool found in /tools/catalog!")
            else:
                print("❌ Test tool not found in /tools/catalog")
        else:
            print(
                f"❌ /tools/catalog endpoint failed: {tools_catalog_resp.status_code}"
            )
    except Exception as e:
        print(f"❌ /tools/catalog request error: {e}")


if __name__ == "__main__":
    asyncio.run(test_mcp_abstraction())
