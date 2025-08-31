#!/usr/bin/env python3
"""
Mangle Integration Validation Script

This script validates that the Mangle integration is working correctly
by testing basic functionality.
"""

import asyncio
import sys
from pathlib import Path

import requests

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.absolute()))


async def validate_mangle_integration():
    """Validate that the Mangle integration is working correctly."""
    print("=== Mangle Integration Validation ===")

    # Check if server is running
    try:
        health_response = requests.get("http://127.0.0.1:8080/healthz")
        if health_response.status_code != 200:
            print("❌ Super Alita server is not running or not healthy!")
            print(f"   Status Code: {health_response.status_code}")
            return False

        print("✅ Super Alita server is running and healthy")
    except Exception as e:
        print(f"❌ Failed to connect to Super Alita server: {e}")
        print("   Make sure the server is running with: uvicorn app:app --reload --port 8080")
        return False

    # Check for tool catalog to see if Mangle tools are registered
    try:
        catalog_response = requests.get("http://127.0.0.1:8080/tools/catalog")
        if catalog_response.status_code != 200:
            print("❌ Failed to retrieve tool catalog!")
            return False

        catalog = catalog_response.json()

        # Look for Mangle-related tools
        mangle_tools = [tool for tool in catalog if "mangle" in tool.get("name", "")]

        if not mangle_tools:
            print("❌ No Mangle tools found in the catalog!")
            return False

        print(f"✅ Found {len(mangle_tools)} Mangle tools in the catalog:")
        for tool in mangle_tools:
            print(f"   - {tool.get('name')}: {tool.get('description')}")

    except Exception as e:
        print(f"❌ Failed to check tool catalog: {e}")
        return False

    # Test direct tool execution
    try:
        # Add a test fact
        fact_response = requests.post(
            "http://127.0.0.1:8080/ability/execute/mangle_add_fact",
            json={"fact": "test_component('validation')"}
        )

        if fact_response.status_code != 200:
            print("❌ Failed to add test fact!")
            return False

        fact_result = fact_response.json()
        if fact_result.get("success") is not True:
            print("❌ Adding test fact returned failure!")
            return False

        print("✅ Successfully added test fact to knowledge base")

        # Add a test rule
        rule_response = requests.post(
            "http://127.0.0.1:8080/ability/execute/mangle_add_rule",
            json={
                "name": "test_rule",
                "rule": "validated(X) :- test_component(X)"
            }
        )

        if rule_response.status_code != 200:
            print("❌ Failed to add test rule!")
            return False

        rule_result = rule_response.json()
        if rule_result.get("success") is not True:
            print("❌ Adding test rule returned failure!")
            return False

        print("✅ Successfully added test rule to knowledge base")

        # Execute a query
        query_response = requests.post(
            "http://127.0.0.1:8080/ability/execute/mangle_query",
            json={"query": "validated(X)"}
        )

        if query_response.status_code != 200:
            print("❌ Failed to execute test query!")
            return False

        query_result = query_response.json()
        if not query_result.get("success"):
            print("❌ Query execution returned failure!")
            return False

        if query_result.get("count", 0) == 0:
            print("❌ Query returned no results!")
            return False

        print("✅ Successfully executed test query with expected results")

    except Exception as e:
        print(f"❌ Failed to test direct tool execution: {e}")
        return False

    print("\n✅ Mangle integration validation PASSED!")
    print("Mangle is successfully integrated with Super Alita")
    return True


if __name__ == "__main__":
    result = asyncio.run(validate_mangle_integration())
    sys.exit(0 if result else 1)
