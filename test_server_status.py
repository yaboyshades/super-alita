#!/usr/bin/env python3
"""Test endpoint to check ability registry status."""

import requests
import json


def test_server_status():
    """Test the server and its components."""
    base_url = "http://127.0.0.1:8080"

    try:
        print("🌐 Testing server endpoints...")

        # Test health
        print("\n📊 Health check:")
        health = requests.get(f"{base_url}/healthz", timeout=5)
        print(f"  Status: {health.status_code}")
        print(f"  Response: {health.json()}")

        # Test tools catalog
        print("\n🛠️ Tools catalog:")
        catalog = requests.get(f"{base_url}/tools/catalog", timeout=5)
        print(f"  Status: {catalog.status_code}")
        tools = catalog.json()
        print(f"  Tool count: {len(tools)}")

        # Look for consensus tools
        consensus_tools = [
            t
            for t in tools
            if "consensus" in t.get("name", "").lower()
            or "deepconf" in t.get("name", "").lower()
        ]
        print(f"  Consensus tools: {len(consensus_tools)}")

        if consensus_tools:
            for tool in consensus_tools:
                print(f"    - {tool['name']}: {tool['description']}")
        else:
            print("  Available tools:")
            for tool in tools[:10]:  # Show first 10
                print(f"    - {tool['name']}")

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    test_server_status()
