#!/usr/bin/env python3
"""Simple test to verify consensus tool execution."""

import requests


def test_simple_consensus():
    """Test consensus tool via direct HTTP call."""
    print("🧪 Testing consensus tool...")

    # Test if the tool is in catalog first
    try:
        catalog_response = requests.get(
            "http://127.0.0.1:8080/tools/catalog", timeout=10
        )
        if catalog_response.status_code == 200:
            tools = catalog_response.json()
            consensus_tools = [
                t for t in tools if t.get("name") == "deepconf_consensus"
            ]
            if consensus_tools:
                print("✅ Consensus tool found in catalog")
                print(
                    f"   Description: {consensus_tools[0].get('description', 'N/A')}"
                )
            else:
                print("❌ Consensus tool not found in catalog")
                return
        else:
            print(f"❌ Failed to get catalog: {catalog_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Catalog error: {e}")
        return

    # Now let's test if we can trigger the consensus tool through REUG
    start_payload = {
        "message": "Use deepconf_consensus with prompt 'Hello world' and 2 samples"
    }

    try:
        print("🚀 Starting REUG turn to test consensus tool...")
        response = requests.post(
            "http://127.0.0.1:8080/tools/reug_start_turn",
            json=start_payload,
            timeout=30,
        )

        print(f"📤 Response status: {response.status_code}")
        print(f"📤 Response body: {response.text[:500]}...")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_simple_consensus()
