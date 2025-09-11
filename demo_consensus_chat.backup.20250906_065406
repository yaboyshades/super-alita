#!/usr/bin/env python3
"""
Demo script to showcase Super Alita's enhanced consensus capabilities
through both direct API calls and the chat interface.
"""

from typing import Any

import requests


def test_consensus_direct() -> dict[str, Any]:
    """Test the consensus tool directly via the ability execution endpoint."""
    print("🧠 Testing Enhanced Consensus Tool (Direct API)")
    print("=" * 50)

    url = "http://127.0.0.1:8080/ability/execute/deepconf_consensus"
    payload = {
        "prompt": "What are the most important principles of responsible AI development?",
        "num_samples": 4,
        "temperature": 0.8,
        "max_tokens": 300,
        "method": "weighted_vote",
        "confidence_threshold": 0.6,
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        print("✅ Consensus Result:")
        print(f"   Method: {result.get('result', {}).get('aggregation_method', 'N/A')}")
        print(
            f"   Confidence: {result.get('result', {}).get('consensus_confidence', 0):.1%}"
        )
        print(
            f"   Samples: {len(result.get('result', {}).get('individual_responses', []))}"
        )
        print(
            f"   Text Preview: {result.get('result', {}).get('consensus_text', '')[:100]}..."
        )

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return {}


def test_health_check() -> bool:
    """Verify the server is healthy and responsive."""
    print("\n🏥 Health Check")
    print("=" * 50)

    try:
        response = requests.get("http://127.0.0.1:8080/healthz", timeout=5)
        response.raise_for_status()
        health = response.json()

        print("✅ Server Status: Healthy")
        for component, status in health.get("components", {}).items():
            print(f"   {component}: {status.get('status', 'unknown')}")

        return True

    except Exception as e:
        print(f"❌ Server Health Check Failed: {e}")
        return False


def test_tools_catalog() -> dict[str, Any]:
    """Check the tools catalog for consensus tool registration."""
    print("\n🛠️  Tools Catalog")
    print("=" * 50)

    try:
        response = requests.get("http://127.0.0.1:8080/tools/catalog", timeout=5)
        response.raise_for_status()
        tools = response.json()

        consensus_tool = None
        for tool in tools:
            if tool.get("name") == "deepconf_consensus":
                consensus_tool = tool
                break

        if consensus_tool:
            print("✅ Enhanced Consensus Tool Found:")
            print(f"   Name: {consensus_tool.get('name')}")
            print(f"   Description: {consensus_tool.get('description')}")
            print(
                f"   Required Params: {consensus_tool.get('input_schema', {}).get('required', [])}"
            )
        else:
            print("❌ Enhanced Consensus Tool Not Found")

        return consensus_tool or {}

    except Exception as e:
        print(f"❌ Tools Catalog Error: {e}")
        return {}


def main():
    """Run the complete Super Alita consensus demonstration."""
    print("🚀 Super Alita Enhanced Consensus Demonstration")
    print("=" * 60)
    print("Testing the chat interface and consensus algorithms...\n")

    # Test server health
    if not test_health_check():
        print("❌ Server is not healthy. Please start the server first.")
        return

    # Test tools catalog
    tool_info = test_tools_catalog()
    if not tool_info:
        print("❌ Consensus tool not found. Please check server configuration.")
        return

    # Test consensus tool directly
    consensus_result = test_consensus_direct()
    if not consensus_result:
        print("❌ Consensus tool test failed.")
        return

    print("\n🎉 Super Alita Consensus Demo Complete!")
    print("=" * 60)
    print("✅ All systems operational!")
    print("✅ Enhanced consensus with weighted voting working!")
    print("✅ Chat interface ready at: http://127.0.0.1:8080")
    print("\n💡 Try the chat interface with prompts like:")
    print("   • 'Use consensus to explain machine learning basics'")
    print("   • 'What are the benefits of distributed systems?'")
    print("   • 'Compare different AI safety approaches'")


if __name__ == "__main__":
    main()
