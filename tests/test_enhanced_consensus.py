#!/usr/bin/env python3
"""Test enhanced consensus algorithms with multiple methods."""

import asyncio

import httpx


async def test_enhanced_consensus_methods():
    """Test all enhanced consensus methods."""
    print("🧪 Testing Enhanced Consensus Methods...")

    base_url = "http://127.0.0.1:8080"
    test_prompt = "What is the capital of France? Be brief and specific."

    methods_to_test = [
        "simple_vote",
        "weighted_vote",
        "confidence_based",
        "semantic_similarity",
        "ensemble_ranking",
    ]

    async with httpx.AsyncClient(timeout=60.0) as client:
        for method in methods_to_test:
            print(f"\n🔍 Testing {method} method...")

            try:
                # Test via the tool endpoint
                response = await client.post(
                    f"{base_url}/tools/deepconf_consensus",
                    json={
                        "prompt": test_prompt,
                        "num_samples": 3,
                        "temperature": 0.6,
                        "max_tokens": 100,
                        "method": method,
                        "confidence_threshold": 0.7,
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ {method}:")
                    print(
                        f"   Consensus: {result.get('consensus_text', 'N/A')}"
                    )
                    print(
                        f"   Confidence: {result.get('consensus_confidence', 0):.3f}"
                    )
                    print(
                        f"   Method: {result.get('aggregation_method', 'N/A')}"
                    )
                    print(
                        f"   Responses: {len(result.get('individual_responses', []))}"
                    )

                    # Show metadata for more details
                    metadata = result.get("metadata", {})
                    if metadata:
                        print(f"   Metadata: {list(metadata.keys())}")

                else:
                    error_text = response.text
                    print(f"❌ {method}: HTTP {response.status_code}")
                    print(f"   Error: {error_text}")

            except Exception as e:
                print(f"❌ {method}: Exception - {e}")

            # Small delay between tests
            await asyncio.sleep(1)


async def test_consensus_parameter_variations():
    """Test consensus with different parameter variations."""
    print("\n\n🔬 Testing Parameter Variations...")

    base_url = "http://127.0.0.1:8080"

    test_cases = [
        {
            "name": "High Samples (5)",
            "params": {
                "prompt": "Explain quantum computing in one sentence.",
                "num_samples": 5,
                "temperature": 0.5,
                "method": "weighted_vote",
            },
        },
        {
            "name": "High Temperature",
            "params": {
                "prompt": "What makes a good leader?",
                "num_samples": 3,
                "temperature": 0.9,
                "method": "confidence_based",
            },
        },
        {
            "name": "Low Temperature",
            "params": {
                "prompt": "What is 2 + 2?",
                "num_samples": 3,
                "temperature": 0.1,
                "method": "ensemble_ranking",
            },
        },
    ]

    async with httpx.AsyncClient(timeout=90.0) as client:
        for test_case in test_cases:
            print(f"\n🔍 Testing {test_case['name']}...")

            try:
                response = await client.post(
                    f"{base_url}/tools/deepconf_consensus",
                    json=test_case["params"],
                )

                if response.status_code == 200:
                    result = response.json()
                    print("✅ Success:")
                    print(
                        f"   Consensus: {result.get('consensus_text', 'N/A')[:100]}..."
                    )
                    print(
                        f"   Confidence: {result.get('consensus_confidence', 0):.3f}"
                    )
                    print(
                        f"   Responses: {len(result.get('individual_responses', []))}"
                    )
                else:
                    print(f"❌ Failed: HTTP {response.status_code}")

            except Exception as e:
                print(f"❌ Exception: {e}")


async def main():
    """Run all enhanced consensus tests."""
    print("🚀 Enhanced Consensus Algorithm Test Suite")
    print("=" * 50)

    await test_enhanced_consensus_methods()
    await test_consensus_parameter_variations()

    print("\n" + "=" * 50)
    print("🎯 Enhanced Consensus Testing Complete!")


if __name__ == "__main__":
    asyncio.run(main())
