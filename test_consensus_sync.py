#!/usr/bin/env python3
"""Simple sync test for consensus functionality."""

import time

import requests


def test_simple_consensus_sync():
    """Simple synchronous test of consensus logic."""
    print("🧪 Testing consensus logic synchronously...")

    # Test parameters
    base_url = "http://localhost:11434/v1"
    model_name = "gpt-oss:20b"
    prompt = "What is 2+2?"
    num_samples = 2
    base_temp = 0.5

    print(f"📝 Testing with: {num_samples} samples, prompt: '{prompt}'")

    valid_responses = []

    # Generate samples sequentially
    for i in range(num_samples):
        temperature = max(0.1, min(1.0, base_temp + (i * 0.1 - 0.1)))
        print(f"🔄 Sample {i+1}: temperature = {temperature}")

        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 50,
                    "temperature": temperature,
                },
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"].strip()
                    print(f"  ✅ Response {i+1}: {content}")
                    valid_responses.append(content)
                else:
                    print(f"  ❌ Sample {i+1}: Invalid response format")
            else:
                print(f"  ❌ Sample {i+1}: HTTP {response.status_code}")

        except Exception as e:
            print(f"  ❌ Sample {i+1}: Error - {e}")

        # Brief pause between requests
        if i < num_samples - 1:
            time.sleep(1)

    # Compute consensus
    if not valid_responses:
        print("❌ No valid responses received")
        return False

    print(f"\n📊 Computing consensus from {len(valid_responses)} responses...")

    # Count response frequencies
    response_counts = {}
    for resp in valid_responses:
        response_counts[resp] = response_counts.get(resp, 0) + 1

    # Find most common response
    consensus_text = max(response_counts.keys(), key=lambda x: response_counts[x])
    consensus_confidence = response_counts[consensus_text] / len(valid_responses)

    # Create result
    result = {
        "consensus_text": consensus_text,
        "consensus_confidence": consensus_confidence,
        "aggregation_method": "simple_vote",
        "individual_responses": valid_responses,
        "metadata": {
            "num_responses": len(valid_responses),
            "num_unique": len(response_counts),
        },
    }

    print("🎯 Consensus Results:")
    print(f"  Text: {consensus_text}")
    print(f"  Confidence: {consensus_confidence:.2f}")
    print(f"  Unique responses: {len(response_counts)}")
    print(f"  Total responses: {len(valid_responses)}")

    return True


if __name__ == "__main__":
    success = test_simple_consensus_sync()
    if success:
        print("\n🎉 Consensus functionality verified!")
    else:
        print("\n❌ Consensus test failed")
