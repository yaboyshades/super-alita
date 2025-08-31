#!/usr/bin/env python3
"""
Simple consensus sampling tool for Super Alita + Ollama integration
This bypasses the complex DeepConf pipeline to provide basic consensus functionality
"""

import asyncio
import json
from typing import Any, Dict, List
import httpx


class SimpleConsensusProvider:
    """Simple consensus provider using direct API calls"""

    def __init__(self, base_url: str, model_name: str, timeout: float = 60.0):
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout

    async def generate_response(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 512
    ) -> str:
        """Generate a single response from the model"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant. Give clear, direct answers.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {str(e)}"

    async def consensus_sampling(
        self,
        prompt: str,
        num_samples: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """Perform consensus sampling with multiple responses"""

        print(f"🧠 Generating {num_samples} responses for consensus...")

        # Generate multiple responses
        tasks = []
        for i in range(num_samples):
            # Vary temperature slightly for diversity
            temp = temperature + (i * 0.1 - 0.1)  # -0.1, 0.0, +0.1 for 3 samples
            temp = max(0.1, min(1.0, temp))  # Keep in valid range
            tasks.append(self.generate_response(prompt, temp, max_tokens))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out error responses
        valid_responses = [
            r for r in responses if isinstance(r, str) and not r.startswith("Error:")
        ]

        if not valid_responses:
            return {
                "consensus_text": "Error: No valid responses generated",
                "consensus_confidence": 0.0,
                "aggregation_method": "simple",
                "individual_responses": [str(r) for r in responses],
                "metadata": {"error": "No valid responses"},
            }

        # Simple consensus: pick the most common response, or the first if all unique
        response_counts = {}
        for resp in valid_responses:
            response_counts[resp] = response_counts.get(resp, 0) + 1

        # Get the most common response
        consensus_text = max(response_counts.keys(), key=lambda x: response_counts[x])
        consensus_confidence = response_counts[consensus_text] / len(valid_responses)

        print(f"✅ Consensus complete: {len(valid_responses)} valid responses")

        return {
            "consensus_text": consensus_text,
            "consensus_confidence": consensus_confidence,
            "aggregation_method": "simple_vote",
            "individual_responses": valid_responses,
            "metadata": {
                "num_responses": len(valid_responses),
                "num_unique": len(response_counts),
                "temperature_range": f"{temperature-0.1:.1f}-{temperature+0.1:.1f}",
            },
        }


async def test_simple_consensus():
    """Test the simple consensus provider"""
    try:
        print("🚀 Testing Simple Consensus Provider")
        print("=" * 50)

        provider = SimpleConsensusProvider(
            base_url="http://localhost:11434/v1", model_name="gpt-oss:20b", timeout=60.0
        )

        # Test single response first
        print("🧪 Testing single response...")
        single_response = await provider.generate_response("What is 2+2?", 0.1, 20)
        print(f"📝 Single response: {single_response}")

        # Test consensus sampling
        print("\n🧪 Testing consensus sampling...")
        consensus_result = await provider.consensus_sampling(
            prompt="What is 2+2? Give a brief answer.",
            num_samples=2,
            temperature=0.2,
            max_tokens=30,
        )

        print(f"📝 Consensus: {consensus_result['consensus_text']}")
        print(f"🎯 Confidence: {consensus_result['consensus_confidence']:.3f}")
        print(f"📊 Method: {consensus_result['aggregation_method']}")
        print(f"🔢 Responses: {len(consensus_result['individual_responses'])}")

        return consensus_result

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = asyncio.run(test_simple_consensus())
    if result:
        print("\n🎉 Simple consensus test successful!")
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print("\n⚠️ Simple consensus test failed")
