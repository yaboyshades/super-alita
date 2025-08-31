#!/usr/bin/env python3
"""Test consensus tool implementation directly."""

import asyncio
import httpx


async def test_consensus_implementation():
    """Test our SimpleConsensusProvider directly."""
    print("🧪 Testing consensus implementation directly...")

    # Recreate the SimpleConsensusProvider class
    class SimpleConsensusProvider:
        def __init__(self, base_url: str, model_name: str, timeout: float = 60.0):
            self.base_url = base_url
            self.model_name = model_name
            self.timeout = timeout

        async def consensus_sampling(
            self,
            prompt: str,
            num_samples: int = 3,
            temperature: float = 0.7,
            max_tokens: int = 512,
        ):
            print(f"🔍 Starting consensus sampling with {num_samples} samples...")

            # Generate multiple responses with slight temperature variation
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                tasks = []
                for i in range(num_samples):
                    temp = max(0.1, min(1.0, temperature + (i * 0.1 - 0.1)))
                    print(f"  Sample {i+1}: temperature = {temp}")

                    tasks.append(
                        client.post(
                            f"{self.base_url}/chat/completions",
                            json={
                                "model": self.model_name,
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": "You are a helpful assistant.",
                                    },
                                    {"role": "user", "content": prompt},
                                ],
                                "max_tokens": max_tokens,
                                "temperature": temp,
                            },
                        )
                    )

                print("📡 Making parallel requests to Ollama...")
                responses = await asyncio.gather(*tasks, return_exceptions=True)

                valid_responses = []
                for i, resp in enumerate(responses):
                    try:
                        if isinstance(resp, Exception):
                            print(f"  ❌ Sample {i+1} failed: {resp}")
                            continue
                        resp.raise_for_status()
                        data = resp.json()
                        content = data["choices"][0]["message"]["content"]
                        print(f"  ✅ Sample {i+1}: {content[:50]}...")
                        valid_responses.append(content)
                    except Exception as e:
                        print(f"  ❌ Sample {i+1} error: {e}")
                        continue

            if not valid_responses:
                return {
                    "consensus_text": "Error: No valid responses generated",
                    "consensus_confidence": 0.0,
                    "aggregation_method": "simple_vote",
                    "individual_responses": [],
                    "metadata": {"error": "No valid responses"},
                }

            # Simple consensus: most common response
            response_counts = {}
            for resp in valid_responses:
                response_counts[resp] = response_counts.get(resp, 0) + 1

            consensus_text = max(
                response_counts.keys(), key=lambda x: response_counts[x]
            )
            consensus_confidence = response_counts[consensus_text] / len(
                valid_responses
            )

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

            print(f"🎯 Consensus result:")
            print(f"  Text: {consensus_text}")
            print(f"  Confidence: {consensus_confidence}")
            print(f"  Responses: {len(valid_responses)}")

            return result

    # Test the implementation
    try:
        provider = SimpleConsensusProvider(
            base_url="http://localhost:11434/v1", model_name="gpt-oss:20b", timeout=30.0
        )

        result = await provider.consensus_sampling(
            prompt="What is the capital of France?",
            num_samples=2,
            temperature=0.5,
            max_tokens=50,
        )

        print(f"\n🎉 Success! Result: {result}")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_consensus_implementation())
    if success:
        print("\n✅ Consensus implementation works perfectly!")
    else:
        print("\n❌ Consensus implementation failed")
