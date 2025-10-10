#!/usr/bin/env python3
"""Direct test of enhanced consensus algorithms."""

import asyncio
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from abilities.enhanced_consensus_ability import EnhancedConsensusProvider


async def test_enhanced_consensus_direct():
    """Test enhanced consensus algorithms directly."""
    print("🧪 Direct Test of Enhanced Consensus Algorithms")
    print("=" * 60)

    # Initialize provider
    provider = EnhancedConsensusProvider(
        {
            "base_url": "http://localhost:11434/v1",
            "model_name": "gpt-oss:20b",
            "timeout": 60.0,
        }
    )

    await provider.initialize()

    # Test prompt
    prompt = "What is the capital of France?"

    # Test all methods
    methods = [
        "simple_vote",
        "weighted_vote",
        "confidence_based",
        "semantic_similarity",
        "ensemble_ranking",
    ]

    results = {}

    for i, method in enumerate(methods, 1):
        print(f"\n{i}️⃣ Testing {method.upper()}")
        print("-" * 40)

        try:
            result = await provider.consensus_sampling(
                prompt=prompt,
                num_samples=3,
                temperature=0.7,
                max_tokens=100,
                method=method,
                confidence_threshold=0.7,
                temperature_range=0.2,
            )

            print("✅ Success!")
            print(f"📝 Consensus: {result['consensus_text'][:80]}...")
            print(f"📊 Confidence: {result['consensus_confidence']:.3f}")
            print(f"🔢 Responses: {len(result['individual_responses'])}")
            print(f"📈 Method: {result['aggregation_method']}")

            # Method-specific details
            metadata = result.get("metadata", {})
            if method == "weighted_vote":
                total_weight = metadata.get("total_weight", 0)
                print(f"⚖️ Total Weight: {total_weight:.3f}")

            elif method == "confidence_based":
                threshold = metadata.get("threshold", 0)
                qualified = metadata.get("qualified_responses", 0)
                fallback = metadata.get("fallback_used", False)
                print(
                    f"🎯 Threshold: {threshold}, Qualified: {qualified}, Fallback: {fallback}"
                )

            elif method == "semantic_similarity":
                similarity_scores = metadata.get("similarity_scores", [])
                if similarity_scores:
                    avg_sim = sum(similarity_scores) / len(similarity_scores)
                    print(f"🔗 Avg Similarity: {avg_sim:.3f}")

            elif method == "ensemble_ranking":
                components = metadata.get("scoring_components", [])
                ensemble_scores = metadata.get("ensemble_scores", [])
                print(f"🏆 Components: {components}")
                if ensemble_scores:
                    print(
                        f"📊 Scores: {[round(s, 3) for s in ensemble_scores]}"
                    )

            results[method] = {"success": True, "result": result}

        except Exception as e:
            print(f"❌ Failed: {e}")
            results[method] = {"success": False, "error": str(e)}

    # Summary
    print("\n📊 SUMMARY")
    print("=" * 60)
    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)
    print(f"✅ Successful: {successful}/{total}")
    print(f"📈 Success Rate: {(successful/total)*100:.1f}%")

    for method, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"{status} {method}")

    if successful == total:
        print("\n🎉 ALL ENHANCED CONSENSUS ALGORITHMS WORKING!")
        print("✅ Ready for production deployment!")
    else:
        print(f"\n⚠️ {total - successful} algorithm(s) need attention")

    return results


if __name__ == "__main__":
    asyncio.run(test_enhanced_consensus_direct())
