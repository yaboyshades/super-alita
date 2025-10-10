#!/usr/bin/env python3
"""Test script for snippet optimization system."""

import asyncio

from src.core.copilot_snippet_optimizer import (
    NeuralAtomMetadata,
    SnippetIntelligenceAtom,
)


async def test_snippet_optimization():
    """Test the snippet optimization system functionality."""
    metadata = NeuralAtomMetadata(
        name="test_snippet_atom",
        description="Test snippet intelligence",
        capabilities=["snippet_analysis"],
    )

    atom = SnippetIntelligenceAtom(metadata)

    # Test snippet suggestion
    result = await atom.execute(
        {
            "operation": "suggest_snippets",
            "context": "I want to create a function that processes data",
            "code_intent": "create a function with loops and error handling",
            "estimated_tokens": 100,
        }
    )

    print("🎯 Snippet Suggestions Test:")
    print(f"  Suggestions found: {len(result.get('suggestions', []))}")
    print(
        f"  Total estimated savings: {result.get('total_estimated_savings', 0)} tokens"
    )

    # Show suggestions if any
    if result.get("suggestions"):
        print("  Top suggestions:")
        for i, suggestion in enumerate(result["suggestions"][:3], 1):
            print(
                f"    {i}. {suggestion['trigger']} - {suggestion['description']}"
            )
            print(f"       Saves ~{suggestion['estimated_savings']} tokens")

    # Test token savings calculation
    savings = await atom.execute(
        {
            "operation": "calculate_savings",
            "approach": "snippet",
            "baseline_tokens": 100,
        }
    )

    print("\n📈 Token Efficiency Test:")
    print(f"  Tokens saved: {savings.get('tokens_saved', 0)}")
    print(
        f"  Efficiency rating: {savings.get('efficiency_rating', 'unknown')}"
    )
    print(f"  Savings percent: {savings.get('savings_percent', 0):.1%}")

    # Test optimization response generation
    optimization = await atom.execute(
        {
            "operation": "optimize_response",
            "user_request": "Create a Python class with methods",
            "context": "object-oriented programming with inheritance",
        }
    )

    print("\n🔀 Response Optimization Test:")
    print(f"  Approach: {optimization.get('approach', 'none')}")
    print(f"  Token cost: {optimization.get('token_cost', 0)}")
    print(f"  Estimated savings: {optimization.get('estimated_savings', 0)}")

    print("\n✅ Snippet optimization system functional!")


if __name__ == "__main__":
    asyncio.run(test_snippet_optimization())
