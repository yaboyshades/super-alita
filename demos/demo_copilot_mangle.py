#!/usr/bin/env python3
"""
GitHub Copilot Mangle Integration Demo.

This script demonstrates how GitHub Copilot questions are automatically
enhanced with Mangle reasoning and constitutional compliance.
"""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Enable Mangle mode
os.environ["COPILOT_MANGLE_MODE"] = "true"


def demo_enhanced_copilot():
    """Demonstrate enhanced GitHub Copilot responses."""

    print("🧠 GitHub Copilot + Mangle Integration Demo")
    print("=" * 50)

    # Sample questions that would be asked to GitHub Copilot
    sample_questions = [
        "What functions are untested?",
        "What violates the constitution?",
        "How can I improve code quality?",
        "What libraries should I use?",
        "Is this function too complex?",
    ]

    for i, question in enumerate(sample_questions, 1):
        print(f"\n{i}. User asks GitHub Copilot: '{question}'")
        print("   " + "-" * 40)

        # Simulate enhanced response
        enhanced_response = enhance_copilot_response(question)
        print(f"   GitHub Copilot (Enhanced): {enhanced_response}")


def enhance_copilot_response(question: str) -> str:
    """Enhance a GitHub Copilot response with Mangle reasoning."""

    # Try to use actual Mangle integration
    try:
        from sdd.mangle_rules import get_query_for_question

        query = get_query_for_question(question)
        if query:
            return f"""
🧠 I can analyze that with Mangle reasoning!

Your question maps to the query: '{query}'

Let me check your codebase... I found several items that match this pattern.
Here's what I recommend:

1. Follow Constitutional Article II (Test-First Development)
2. Keep functions under 50 lines (Simplicity Gate)
3. Use existing libraries when possible (Library-First)

Would you like me to show specific examples from your code?"""

    except Exception:
        pass

    # Fallback enhanced response
    return f"""
🧠 Enhanced with constitutional guidance!

I can help with '{question}' using these principles:

• Library-First: Research existing solutions first
• Test-First: Write tests before implementation
• Simplicity: Keep code clear and focused
• Integration: Test with real dependencies
• Clarity: Write unambiguous code
• Counterfactual: Consider alternatives

Let me provide specific recommendations for your code..."""


def show_integration_benefits():
    """Show the benefits of Mangle integration."""

    print("\n🎯 Benefits of GitHub Copilot + Mangle Integration:")
    print("=" * 50)

    benefits = [
        "🔍 Automatic code knowledge graph analysis",
        "🏛️ Constitutional compliance checking (6 articles)",
        "📊 Multi-dimensional quality assessment",
        "🔗 Specification-to-code traceability",
        "💡 Natural language code querying",
        "⚡ Contextual improvement suggestions",
        "🎯 Zero workflow disruption - works with existing Copilot",
    ]

    for benefit in benefits:
        print(f"   {benefit}")


def show_example_queries():
    """Show example queries that work with the integration."""

    print("\n📝 Example Questions You Can Ask:")
    print("=" * 50)

    try:
        from sdd.mangle_rules import get_available_queries

        queries = get_available_queries()[:8]

        for i, query_pattern in enumerate(queries, 1):
            print(f"   {i}. {query_pattern.title()}")

    except Exception:
        # Fallback examples
        examples = [
            "What functions are untested",
            "What violates constitution",
            "Quality issues",
            "Complex functions",
            "Incomplete features",
            "Circular dependencies",
            "Orphaned specifications",
            "Library first violations",
        ]

        for i, example in enumerate(examples, 1):
            print(f"   {i}. {example}")


def main():
    """Run the demo."""

    # Run the demo
    demo_enhanced_copilot()

    # Show benefits
    show_integration_benefits()

    # Show example queries
    show_example_queries()

    print("\n" + "=" * 50)
    print("✅ This is how GitHub Copilot works with Mangle integration!")
    print("🚀 Every question automatically gets enhanced reasoning.")
    print("💡 Constitutional compliance and quality guidance built-in.")
    print("=" * 50)


if __name__ == "__main__":
    main()
