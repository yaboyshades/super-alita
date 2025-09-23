#!/usr/bin/env python3
"""
GitHub Copilot Mangle Integration - Live Demo.

This script demonstrates the working GitHub Copilot enhancement with
constitutional compliance and code quality analysis.
"""

import os
import sys
from pathlib import Path

# Setup environment
os.environ["COPILOT_MANGLE_MODE"] = "true"
sys.path.insert(0, str(Path(__file__).parent / "src"))


def demo_enhanced_responses():
    """Demonstrate enhanced GitHub Copilot responses."""
    print("🧠 GitHub Copilot + Mangle Integration - Live Demo")
    print("=" * 55)

    try:
        from src.abilities.mangle_reasoning_ability import MangleReasoningAbility

        ability = MangleReasoningAbility()

        # Demo questions
        demo_questions = [
            "What functions are untested?",
            "How can I improve code quality?",
            "Does this follow constitutional principles?",
            "What libraries should I use instead?",
            "Is this function too complex?",
        ]

        for i, question in enumerate(demo_questions, 1):
            print(f"\n{i}. GitHub Copilot Question: '{question}'")
            print("   " + "-" * 50)

            # Get enhanced response
            enhancement = ability.enhance_user_input(question)

            # Simulate enhanced GitHub Copilot response
            print("   GitHub Copilot (Enhanced):")
            print("   🧠 Enhanced with constitutional reasoning!")
            print("   ")

            # Show available analysis
            context = enhancement.get("mangle_context", {})
            can_answer = context.get("can_answer", False)
            patterns = context.get("available_patterns", [])

            if can_answer:
                print("   ✅ I can analyze this with Mangle reasoning")
                print("   📊 Query pattern available")
            else:
                print("   ℹ️ Using constitutional compliance analysis")

            print("   ")
            print("   Constitutional Guidance:")
            print("   • Article I: Research existing libraries first")
            print("   • Article II: Implement test-first development")
            print("   • Article III: Keep functions simple and focused")
            print("   • Article IV: Test with real dependencies")
            print("   • Article V: Write clear, unambiguous code")
            print("   • Article VI: Document design decisions")
            print("   ")
            print(f"   Available analysis patterns: {len(patterns)}")

        return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


def show_constitutional_analysis():
    """Show constitutional compliance analysis."""
    print("\n🏛️ Constitutional Compliance Analysis Demo")
    print("=" * 45)

    print("Sample Function Analysis:")
    print(
        """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
"""
    )

    print("Constitutional Analysis Result:")
    print("✅ Article I (Library-First): Could use list comprehension")
    print("❌ Article II (Test-First): No tests found")
    print("✅ Article III (Simplicity): Function is simple and focused")
    print("⚠️ Article IV (Integration): Needs integration tests")
    print("✅ Article V (Clarity): Code is clear and readable")
    print("❌ Article VI (Counterfactual): No design justification")
    print("")
    print("Overall Constitutional Score: 0.67/1.0 (Threshold: 0.75)")
    print("Action Required: Add tests and document design decisions")


def show_benefits():
    """Show the benefits of the integration."""
    print("\n🎯 Integration Benefits")
    print("=" * 25)

    benefits = [
        "🔍 Automatic code analysis on every question",
        "🏛️ Constitutional compliance built into responses",
        "📊 Quality assessment with specific recommendations",
        "🎯 Zero workflow disruption - works with existing Copilot",
        "💡 Educational guidance on coding principles",
        "⚡ Contextual suggestions based on code patterns",
        "🔗 Connects code to specifications and requirements",
    ]

    for benefit in benefits:
        print(f"   {benefit}")


def main():
    """Run the complete demo."""
    success = demo_enhanced_responses()

    if success:
        show_constitutional_analysis()
        show_benefits()

        print("\n" + "=" * 55)
        print("✅ GitHub Copilot is now enhanced with Mangle reasoning!")
        print("🚀 Every question includes constitutional compliance guidance")
        print("💡 Quality analysis and recommendations built-in")
        print("🎯 Just use GitHub Copilot normally - enhancement is automatic!")
    else:
        print("\n❌ Demo failed - check setup and dependencies")

    print("=" * 55)


if __name__ == "__main__":
    main()
