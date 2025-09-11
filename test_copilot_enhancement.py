#!/usr/bin/env python3
"""
Quick test of GitHub Copilot Mangle integration.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_copilot_enhancement():
    """Test the enhanced GitHub Copilot functionality."""
    print("🧪 Testing GitHub Copilot Mangle Enhancement")
    print("-" * 45)

    try:
        # Test the enhanced ability
        from src.abilities.mangle_reasoning_ability import MangleReasoningAbility

        ability = MangleReasoningAbility()

        # Test enhancement
        test_question = "What functions are untested in my code?"
        enhancement = ability.enhance_user_input(test_question)

        print(f"Question: {test_question}")
        print(f"Can answer: {enhancement['mangle_context']['can_answer']}")
        print(
            f"Available patterns: {len(enhancement['mangle_context']['available_patterns'])}"
        )

        print("\n✅ GitHub Copilot Mangle integration working!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def show_example_interaction():
    """Show what an enhanced GitHub Copilot interaction looks like."""
    print("\n🎯 Example Enhanced GitHub Copilot Interaction:")
    print("-" * 45)

    print("User: 'How can I improve the quality of this function?'")
    print("")
    print("GitHub Copilot (Enhanced): 🧠 Enhanced with Mangle reasoning!")
    print("")
    print("I can analyze your code for quality improvements using:")
    print("• Constitutional compliance (6 articles)")
    print("• Complexity analysis and recommendations")
    print("• Test coverage assessment")
    print("• Library-first principle validation")
    print("• Specification traceability")
    print("")
    print("Let me check your specific function...")


if __name__ == "__main__":
    success = test_copilot_enhancement()
    if success:
        show_example_interaction()
    print("\n" + "=" * 45)
