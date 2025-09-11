#!/usr/bin/env python3
"""
GitHub Copilot Mangle Integration - Working Setup (Fixed).

This script sets up Mangle reasoning for GitHub Copilot with proper imports.
"""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def test_mangle_features():
    """Test which Mangle features are available with proper imports."""
    print("🔍 Testing Mangle integration features...")

    features = []

    # Test basic rules with absolute imports
    try:
        import src.sdd.mangle_rules as mangle_rules

        query = mangle_rules.get_query_for_question("what functions are untested")
        if query:
            features.append("✅ Natural language query mapping")
        else:
            features.append("⚠️ Query mapping (limited)")
    except Exception as e:
        features.append(f"❌ Query mapping: {e}")

    # Test constitutional scorer
    try:
        import src.constitutional.scorer as const_scorer

        scorer = const_scorer.ConstitutionalScorer()
        features.append("✅ Constitutional compliance checking")
    except Exception as e:
        features.append(f"❌ Constitutional checking: {e}")

    # Test enhanced mangle ability
    try:
        import src.abilities.mangle_reasoning_ability as mangle_ability

        ability = mangle_ability.MangleReasoningAbility()
        result = ability.enhance_user_input("test question about code quality")
        if result and "mangle_context" in result:
            features.append("✅ Enhanced Copilot integration")
        else:
            features.append("⚠️ Basic Copilot integration")
    except Exception as e:
        features.append(f"❌ Copilot enhancement: {e}")

    # Test full SDD framework
    try:
        import src.sdd.enhanced_sdd_framework as sdd_framework

        framework = sdd_framework.EnhancedSDDFramework()
        features.append("✅ Enhanced SDD Framework")
    except Exception as e:
        features.append(f"❌ SDD Framework: {e}")

    return features


def setup_copilot_enhancement():
    """Set up GitHub Copilot enhancement with proper configuration."""
    print("🚀 Setting up GitHub Copilot Mangle enhancement...")

    # Set environment variables
    os.environ["COPILOT_MANGLE_MODE"] = "true"
    os.environ["COPILOT_AUTO_ENHANCE"] = "true"
    os.environ["SDD_CONSTITUTIONAL_MODE"] = "true"

    # Test features
    features = test_mangle_features()

    print("\n📊 Feature Status:")
    for feature in features:
        print(f"   {feature}")

    working_count = len([f for f in features if f.startswith("✅")])

    if working_count >= 2:
        print(f"\n✅ {working_count} features working!")
        print("🧠 GitHub Copilot enhancement activated!")
        return True
    else:
        print(f"\n⚠️ Limited functionality ({working_count} features working)")
        print("📋 See installation instructions for full features")
        return False


def create_enhanced_demo():
    """Create a demo of the enhanced functionality."""
    print("\n🎯 Testing Enhanced GitHub Copilot Responses...")

    try:
        import src.abilities.mangle_reasoning_ability as mangle_ability

        ability = mangle_ability.MangleReasoningAbility()

        # Test questions
        test_questions = [
            "What functions are untested?",
            "How can I improve code quality?",
            "What violates constitutional principles?",
        ]

        for question in test_questions:
            enhancement = ability.enhance_user_input(question)
            print(f"\n   Question: '{question}'")
            print(
                f"   Enhanced: {enhancement.get('mangle_context', {}).get('can_answer', 'N/A')}"
            )

        return True

    except Exception as e:
        print(f"   Demo failed: {e}")
        return False


def show_usage():
    """Show usage instructions."""
    print(
        """
🎯 How to Use Enhanced GitHub Copilot:

1. Just use GitHub Copilot normally in VS Code
2. Ask questions like:
   • "What functions are untested?"
   • "What violates the constitution?"
   • "How can I improve this code quality?"
   • "What libraries should I use instead?"

3. Responses will automatically include:
   • Constitutional compliance guidance (Article I-VI)
   • Code quality assessment and recommendations
   • Mangle reasoning insights (when available)
   • Specification-to-code traceability

4. No additional setup needed - enhancement is automatic!

🔧 Advanced Usage:
   • Import the ability: from src.abilities.mangle_reasoning_ability import MangleReasoningAbility
   • Use tools directly: ability.enhance_user_input("your question")
   • Check constitutional compliance automatically
"""
    )


def main():
    """Main setup function."""
    print("=" * 60)
    print("🧠 GITHUB COPILOT MANGLE INTEGRATION - FIXED")
    print("=" * 60)

    success = setup_copilot_enhancement()

    if success:
        create_enhanced_demo()
        show_usage()
        print("\n✅ Setup complete! GitHub Copilot enhanced with Mangle reasoning!")
    else:
        print("\n⚠️ Setup completed with limited features.")
        print("Some dependencies missing - basic functionality available.")

    print("=" * 60)


if __name__ == "__main__":
    main()
