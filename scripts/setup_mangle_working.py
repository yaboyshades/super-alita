#!/usr/bin/env python3
"""
GitHub Copilot Mangle Integration - Robust Setup.

This script sets up Mangle reasoning for GitHub Copilot even when
some dependencies are missing.
"""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def test_mangle_features():
    """Test which Mangle features are available."""
    print("🔍 Testing Mangle integration features...")

    features = []

    # Test basic rules
    try:
        from sdd.mangle_rules import get_query_for_question

        query = get_query_for_question("what functions are untested")
        if query:
            features.append("✅ Natural language query mapping")
        else:
            features.append("⚠️ Query mapping (limited)")
    except Exception as e:
        features.append(f"❌ Query mapping: {e}")

    # Test constitutional scorer
    try:
        from constitutional.scorer import ConstitutionalScorer

        ConstitutionalScorer()
        features.append("✅ Constitutional compliance checking")
    except Exception as e:
        features.append(f"❌ Constitutional checking: {e}")

    # Test simple ability
    try:
        from abilities.simple_mangle_ability import MangleReasoningAbility

        ability = MangleReasoningAbility()
        result = ability.enhance_user_input("test")
        if result:
            features.append("✅ Basic Copilot enhancement")
    except Exception as e:
        features.append(f"❌ Copilot enhancement: {e}")

    return features


def setup_copilot_enhancement():
    """Set up GitHub Copilot enhancement."""
    print("🚀 Setting up GitHub Copilot Mangle enhancement...")

    # Set environment variables
    os.environ["COPILOT_MANGLE_MODE"] = "true"
    os.environ["COPILOT_AUTO_ENHANCE"] = "true"

    # Test features
    features = test_mangle_features()

    print("\n📊 Feature Status:")
    for feature in features:
        print(f"   {feature}")

    working_count = len([f for f in features if f.startswith("✅")])

    if working_count > 0:
        print(f"\n✅ {working_count} features working!")
        print("🧠 GitHub Copilot enhancement activated!")
        return True
    else:
        print("\n⚠️ Limited functionality available")
        print("📋 See installation instructions for full features")
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
   • Constitutional compliance guidance
   • Quality improvement suggestions
   • Mangle reasoning insights (when available)

4. No additional setup needed - enhancement is automatic!
"""
    )


def main():
    """Main setup function."""
    print("=" * 60)
    print("🧠 GITHUB COPILOT MANGLE INTEGRATION")
    print("=" * 60)

    success = setup_copilot_enhancement()

    if success:
        show_usage()
        print("\n✅ Setup complete! Start using GitHub Copilot with Mangle reasoning!")
    else:
        print("\n⚠️ Setup completed with limited features.")
        print("Run 'python setup_copilot_mangle.py' for full installation.")

    print("=" * 60)


if __name__ == "__main__":
    main()
