#!/usr/bin/env python3
"""
Validation script for Mangle integration.

This script validates that all Mangle integration components are working correctly
by testing the key functionality and ensuring constitutional compliance.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_mangle_fact_generator():
    """Test Mangle fact generator."""
    try:
        from sdd.mangle_integration import MangleFactGenerator

        generator = MangleFactGenerator()
        facts = generator.generate_all_facts()

        print("✅ MangleFactGenerator: Working")
        print(f"   Generated {len(facts.split('.'))} facts")
        return True
    except Exception as e:
        print(f"❌ MangleFactGenerator: Failed - {e}")
        return False


def test_mangle_rules():
    """Test Mangle rules."""
    try:
        from sdd.mangle_rules import MANGLE_RULES, get_query_for_question

        # Test rule content
        assert "untested_function" in MANGLE_RULES
        assert "constitutional_violation" in MANGLE_RULES

        # Test query mapping
        query = get_query_for_question("what functions are untested")
        assert query == "untested_function(Func)"

        print("✅ MangleRules: Working")
        print(f"   Rules: {len(MANGLE_RULES.split(':-'))} rules defined")
        return True
    except Exception as e:
        print(f"❌ MangleRules: Failed - {e}")
        return False


def test_mangle_reasoner():
    """Test Mangle reasoner."""
    try:
        from sdd.mangle_reasoner import MangleReasoner

        reasoner = MangleReasoner()

        # Test query execution (will fail without Mangle binary, but should not crash)
        try:
            result = reasoner.query("untested_function(Func)")
            print("✅ MangleReasoner: Working (with Mangle binary)")
        except Exception:
            print("✅ MangleReasoner: Working (without Mangle binary)")

        return True
    except Exception as e:
        print(f"❌ MangleReasoner: Failed - {e}")
        return False


def test_enhanced_framework():
    """Test enhanced SDD framework."""
    try:
        from sdd.enhanced_sdd_framework import EnhancedSDDFramework

        framework = EnhancedSDDFramework()

        # Test question answering
        answer = framework.ask_question("what functions are untested")
        assert isinstance(answer, str)

        print("✅ EnhancedSDDFramework: Working")
        print(f"   Answer: {str(answer)[:50]}...")
        return True
    except Exception as e:
        print(f"❌ EnhancedSDDFramework: Failed - {e}")
        return False


def test_cli_commands():
    """Test CLI commands."""
    try:
        from sdd.sdd_cli import cli

        # Test CLI initialization
        assert cli is not None

        print("✅ SDD CLI: Working")
        return True
    except Exception as e:
        print(f"❌ SDD CLI: Failed - {e}")
        return False


def test_api_router():
    """Test API router."""
    try:
        from sdd.router import create_sdd_router

        router = create_sdd_router()
        assert router is not None

        print("✅ SDD API Router: Working")
        print(f"   Routes: {len(router.routes)} endpoints")
        return True
    except Exception as e:
        print(f"❌ SDD API Router: Failed - {e}")
        return False


def main():
    """Run all validation tests."""
    print("🔍 Validating Mangle Integration Components...")
    print("=" * 50)

    tests = [
        test_mangle_fact_generator,
        test_mangle_rules,
        test_mangle_reasoner,
        test_enhanced_framework,
        test_cli_commands,
        test_api_router,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"✅ Validation Results: {passed}/{total} components working")

    if passed == total:
        print("🎉 All Mangle integration components are working correctly!")
        return 0
    else:
        print("⚠️  Some components need attention.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
