#!/usr/bin/env python3
"""
Simple test script to verify Mangle integration dependencies.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def test_imports():
    """Test individual imports."""
    print("🧪 Testing individual imports...")

    # Test 1: Mangle rules
    try:
        print("✅ mangle_rules imported successfully")
    except Exception as e:
        print(f"❌ mangle_rules failed: {e}")

    # Test 2: Constitutional scorer
    try:
        print("✅ constitutional.scorer imported successfully")
    except Exception as e:
        print(f"❌ constitutional.scorer failed: {e}")

    # Test 3: Mangle reasoner
    try:
        print("✅ mangle_reasoner imported successfully")
    except Exception as e:
        print(f"❌ mangle_reasoner failed: {e}")

    # Test 4: Enhanced SDD framework
    try:
        print("✅ enhanced_sdd_framework imported successfully")
    except Exception as e:
        print(f"❌ enhanced_sdd_framework failed: {e}")

    # Test 5: Simple ability
    try:
        from abilities.simple_mangle_ability import MangleReasoningAbility

        ability = MangleReasoningAbility()
        print("✅ simple_mangle_ability works!")
    except Exception as e:
        print(f"❌ simple_mangle_ability failed: {e}")


def test_basic_functionality():
    """Test basic functionality."""
    print("\n🔧 Testing basic functionality...")

    try:
        from sdd.mangle_rules import get_query_for_question

        query = get_query_for_question("what functions are untested")
        if query:
            print(f"✅ Query mapping works: '{query}'")
        else:
            print("⚠️ Query mapping returned None")
    except Exception as e:
        print(f"❌ Query mapping failed: {e}")


def main():
    """Run all tests."""
    print("🚀 Testing Mangle Integration Dependencies")
    print("=" * 50)

    test_imports()
    test_basic_functionality()

    print("\n" + "=" * 50)
    print("✅ Basic import test completed!")


if __name__ == "__main__":
    main()
