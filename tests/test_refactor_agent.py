#!/usr/bin/env python3
"""Test script to validate the refactoring agent functionality."""

import sys
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent / "tools"))


def test_basic_import():
    """Test that we can import the refactoring agent components."""
    try:
        from refactor_hotspots import (
            AutonomousRefactoringAgent,
            CodeAnalyzer,
            MangleReasoningAbility,
            auto_code_reason,
            mangle_is_available,
        )

        print("✅ Import successful - all components loaded")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_mangle_availability():
    """Test Mangle availability check."""
    try:
        from refactor_hotspots import mangle_is_available

        available = mangle_is_available()
        print(f"✅ Mangle availability check: {available}")
        return True
    except Exception as e:
        print(f"❌ Mangle check failed: {e}")
        return False


def test_code_analyzer():
    """Test basic CodeAnalyzer functionality."""
    try:
        from refactor_hotspots import CodeAnalyzer

        analyzer = CodeAnalyzer()

        # Test on tools directory
        tools_path = Path(__file__).parent / "tools"
        if tools_path.exists():
            ops = analyzer.scan_directory(tools_path)
            print(f"✅ CodeAnalyzer scan: found {len(ops)} opportunities")

            # Show first few opportunities
            for i, op in enumerate(ops[:3]):
                print(
                    f"   {i+1}. {op.issue_type} in {Path(op.file_path).name}: {op.description[:60]}..."
                )

            return True
        else:
            print("⚠️  Tools directory not found - skipping scan test")
            return True
    except Exception as e:
        print(f"❌ CodeAnalyzer test failed: {e}")
        return False


def test_auto_code_reason():
    """Test the auto_code_reason helper function."""
    try:
        from refactor_hotspots import auto_code_reason

        result = auto_code_reason(
            "find functions with high complexity", "tools"
        )
        print(f"✅ auto_code_reason test: used_mangle={result['used_mangle']}")
        print(f"   Results: {len(result['results'])} semantic matches")
        print(f"   Hints: {len(result['hints'])} analyzer hints")

        if result["hints"]:
            print(f"   First hint: {result['hints'][0]}")

        return True
    except Exception as e:
        print(f"❌ auto_code_reason test failed: {e}")
        return False


def test_agent_creation():
    """Test creating an AutonomousRefactoringAgent."""
    try:
        from refactor_hotspots import AutonomousRefactoringAgent

        # Test on tools directory
        tools_path = Path(__file__).parent / "tools"
        if tools_path.exists():
            agent = AutonomousRefactoringAgent(tools_path)
            print("✅ AutonomousRefactoringAgent creation successful")

            # Test project analysis
            plan = agent.analyze_project()
            print(
                f"   Analysis: {len(plan.opportunities)} opportunities found"
            )
            print(
                f"   Execution order: {len(plan.execution_order)} files to process"
            )

            return True
        else:
            print("⚠️  Tools directory not found - creating mock agent")
            agent = AutonomousRefactoringAgent(Path("."))
            print("✅ AutonomousRefactoringAgent creation successful (mock)")
            return True
    except Exception as e:
        print(f"❌ AutonomousRefactoringAgent test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Autonomous Refactoring Agent Kit...")
    print("=" * 50)

    tests = [
        test_basic_import,
        test_mangle_availability,
        test_code_analyzer,
        test_auto_code_reason,
        test_agent_creation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        print(f"\n🔍 {test.__doc__}")
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test error: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print(
            "🎉 All tests passed! Autonomous Refactoring Agent Kit is functional."
        )
        return 0
    else:
        print(
            "⚠️  Some tests failed. Check dependency installation and configuration."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
