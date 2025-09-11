#!/usr/bin/env python3
"""
Jest for Python Demo Script

This script demonstrates how Super Alita uses pytest and related packages to provide
Jest-equivalent functionality. Run this to see various Jest patterns in action.

Usage:
    python demo_jest_for_python.py
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: str, description: str, timeout: int = 30) -> bool:
    """Run a command and show its output."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd.split(),
            capture_output=False,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent,
        )

        success = result.returncode == 0
        print(
            f"\n{'✅' if success else '❌'} {description} - {'Success' if success else 'Failed'}"
        )
        return success

    except subprocess.TimeoutExpired:
        print(f"\n⏰ {description} - Timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"\n❌ {description} - Error: {e}")
        return False


def main():
    """Main demo function."""
    print("🚀 Jest for Python Demonstration")
    print("=" * 60)
    print(
        "This demo shows how pytest + related packages provide Jest-equivalent functionality"
    )
    print("in the Super Alita project.")

    # Check if we're in the right directory
    if not Path("tests/test_jest_like_patterns.py").exists():
        print("❌ Error: Please run this script from the super-alita repository root")
        sys.exit(1)

    demos = [
        {
            "cmd": "pytest tests/test_jest_like_patterns.py::test_add_parametrized -v",
            "desc": "Parametrized Tests (Jest's test.each equivalent)",
            "timeout": 30,
        },
        {
            "cmd": "pytest tests/test_jest_like_patterns.py::test_calculator_with_mock tests/test_jest_like_patterns.py::test_calculator_spy_on_existing_method -v",
            "desc": "Mocking & Spying (Jest's jest.fn/jest.mock equivalent)",
            "timeout": 30,
        },
        {
            "cmd": "pytest tests/test_jest_like_patterns.py::test_user_payload_snapshot -v",
            "desc": "Snapshot Testing (Jest snapshots equivalent)",
            "timeout": 30,
        },
        {
            "cmd": "pytest tests/test_jest_like_patterns.py::TestCalculatorFeatures -v",
            "desc": "Test Grouping (Jest's describe equivalent)",
            "timeout": 30,
        },
        {
            "cmd": "pytest tests/test_jest_like_patterns.py::test_async_service_mock -v",
            "desc": "Async Testing (Jest async/await equivalent)",
            "timeout": 30,
        },
        {
            "cmd": "pytest tests/test_jest_like_patterns.py -k 'fixture' -v",
            "desc": "Fixtures (Jest's beforeEach/afterEach equivalent)",
            "timeout": 30,
        },
        {
            "cmd": "pytest tests/test_jest_like_patterns.py --cov=tests/test_jest_like_patterns --cov-report=term-missing",
            "desc": "Coverage Reporting (Jest's --coverage equivalent)",
            "timeout": 45,
        },
        {
            "cmd": "pytest tests/test_jest_like_patterns.py -n 2",
            "desc": "Parallel Execution (Jest's built-in parallelism equivalent)",
            "timeout": 45,
        },
        {
            "cmd": "pytest tests/test_jest_like_patterns.py -m unit",
            "desc": "Test Filtering by Markers (Jest's pattern matching equivalent)",
            "timeout": 30,
        },
    ]

    print(f"\nRunning {len(demos)} demonstrations...")

    results = []
    for i, demo in enumerate(demos, 1):
        print(f"\n📍 Demo {i}/{len(demos)}")
        success = run_command(demo["cmd"], demo["desc"], demo["timeout"])
        results.append((demo["desc"], success))

        # Small delay between demos for readability
        if i < len(demos):
            time.sleep(1)

    # Summary
    print(f"\n\n{'='*60}")
    print("📊 DEMO SUMMARY")
    print(f"{'='*60}")

    successful = sum(1 for _, success in results if success)
    total = len(results)

    print(f"Total Demos: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success Rate: {successful/total*100:.1f}%")

    print("\nDemo Results:")
    for desc, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status} {desc}")

    print(f"\n{'='*60}")
    print("🎯 Jest-to-Python Mapping Summary")
    print(f"{'='*60}")

    mappings = [
        ("Jest Test Runner", "pytest"),
        ("Jest expect(...)", "assert statement with rich diff"),
        ("jest.fn() / jest.mock()", "unittest.mock.Mock + pytest-mock"),
        ("jest.spyOn()", "patch with wraps parameter"),
        ("Jest snapshots", "syrupy package"),
        ("jest --watch", "pytest-watch (ptw command)"),
        ("jest --coverage", "pytest --cov"),
        ("Jest parallelism", "pytest -n auto (pytest-xdist)"),
        ("test.each([...])", "@pytest.mark.parametrize"),
        ("beforeEach/afterEach", "pytest fixtures"),
        ("describe blocks", "test classes"),
        ("Jest async testing", "@pytest.mark.asyncio"),
    ]

    for jest_feature, python_equivalent in mappings:
        print(f"  {jest_feature:<25} → {python_equivalent}")

    print(f"\n{'='*60}")
    print("📚 Next Steps")
    print(f"{'='*60}")
    print("1. Check out the comprehensive examples in tests/test_jest_like_patterns.py")
    print("2. Read the detailed guide in docs/jest_for_python_guide.md")
    print("3. Try running individual commands:")
    print("   - pytest --help")
    print("   - ptw --help  # for watch mode")
    print("   - pytest --snapshot-update  # for snapshot testing")
    print("4. Install packages: pip install -r requirements-test.txt")

    if successful == total:
        print(
            "\n🎉 All demos passed! Jest for Python is working perfectly in Super Alita."
        )
    else:
        print(
            f"\n⚠️  {total - successful} demo(s) failed. Check the output above for details."
        )

    return successful == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
