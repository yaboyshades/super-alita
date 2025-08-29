#!/usr/bin/env python3
"""
Final validation script to demonstrate that the Jules agent import issue is resolved.

This script validates:
1. That the import system works correctly through pytest
2. That the Jules indexer is fully functional
3. That proper error guidance is provided for incorrect usage
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: list[str], cwd: Path = None) -> tuple[int, str, str]:
    """Run a command and return (exit_code, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=cwd or Path.cwd(),
            timeout=60
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"

def main():
    """Run validation tests."""
    repo_root = Path(__file__).parent
    print("🔍 Final Jules Agent Import Issue Resolution Validation")
    print("=" * 60)
    
    # Test 1: Verify pytest execution works
    print("\n📋 Test 1: Verify pytest execution works properly")
    exit_code, stdout, stderr = run_command([
        sys.executable, "-m", "pytest", 
        "tests/test_indexer.py::test_jules_indexer_import", 
        "-v"
    ], repo_root)
    
    if exit_code == 0:
        print("✅ PASS: pytest execution works")
    else:
        print(f"❌ FAIL: pytest execution failed (exit code {exit_code})")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return 1
    
    # Test 2: Verify direct script execution provides proper guidance
    print("\n📋 Test 2: Verify proper error guidance for direct execution")
    exit_code, stdout, stderr = run_command([
        sys.executable, "tests/test_indexer.py"
    ], repo_root)
    
    if exit_code == 1 and "Use: python -m pytest" in stdout:
        print("✅ PASS: Direct execution provides proper guidance")
    else:
        print(f"❌ FAIL: Direct execution should fail with guidance (got exit code {exit_code})")
        print(f"STDOUT: {stdout}")
        return 1
    
    # Test 3: Verify quick test utility works
    print("\n📋 Test 3: Verify quick test utility works")
    exit_code, stdout, stderr = run_command([
        sys.executable, "quick_test_jules.py"
    ], repo_root)
    
    if exit_code == 0 and "✅ All tests passed!" in stdout:
        print("✅ PASS: Quick test utility works")
    else:
        print(f"❌ FAIL: Quick test utility failed (exit code {exit_code})")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return 1
    
    # Test 4: Verify all indexer tests pass
    print("\n📋 Test 4: Verify all indexer tests pass")
    exit_code, stdout, stderr = run_command([
        sys.executable, "-m", "pytest", 
        "tests/test_indexer.py", 
        "-v"
    ], repo_root)
    
    if exit_code == 0 and "passed" in stdout:
        print("✅ PASS: All indexer tests pass")
    else:
        print(f"❌ FAIL: Some indexer tests failed (exit code {exit_code})")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return 1
    
    # Test 5: Verify manual import works with proper setup
    print("\n📋 Test 5: Verify manual import works with proper setup")
    exit_code, stdout, stderr = run_command([
        sys.executable, "-c", 
        "import sys; from pathlib import Path; sys.path.insert(0, str(Path('src'))); "
        "from agents.jules.indexer import RepositoryIndexer; "
        "indexer = RepositoryIndexer(Path('.')); "
        "result = indexer.index_repository(); "
        "print(f'SUCCESS: Found {len(result[\"files\"])} files')"
    ], repo_root)
    
    if exit_code == 0 and "SUCCESS: Found" in stdout:
        print("✅ PASS: Manual import works with proper setup")
    else:
        print(f"❌ FAIL: Manual import failed (exit code {exit_code})")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return 1
    
    print("\n🎉 ALL VALIDATION TESTS PASSED!")
    print("\n📝 Summary:")
    print("   ✅ Import system works correctly through pytest")
    print("   ✅ Proper error guidance is provided for incorrect usage")
    print("   ✅ Quick test utility provides standalone testing capability")
    print("   ✅ All test cases pass successfully")
    print("   ✅ Manual imports work with proper path setup")
    print("\n📚 The Jules agent import issue has been completely resolved!")
    print("   See docs/TESTING_GUIDE.md for detailed usage instructions.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())