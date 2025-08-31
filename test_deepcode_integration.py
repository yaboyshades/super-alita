"""Test the DeepCode integration functionality"""

import asyncio
import tempfile
from pathlib import Path

from src.deepcode import analyze_current_file, is_supported_file


async def test_deepcode_integration():
    """Test basic DeepCode functionality"""

    # Create a test Python file with some issues
    test_code = """
def bad_function():
    # This function has some issues
    result = eval("1 + 1")  # Security issue
    for i in range(len([1, 2, 3])):  # Performance issue
        print(i)

    # Complex function
    if True:
        if True:
            if True:
                if True:
                    if True:
                        pass

    return result

class VeryLongClassNameThatDoesWayTooManyThings:
    def method1(self): pass
    def method2(self): pass
    def method3(self): pass
    def method4(self): pass
    def method5(self): pass
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_code)
        test_file_path = f.name

    try:
        # Test file type detection
        assert is_supported_file(test_file_path), "Python file should be supported"
        assert not is_supported_file("test.txt"), "Text file should not be supported"

        # Test file analysis
        result = await analyze_current_file(test_file_path)

        print("DeepCode Analysis Results:")
        print(f"Enabled: {result.get('enabled', False)}")
        print(f"Issues found: {result.get('issues_count', 0)}")
        print(f"Quality score: {result.get('quality_score', 0):.1f}")
        print(f"Execution time: {result.get('execution_time', 0):.3f}s")

        if "issues" in result:
            print("\nTop Issues:")
            for issue in result["issues"][:3]:
                print(
                    f"  - {issue['severity']}: {issue['message']} (line {issue['line_number']})"
                )

        if "metrics" in result:
            print("\nMetrics:")
            metrics = result["metrics"]
            print(f"  Lines of code: {metrics.get('lines_of_code', 0)}")
            print(f"  Functions: {metrics.get('functions', 0)}")
            print(f"  Classes: {metrics.get('classes', 0)}")

        # Verify we found some issues
        assert result.get("issues_count", 0) > 0, "Should have found some issues"
        assert result.get("enabled", False), "Should be enabled"

        print("\n✅ DeepCode integration test passed!")
        return True

    except Exception as e:
        print(f"❌ DeepCode integration test failed: {e}")
        return False

    finally:
        # Cleanup
        Path(test_file_path).unlink(missing_ok=True)


if __name__ == "__main__":
    success = asyncio.run(test_deepcode_integration())
    exit(0 if success else 1)
