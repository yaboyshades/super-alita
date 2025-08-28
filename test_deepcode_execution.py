#!/usr/bin/env python3
"""
Test actual deepcode tool execution with a real Python file
"""

import asyncio
import json
import tempfile
from pathlib import Path

async def test_deepcode_execution():
    """Test deepcode tool execution with real files"""
    
    # Import the registry
    from src.main import SimpleAbilityRegistry
    
    # Create a test Python file
    test_code = '''
import os
import sys

def vulnerable_function(user_input):
    # This is intentionally vulnerable for testing
    eval(user_input)  # Security issue
    result = ""
    for i in range(len(sys.argv)):  # Performance issue
        result += sys.argv[i]
    return result

class ComplexClass:
    def complex_method(self, a, b, c, d, e):
        if a > 0:
            if b > 0:
                if c > 0:
                    if d > 0:
                        if e > 0:  # High complexity
                            return a + b + c + d + e
                        else:
                            return 0
                    else:
                        return 0
                else:
                    return 0
            else:
                return 0
        else:
            return 0
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        test_file_path = f.name
    
    try:
        print(f"Created test file: {test_file_path}")
        
        # Test the registry
        registry = SimpleAbilityRegistry()
        
        # Test 1: Check file support
        print("\n🔍 Testing check_file_support...")
        result = await registry.execute("check_file_support", {"file_path": test_file_path})
        print(f"Result: {json.dumps(result, indent=2)}")
        
        # Test 2: Get supported extensions
        print("\n🔍 Testing get_supported_extensions...")
        result = await registry.execute("get_supported_extensions", {})
        print(f"Result: {json.dumps(result, indent=2)}")
        
        # Test 3: Analyze the test file
        print("\n🔍 Testing analyze_code_file...")
        result = await registry.execute("analyze_code_file", {
            "file_path": test_file_path,
            "analysis_level": "deep"
        })
        print(f"Found {result.get('issues_found', 0)} issues")
        if "issues" in result:
            for issue in result["issues"][:5]:  # Show first 5 issues
                print(f"  - {issue.get('severity', 'UNKNOWN')}: {issue.get('message', 'No message')}")
        
        # Test 4: Detect specific patterns
        print("\n🔍 Testing detect_code_patterns...")
        result = await registry.execute("detect_code_patterns", {
            "file_path": test_file_path,
            "pattern_types": ["security", "performance"]
        })
        print(f"Found {result.get('patterns_found', 0)} patterns")
        if "patterns" in result:
            for pattern in result["patterns"]:
                print(f"  - {pattern.get('category', 'UNKNOWN')}: {pattern.get('message', 'No message')}")
        
        # Test 5: Get quality score
        print("\n🔍 Testing get_code_quality_score...")
        result = await registry.execute("get_code_quality_score", {
            "target_path": test_file_path,
            "include_metrics": True
        })
        print(f"Quality score: {result.get('quality_score', 'N/A')}")
        
        print("\n🎉 All deepcode tool execution tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        Path(test_file_path).unlink(missing_ok=True)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./src")
    sys.path.insert(0, "./reug")
    
    success = asyncio.run(test_deepcode_execution())
    exit(0 if success else 1)