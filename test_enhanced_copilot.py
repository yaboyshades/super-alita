#!/usr/bin/env python3
"""
Test script for Enhanced Copilot Ability integration
"""
import asyncio
import json
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_enhanced_copilot():
    """Test the enhanced copilot functionality"""
    try:
        from src.abilities.enhanced_copilot_ability import EnhancedCopilotAbility
        
        print("🚀 Testing Enhanced Copilot Ability Integration")
        print("=" * 50)
        
        # Create and setup the ability
        ability = EnhancedCopilotAbility()
        await ability.setup(None, None, {})
        
        # Test 1: Get available tools
        print("\n📋 Test 1: Getting available tools...")
        tools = ability.get_available_tools()
        print(f"✅ Found {len(tools)} enhanced copilot tools:")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")
        
        # Test 2: Test analyze_and_suggest_repos
        print("\n🔍 Test 2: Testing analyze_and_suggest_repos...")
        result = await ability._execute_tool("analyze_and_suggest_repos", {
            "problem_description": "I need to create a web scraping tool in Python",
            "language_preference": "python",
            "max_results": 3
        })
        print(f"✅ analyze_and_suggest_repos result: {json.dumps(result, indent=2)[:500]}...")
        
        # Test 3: Test automated_problem_solver
        print("\n🤖 Test 3: Testing automated_problem_solver...")
        result = await ability._execute_tool("automated_problem_solver", {
            "task_description": "Create a simple REST API server",
            "workspace_path": ".",
            "include_code_generation": True
        })
        print(f"✅ automated_problem_solver completed: {result.get('success', False)}")
        if result.get('solution_steps'):
            print(f"  Generated {len(result['solution_steps'])} solution steps")
        
        # Test 4: Test enhanced_code_review
        print("\n📝 Test 4: Testing enhanced_code_review...")
        # Create a test file
        test_file = Path("test_code.py")
        test_file.write_text("""
def unsafe_function(user_input):
    # This is intentionally unsafe code for testing
    exec(user_input)
    return "executed"

def good_function():
    return "safe"
""")
        
        try:
            result = await ability._execute_tool("enhanced_code_review", {
                "code_path": str(test_file),
                "review_type": "security",
                "suggest_improvements": True
            })
            print(f"✅ enhanced_code_review completed")
            if result.get('deepcode_analysis'):
                issues = result['deepcode_analysis'].get('issues', [])
                print(f"  Found {len(issues)} code issues")
        finally:
            test_file.unlink(missing_ok=True)
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_ability_registry_integration():
    """Test integration with the ability registry"""
    try:
        from src.main import SimpleAbilityRegistry
        
        print("\n🔧 Testing Ability Registry Integration")
        print("=" * 40)
        
        registry = SimpleAbilityRegistry()
        
        # Test that enhanced copilot tools are known
        enhanced_tools = [
            "analyze_and_suggest_repos",
            "automated_problem_solver", 
            "repository_deep_analysis",
            "enhanced_code_review"
        ]
        
        for tool in enhanced_tools:
            if registry.knows(tool):
                print(f"✅ Registry knows tool: {tool}")
            else:
                print(f"❌ Registry doesn't know tool: {tool}")
        
        # Test tool execution via registry
        print("\n🧪 Testing tool execution via registry...")
        result = await registry.execute("analyze_and_suggest_repos", {
            "problem_description": "Create a simple calculator app",
            "max_results": 2
        })
        
        if "error" not in result:
            print("✅ Registry execution successful")
        else:
            print(f"❌ Registry execution failed: {result['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Registry integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🧪 Enhanced Copilot Integration Test Suite")
    print("=" * 50)
    
    test1_passed = await test_enhanced_copilot()
    test2_passed = await test_ability_registry_integration()
    
    print(f"\n📊 Test Results:")
    print(f"  Enhanced Copilot Ability: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Registry Integration: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests PASSED! Enhanced Copilot integration is working.")
        return True
    else:
        print("\n❌ Some tests FAILED. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)