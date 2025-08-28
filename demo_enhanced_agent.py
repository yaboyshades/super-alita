#!/usr/bin/env python3
"""
Demonstration of Super Alita Agent's Enhanced DeepCode Capabilities

This script demonstrates how the agent now has access to advanced code analysis,
security scanning, and architectural insights through the integrated deepcode abilities.
"""

import asyncio
import json
import tempfile
from pathlib import Path

# Example code with various issues for demonstration
DEMO_CODE = '''
import os
import sys
import pickle
import subprocess

# Security vulnerabilities
def unsafe_function(user_input):
    """This function has security issues for demonstration"""
    # Critical: Code injection vulnerability
    result = eval(user_input)
    
    # Medium: Unsafe deserialization
    data = pickle.loads(user_input.encode())
    
    # Medium: Command injection potential
    subprocess.Popen(f"echo {user_input}", shell=True)
    
    return result

# Performance issues
def inefficient_function(items):
    """This function has performance issues"""
    result = []
    # Performance: Use enumerate instead of range(len())
    for i in range(len(items)):
        if items[i] > 0:
            result.append(items[i] * 2)
    return result

# Complexity issues
class ComplexClass:
    """This class demonstrates high complexity"""
    
    def complex_method(self, a, b, c, d, e, f):
        """High cyclomatic complexity method"""
        if a > 0:
            if b > 0:
                if c > 0:
                    if d > 0:
                        if e > 0:
                            if f > 0:
                                return a + b + c + d + e + f
                            else:
                                return a + b + c + d + e
                        else:
                            return a + b + c + d
                    else:
                        return a + b + c
                else:
                    return a + b
            else:
                return a
        else:
            return 0

# Better implementation examples
def safe_function(data):
    """Example of safer implementation"""
    try:
        # Use ast.literal_eval for safe evaluation
        import ast
        result = ast.literal_eval(data)
        return result
    except (ValueError, SyntaxError):
        return None

def efficient_function(items):
    """More efficient implementation"""
    return [item * 2 for item in items if item > 0]
'''

async def demonstrate_agent_capabilities():
    """Demonstrate the enhanced Super Alita agent's deepcode capabilities"""
    
    print("🚀 Super Alita Agent - Enhanced DeepCode Capabilities Demo")
    print("=" * 60)
    
    # Import the agent's ability registry
    from src.main import SimpleAbilityRegistry
    
    # Create the enhanced registry with deepcode tools
    registry = SimpleAbilityRegistry()
    
    # Create a demo file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(DEMO_CODE)
        demo_file = f.name
    
    try:
        print(f"📁 Created demo file: {demo_file}")
        print(f"📊 File size: {len(DEMO_CODE)} characters")
        
        # 1. Check file support
        print("\n1️⃣ CHECKING FILE SUPPORT")
        print("-" * 30)
        result = await registry.execute("check_file_support", {"file_path": demo_file})
        print(f"   File extension: {result['file_extension']}")
        print(f"   Supported: {'✅' if result['supported'] else '❌'}")
        print(f"   Can analyze: {'✅' if result['can_analyze'] else '❌'}")
        
        # 2. Comprehensive code analysis
        print("\n2️⃣ COMPREHENSIVE CODE ANALYSIS")
        print("-" * 30)
        result = await registry.execute("analyze_code_file", {
            "file_path": demo_file,
            "analysis_level": "deep"
        })
        
        print(f"   Analysis completed in {result['execution_time']:.3f}s")
        print(f"   Issues found: {result['issues_found']}")
        print(f"   Lines of code: {result['metrics']['lines_of_code']}")
        print(f"   Functions: {result['metrics']['functions']}")
        print(f"   Classes: {result['metrics']['classes']}")
        
        # Show severity breakdown
        severity = result['severity_breakdown']
        print(f"   Severity breakdown:")
        for level, count in severity.items():
            if count > 0:
                emoji = {"critical": "🔴", "error": "🟠", "warning": "🟡", "info": "🔵"}.get(level, "⚪")
                print(f"     {emoji} {level.title()}: {count}")
        
        # 3. Security pattern detection
        print("\n3️⃣ SECURITY PATTERN DETECTION")
        print("-" * 30)
        result = await registry.execute("detect_code_patterns", {
            "file_path": demo_file,
            "pattern_types": ["security"]
        })
        
        print(f"   Security patterns found: {result['patterns_found']}")
        for i, pattern in enumerate(result['patterns'][:3], 1):  # Show top 3
            print(f"   {i}. {pattern['severity'].upper()}: {pattern['message']}")
            print(f"      → Line {pattern['line_number']}")
            if pattern['suggestions']:
                print(f"      💡 Suggestion: {pattern['suggestions'][0]}")
        
        # 4. Performance analysis
        print("\n4️⃣ PERFORMANCE PATTERN ANALYSIS")
        print("-" * 30)
        result = await registry.execute("detect_code_patterns", {
            "file_path": demo_file,
            "pattern_types": ["performance"]
        })
        
        print(f"   Performance patterns found: {result['patterns_found']}")
        for i, pattern in enumerate(result['patterns'], 1):
            print(f"   {i}. {pattern['message']}")
            print(f"      → Line {pattern['line_number']}")
        
        # 5. Quality score assessment
        print("\n5️⃣ CODE QUALITY ASSESSMENT")
        print("-" * 30)
        result = await registry.execute("get_code_quality_score", {
            "target_path": demo_file,
            "include_metrics": True
        })
        
        quality_score = result['quality_score']
        print(f"   Overall Quality Score: {quality_score:.1f}/100")
        
        # Quality rating
        if quality_score >= 90:
            rating = "🟢 Excellent"
        elif quality_score >= 70:
            rating = "🟡 Good"
        elif quality_score >= 50:
            rating = "🟠 Fair"
        else:
            rating = "🔴 Needs Improvement"
        
        print(f"   Quality Rating: {rating}")
        print(f"   Total issues: {result['total_issues']}")
        
        # Show category breakdown
        if 'category_breakdown' in result:
            print("   Issue categories:")
            for category, count in result['category_breakdown'].items():
                print(f"     • {category}: {count}")
        
        # 6. Architectural insights
        print("\n6️⃣ ARCHITECTURAL INSIGHTS")
        print("-" * 30)
        result = await registry.execute("understand_code_structure", {
            "target_path": demo_file,
            "focus_area": "complexity"
        })
        
        structure = result['structure_analysis']
        print(f"   Complexity analysis:")
        print(f"     • Complexity issues: {structure.get('complexity_issues', 0)}")
        
        if 'complexity_details' in structure:
            for detail in structure['complexity_details'][:2]:  # Show top 2
                print(f"     • {detail}")
        
        # 7. Workspace context (analyze src directory)
        print("\n7️⃣ WORKSPACE INTELLIGENCE")
        print("-" * 30)
        result = await registry.execute("analyze_workspace_context", {
            "workspace_path": "./src",
            "max_files": 10,
            "include_quality_metrics": False
        })
        
        print(f"   Project type: {result.get('project_type', 'Unknown')}")
        print(f"   Main languages: {', '.join(result.get('main_languages', []))}")
        print(f"   Total files scanned: {result.get('total_files_found', 0)}")
        
        # 8. Agent recommendations
        print("\n8️⃣ AGENT RECOMMENDATIONS")
        print("-" * 30)
        print("   Based on the analysis, the agent recommends:")
        print("   🔐 Address security vulnerabilities immediately")
        print("   ⚡ Optimize performance-critical sections") 
        print("   🔧 Refactor complex methods for maintainability")
        print("   ✅ Add input validation and error handling")
        print("   📚 Consider using safer alternatives (ast.literal_eval)")
        
        print(f"\n🎉 DEMO COMPLETE - Agent successfully analyzed code!")
        print(f"📈 The Super Alita agent now has full access to deepcode capabilities!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        Path(demo_file).unlink(missing_ok=True)
    
    return True

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./src")
    sys.path.insert(0, "./reug")
    
    print("Setting up Super Alita Agent Demo...")
    success = asyncio.run(demonstrate_agent_capabilities())
    
    if success:
        print("\n" + "="*60)
        print("🚀 SUCCESS: Super Alita agent now has enhanced deepcode abilities!")
        print("📊 The agent can perform sophisticated code analysis and provide insights")
        print("🔍 Available for integration with chat, orchestration, and automation workflows")
        exit(0)
    else:
        exit(1)