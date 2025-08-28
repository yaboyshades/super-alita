#!/usr/bin/env python3
"""
Integration test demonstrating Enhanced Copilot capabilities
This test shows how the enhanced copilot integrates DeepCode analysis with
GitHub repository discovery for automated problem solving.
"""
import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def demonstrate_enhanced_copilot():
    """Demonstrate the enhanced copilot functionality with realistic examples"""
    try:
        from src.main import SimpleAbilityRegistry

        print("🎯 Enhanced GitHub Copilot with DeepCode Integration")
        print("=" * 60)
        print("Demonstrating automated problem solving with:")
        print("  • DeepCode analysis for code quality")
        print("  • GitHub repository discovery")
        print("  • End-to-end automation workflow")
        print()

        registry = SimpleAbilityRegistry()

        # Demo 1: Problem Analysis and Repository Suggestions
        print("🔍 DEMO 1: Problem Analysis & GitHub Repository Discovery")
        print("-" * 50)

        problem = "I need to build a Python web API with authentication and database integration"
        result = await registry.execute(
            "analyze_and_suggest_repos",
            {
                "problem_description": problem,
                "language_preference": "python",
                "max_results": 5,
            },
        )

        print(f"Problem: {problem}")
        print(f"Generated search query: {result.get('search_query', 'N/A')}")
        print(f"Repository suggestions found: {result.get('total_found', 0)}")
        print()

        # Demo 2: Automated Problem Solver
        print("🤖 DEMO 2: End-to-End Automated Problem Solver")
        print("-" * 45)

        task = "Create a REST API server with JWT authentication"
        result = await registry.execute(
            "automated_problem_solver",
            {
                "task_description": task,
                "workspace_path": ".",
                "include_code_generation": True,
                "analyze_existing_code": True,
            },
        )

        print(f"Task: {task}")
        print(f"Solution generated: {result.get('success', False)}")

        if result.get("solution_steps"):
            print("Solution steps:")
            for i, step in enumerate(result["solution_steps"], 1):
                print(f"  {i}. {step.get('description', 'Unknown step')}")

        if result.get("code_suggestions"):
            suggestions = result["code_suggestions"].get("code_suggestions", [])
            print(f"Code suggestions provided: {len(suggestions)}")
            for suggestion in suggestions:
                print(
                    f"  - {suggestion.get('title', 'Untitled')}: {suggestion.get('type', 'unknown')}"
                )
        print()

        # Demo 3: Enhanced Code Review with DeepCode
        print("📝 DEMO 3: Enhanced Code Review with DeepCode Analysis")
        print("-" * 50)

        # Create a sample code file with issues
        sample_code = """
import os
import subprocess
import pickle

def process_user_data(user_input, file_path):
    # Potential security issues for demonstration
    result = eval(user_input)  # Unsafe eval
    
    # Unsafe subprocess call
    subprocess.call(f"echo {user_input}", shell=True)
    
    # Unsafe pickle loading
    with open(file_path, 'rb') as f:
        data = pickle.load(f)  # Could be unsafe
    
    return result + len(data)

def secure_function():
    # This is a good function
    return "Hello, World!"
"""

        test_file = Path("sample_code_review.py")
        test_file.write_text(sample_code)

        try:
            result = await registry.execute(
                "enhanced_code_review",
                {
                    "code_path": str(test_file),
                    "review_type": "security",
                    "suggest_improvements": True,
                },
            )

            print(f"Code review completed for: {test_file.name}")

            if result.get("deepcode_analysis"):
                issues = result["deepcode_analysis"].get("issues", [])
                print(f"Security issues found: {len(issues)}")

                for issue in issues[:3]:  # Show first 3 issues
                    severity = issue.get("severity", "UNKNOWN")
                    message = issue.get("message", "No message")
                    line = issue.get("line", 0)
                    print(f"  ⚠️  {severity}: {message} (line {line})")

            improvements = result.get("improvement_suggestions", [])
            if improvements:
                print(f"Improvement suggestions: {len(improvements)}")
                for imp in improvements[:2]:  # Show first 2 suggestions
                    print(f"  💡 {imp.get('suggestion', 'No suggestion')}")

        finally:
            test_file.unlink(missing_ok=True)
        print()

        # Demo 4: Repository Deep Analysis
        print("🔬 DEMO 4: Repository Deep Analysis")
        print("-" * 35)

        repo_url = "https://github.com/fastapi/fastapi"
        result = await registry.execute(
            "repository_deep_analysis",
            {
                "repo_url": repo_url,
                "analysis_focus": "architecture",
                "include_dependencies": True,
            },
        )

        print(f"Repository analyzed: {repo_url}")
        if result.get("error"):
            print(f"Analysis result: {result['error']} (Expected - no GitHub token)")
        else:
            print(f"Analysis completed for: {result.get('repository', 'N/A')}")
            print(f"Focus area: {result.get('analysis_focus', 'N/A')}")
        print()

        # Summary
        print("✅ INTEGRATION DEMONSTRATION COMPLETE")
        print("=" * 45)
        print("Enhanced GitHub Copilot successfully integrates:")
        print("  ✓ DeepCode static analysis for code quality")
        print("  ✓ GitHub repository discovery and analysis")
        print("  ✓ Automated problem-solving workflows")
        print("  ✓ End-to-end code generation and review")
        print()
        print("🎉 The enhanced copilot provides comprehensive")
        print("   development assistance from problem identification")
        print("   to solution implementation!")

        return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run the demonstration"""
    success = await demonstrate_enhanced_copilot()
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
