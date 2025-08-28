#!/usr/bin/env python3
"""
Test workspace analysis functionality
"""

import asyncio
import json

async def test_workspace_analysis():
    """Test workspace analysis on the actual project"""
    
    from src.main import SimpleAbilityRegistry
    
    registry = SimpleAbilityRegistry()
    
    # Test workspace analysis on the src directory
    print("🔍 Testing analyze_workspace_context on src/ directory...")
    result = await registry.execute("analyze_workspace_context", {
        "workspace_path": "./src",
        "max_files": 5,
        "include_quality_metrics": True
    })
    
    print(f"Workspace analysis result:")
    print(f"  Project type: {result.get('project_type', 'N/A')}")
    print(f"  Main languages: {result.get('main_languages', [])}")
    print(f"  Total files found: {result.get('total_files_found', 0)}")
    print(f"  Supported files: {len(result.get('supported_files', []))}")
    
    # Test directory analysis
    print("\n🔍 Testing analyze_code_directory on abilities/ directory...")
    result = await registry.execute("analyze_code_directory", {
        "directory_path": "./src/abilities",
        "analysis_level": "semantic",
        "generate_report": True
    })
    
    if "report" in result:
        report = result["report"]
        print(f"Directory analysis report:")
        print(f"  Files analyzed: {report['summary']['files_analyzed']}")
        print(f"  Total issues: {report['summary']['total_issues']}")
        print(f"  Quality score: {report['summary']['quality_score']}")
        print(f"  Severity breakdown: {report['summary']['severity_breakdown']}")
    
    # Test code structure understanding
    print("\n🔍 Testing understand_code_structure on main.py...")
    result = await registry.execute("understand_code_structure", {
        "target_path": "./src/main.py",
        "focus_area": "complexity"
    })
    
    print(f"Code structure analysis:")
    structure = result.get("structure_analysis", {})
    print(f"  Lines of code: {structure.get('lines_of_code', 0)}")
    print(f"  Functions: {structure.get('functions', 0)}")
    print(f"  Classes: {structure.get('classes', 0)}")
    print(f"  Complexity issues: {structure.get('complexity_issues', 0)}")
    
    print("\n🎉 Workspace analysis tests completed!")
    return True

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./src")
    sys.path.insert(0, "./reug")
    
    success = asyncio.run(test_workspace_analysis())
    exit(0 if success else 1)