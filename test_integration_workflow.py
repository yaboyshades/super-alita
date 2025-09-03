"""
Test the new code integration workflow
"""
import pytest
import sys
import os

# Add the source directory to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ecosystem.master_orchestrator import (
    EcosystemOrchestrator,
    WorkflowType,
    NoopNamingEnforcer,
    NoopPatternAnalyzer,
)
from src.ecosystem.eventbus import NoopEventBus
from src.ecosystem.telemetry import Telemetry


class FakeNamingEnforcer:
    """A fake naming enforcer that returns some violations."""
    async def validate_code_block(self, code_block: str, file_path: str) -> list[dict[str, str]]:
        # Return some mock naming violations for testing
        if "my_cache" in code_block:
            return [
                {
                    "name": "my_cache",
                    "suggestion": "MyCache", 
                    "rule": "PascalCase for classes"
                }
            ]
        return []


class FakePatternAnalyzer:
    """A fake pattern analyzer that returns some refactoring suggestions."""
    async def compare_against_codebase(self, code_block: str) -> dict[str, any]:
        suggestions = []
        if "def " in code_block and ":" not in code_block.split("def ")[1].split("(")[0]:
            suggestions.append({
                "prompt": "Add type hints to function parameters and return values to improve code clarity."
            })
        
        return {
            "compliance_score": 0.6 if suggestions else 0.9,
            "suggested_refactors": suggestions
        }


@pytest.mark.asyncio
async def test_integration_workflow_basic():
    """Test that the integration workflow runs and returns expected structure."""
    orchestrator = EcosystemOrchestrator()
    
    user_id = "test_user"
    action = "code_pasted"
    context = {
        "pasted_code": "def hello():\n    print('Hello world')", 
        "file_path": "src/test.py"
    }
    
    result = await orchestrator.handle_developer_action(user_id, action, context)
    
    # Verify the result structure
    assert result is not None
    assert result["workflow_type"] == "pasted_code_integration"
    assert "compliance_score" in result
    assert "issues_found" in result  
    assert "refactoring_prompts" in result
    assert "related_files" in result
    
    # Should have at least one prompt
    assert len(result["refactoring_prompts"]) >= 1


@pytest.mark.asyncio
async def test_integration_workflow_with_issues():
    """Test integration workflow with mock naming and pattern issues."""
    orchestrator = EcosystemOrchestrator(
        naming_enforcer=FakeNamingEnforcer(),
        pattern_analyzer=FakePatternAnalyzer()
    )
    
    user_id = "test_user"
    action = "code_pasted"
    context = {
        "pasted_code": "class my_cache:\n    def get_item(self, key):\n        return None", 
        "file_path": "src/cache.py"
    }
    
    result = await orchestrator.handle_developer_action(user_id, action, context)
    
    # Should find issues
    assert result["issues_found"] > 0
    assert len(result["refactoring_prompts"]) > 1
    
    # Should have naming violation prompt
    prompts = result["refactoring_prompts"]
    naming_prompt_found = any("my_cache" in prompt and "MyCache" in prompt for prompt in prompts)
    assert naming_prompt_found
    
    # Should have type hint prompt
    type_hint_prompt_found = any("type hints" in prompt.lower() for prompt in prompts)
    assert type_hint_prompt_found


@pytest.mark.asyncio
async def test_integration_workflow_missing_code():
    """Test that integration workflow handles missing pasted_code gracefully."""
    orchestrator = EcosystemOrchestrator()
    
    user_id = "test_user"
    action = "code_pasted"
    context = {"file_path": "src/test.py"}  # Missing pasted_code
    
    result = await orchestrator.handle_developer_action(user_id, action, context)
    
    # Should return error
    assert result["status"] == "error"
    assert "pasted_code not provided" in result["message"]


@pytest.mark.asyncio
async def test_workflow_type_enum():
    """Test that the new workflow type is properly defined."""
    assert hasattr(WorkflowType, 'PASTED_CODE_INTEGRATION')
    assert WorkflowType.PASTED_CODE_INTEGRATION.value == "pasted_code_integration"


@pytest.mark.asyncio
async def test_noop_implementations():
    """Test that the new no-op implementations work correctly."""
    naming_enforcer = NoopNamingEnforcer()
    pattern_analyzer = NoopPatternAnalyzer()
    
    violations = await naming_enforcer.validate_code_block("test code", "test.py")
    assert violations == []
    
    analysis = await pattern_analyzer.compare_against_codebase("test code")
    assert "compliance_score" in analysis
    assert "suggested_refactors" in analysis
    assert analysis["compliance_score"] == 0.7
    assert analysis["suggested_refactors"] == []


if __name__ == "__main__":
    import asyncio
    
    async def run_tests():
        print("Running integration workflow tests...")
        
        await test_integration_workflow_basic()
        print("✅ Basic workflow test passed")
        
        await test_integration_workflow_with_issues()
        print("✅ Workflow with issues test passed")
        
        await test_integration_workflow_missing_code()
        print("✅ Missing code handling test passed")
        
        await test_workflow_type_enum()
        print("✅ Workflow type enum test passed")
        
        await test_noop_implementations()
        print("✅ No-op implementations test passed")
        
        print("\n🎉 All integration workflow tests passed!")
    
    asyncio.run(run_tests())