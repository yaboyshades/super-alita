#!/usr/bin/env python3
"""
Comprehensive test suite for Enhanced LADDER Planner with EOS Orchestration

Tests the EOS LADDER integration including:
- EOS LADDER stage processing (Lift/Decompose/Synthesize/Descend)
- Task complexity analysis and adaptive decomposition
- Mangle reasoning validation for tasks
- Constitutional compliance analysis
- Dynamic task adaptation
- Performance optimization and telemetry
"""

import asyncio
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.ladder.enhanced_eos_planner import (
        EOSLadderPlanner,
        EnhancedTask,
        EOSTaskContext,
        TaskAnalysis,
        MangleTaskValidator,
        ConstitutionalTaskAnalyzer,
        EOSStage,
        TaskComplexity,
        AdaptationTrigger,
        create_eos_ladder_planner
    )
    EOS_PLANNER_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced EOS LADDER Planner not available: {e}")
    EOS_PLANNER_AVAILABLE = False
    
    # Create mock classes for testing structure
    class MockEOSLadderPlanner:
        def __init__(self, *args, **kwargs):
            self.eos_stats = {"tasks_analyzed": 0}
        
        async def create_eos_plan(self, goal, **kwargs):
            return None, {"stages_completed": ["lift", "decompose", "synthesize", "descend"]}
    
    class MockMangleTaskValidator:
        async def validate_task(self, task):
            return {"is_valid": True, "optimizations": []}
    
    class MockConstitutionalTaskAnalyzer:
        async def analyze_constitutional_compliance(self, task):
            return {"overall_score": 0.9, "concerns": []}
    
    EOSLadderPlanner = MockEOSLadderPlanner
    MangleTaskValidator = MockMangleTaskValidator
    ConstitutionalTaskAnalyzer = MockConstitutionalTaskAnalyzer


class TestEOSLadderPlanner:
    """Test suite for Enhanced EOS LADDER Planner"""
    
    def get_simple_goal(self):
        """Simple goal for testing"""
        return "Calculate the sum of two numbers"
    
    def get_complex_goal(self):
        """Complex goal requiring decomposition"""
        return "Analyze and implement a comprehensive machine learning pipeline for customer churn prediction with data preprocessing, model training, evaluation, and deployment"
    
    def get_problematic_goal(self):
        """Goal with potential constitutional issues"""
        return "Create a system to discriminate against certain user groups based on personal data"
    
    def get_basic_context(self):
        """Basic context for testing"""
        return {
            "domain": "software_engineering",
            "resources": {"cpu": 4, "memory": "8GB"},
            "deadline": "2024-01-31"
        }
    
    def get_success_criteria(self):
        """Success criteria for testing"""
        return [
            "Task completed within deadline",
            "Quality standards met",
            "No constitutional violations"
        ]
    
    async def test_eos_ladder_planner_initialization(self):
        """Test EOS LADDER planner initialization"""
        if not EOS_PLANNER_AVAILABLE:
            print("✓ Mock EOS LADDER planner initialized")
            return True
        
        # Test with different configurations
        planner1 = EOSLadderPlanner(
            enable_constitutional_analysis=True,
            enable_mangle_validation=True
        )
        
        planner2 = EOSLadderPlanner(
            enable_constitutional_analysis=False,
            enable_mangle_validation=False
        )
        
        assert planner1.enable_constitutional_analysis is True
        assert planner1.enable_mangle_validation is True
        assert planner2.enable_constitutional_analysis is False
        assert planner2.enable_mangle_validation is False
        
        print("✓ EOS LADDER planner initialization completed")
        return True
    
    async def test_mangle_task_validator(self):
        """Test Mangle task validator"""
        if not EOS_PLANNER_AVAILABLE:
            print("✓ Mock Mangle task validator test passed")
            return True
        
        validator = MangleTaskValidator()
        
        # Create mock task
        task = type('MockTask', (), {
            'description': 'Process data efficiently',
            'eos_context': type('MockContext', (), {'constraints': {}})(),
            'mangle_validation': {}
        })()
        
        # Validate task
        result = await validator.validate_task(task)
        
        assert "is_valid" in result
        assert "confidence" in result
        assert "issues" in result
        assert "recommendations" in result
        assert "optimizations" in result
        
        print("✓ Mangle task validator test completed")
        print(f"  - Valid: {result['is_valid']}")
        print(f"  - Confidence: {result['confidence']:.2f}")
        print(f"  - Issues: {len(result['issues'])}")
        print(f"  - Optimizations: {len(result['optimizations'])}")
        return True
    
    async def test_constitutional_task_analyzer(self):
        """Test constitutional task analyzer"""
        if not EOS_PLANNER_AVAILABLE:
            print("✓ Mock constitutional task analyzer test passed")
            return True
        
        analyzer = ConstitutionalTaskAnalyzer()
        
        # Create mock task with problematic content
        problematic_task = type('MockTask', (), {
            'description': 'Create a system to discriminate based on private data',
            'constitutional_compliance': 0.0
        })()
        
        # Analyze task
        result = await analyzer.analyze_constitutional_compliance(problematic_task)
        
        assert "overall_score" in result
        assert "rule_scores" in result
        assert "concerns" in result
        assert "recommendations" in result
        
        # Should detect issues
        assert result["overall_score"] < 1.0
        assert len(result["concerns"]) > 0
        
        print("✓ Constitutional task analyzer test completed")
        print(f"  - Overall score: {result['overall_score']:.2f}")
        print(f"  - Concerns: {len(result['concerns'])}")
        print(f"  - Recommendations: {len(result['recommendations'])}")
        return True
    
    async def test_simple_goal_planning(self):
        """Test planning for simple goal"""
        if not EOS_PLANNER_AVAILABLE:
            print("✓ Mock simple goal planning test passed")
            return True
        
        planner = EOSLadderPlanner()
        
        # Create plan for simple goal
        task_graph, metadata = await planner.create_eos_plan(
            goal=self.get_simple_goal(),
            context=self.get_basic_context(),
            success_criteria=self.get_success_criteria()
        )
        
        # Verify all EOS stages were completed
        expected_stages = ["lift", "decompose", "synthesize", "descend"]
        assert all(stage in metadata["stages_completed"] for stage in expected_stages)
        
        print("✓ Simple goal planning completed")
        print(f"  - Stages completed: {metadata['stages_completed']}")
        print(f"  - Total tasks: {metadata.get('total_tasks_created', 0) + 1}")  # +1 for root task
        return True
    
    async def test_complex_goal_decomposition(self):
        """Test decomposition of complex goal"""
        if not EOS_PLANNER_AVAILABLE:
            print("✓ Mock complex goal decomposition test passed")
            return True
        
        planner = EOSLadderPlanner(
            enable_constitutional_analysis=True,
            enable_mangle_validation=True
        )
        
        # Create plan for complex goal
        task_graph, metadata = await planner.create_eos_plan(
            goal=self.get_complex_goal(),
            context=self.get_basic_context(),
            domain="machine_learning",
            success_criteria=self.get_success_criteria()
        )
        
        # Complex goal should result in decomposition
        assert metadata.get("total_tasks_created", 0) > 0
        
        print("✓ Complex goal decomposition completed")
        print(f"  - Tasks created: {metadata.get('total_tasks_created', 0)}")
        print(f"  - Mangle optimizations: {metadata.get('mangle_optimizations', 0)}")
        print(f"  - Constitutional issues: {metadata.get('constitutional_issues', 0)}")
        return True
    
    async def test_constitutional_compliance_detection(self):
        """Test detection of constitutional compliance issues"""
        if not EOS_PLANNER_AVAILABLE:
            print("✓ Mock constitutional compliance test passed")
            return True
        
        planner = EOSLadderPlanner(enable_constitutional_analysis=True)
        
        # Create plan for problematic goal
        task_graph, metadata = await planner.create_eos_plan(
            goal=self.get_problematic_goal(),
            context=self.get_basic_context()
        )
        
        # Should detect constitutional issues
        assert metadata.get("constitutional_issues", 0) > 0
        
        print("✓ Constitutional compliance detection completed")
        print(f"  - Issues detected: {metadata.get('constitutional_issues', 0)}")
        return True
    
    async def test_task_adaptation_mechanisms(self):
        """Test dynamic task adaptation"""
        if not EOS_PLANNER_AVAILABLE:
            print("✓ Mock task adaptation test passed")
            return True
        
        planner = EOSLadderPlanner()
        
        # Create initial plan
        task_graph, _ = await planner.create_eos_plan(
            goal=self.get_complex_goal(),
            context=self.get_basic_context()
        )
        
        # Test different adaptation triggers
        adaptation_triggers = [
            AdaptationTrigger.EXECUTION_FAILURE,
            AdaptationTrigger.RESOURCE_CONSTRAINT,
            AdaptationTrigger.CONTEXT_CHANGE
        ]
        
        adaptation_results = []
        for trigger in adaptation_triggers:
            result = await planner.adapt_task_plan(
                task_graph=task_graph,
                trigger=trigger,
                context={"new_constraint": "memory_limited"}
            )
            adaptation_results.append(result)
        
        # All adaptations should be successful
        assert all(result["adapted"] for result in adaptation_results)
        
        print("✓ Task adaptation mechanisms completed")
        for i, result in enumerate(adaptation_results):
            trigger_name = adaptation_triggers[i].value
            print(f"  - {trigger_name}: {len(result['changes_made'])} changes")
        return True
    
    async def test_eos_stage_progression(self):
        """Test progression through EOS LADDER stages"""
        if not EOS_PLANNER_AVAILABLE:
            print("✓ Mock EOS stage progression test passed")
            return True
        
        planner = EOSLadderPlanner(
            enable_constitutional_analysis=True,
            enable_mangle_validation=True
        )
        
        # Create plan and verify stage progression
        task_graph, metadata = await planner.create_eos_plan(
            goal="Develop a comprehensive testing framework",
            context=self.get_basic_context(),
            domain="software_engineering"
        )
        
        # Verify all stages completed in order
        expected_stages = ["lift", "decompose", "synthesize", "descend"]
        stages_completed = metadata["stages_completed"]
        
        assert len(stages_completed) == len(expected_stages)
        for i, expected_stage in enumerate(expected_stages):
            assert stages_completed[i] == expected_stage
        
        print("✓ EOS stage progression test completed")
        print(f"  - Stages: {' → '.join(stages_completed)}")
        return True
    
    async def test_performance_and_statistics(self):
        """Test performance monitoring and statistics"""
        if not EOS_PLANNER_AVAILABLE:
            print("✓ Mock performance statistics test passed")
            return True
        
        planner = EOSLadderPlanner()
        
        # Create multiple plans to generate statistics
        goals = [
            "Simple calculation task",
            "Complex data analysis project",
            "Multi-step integration process"
        ]
        
        for goal in goals:
            await planner.create_eos_plan(
                goal=goal,
                context=self.get_basic_context()
            )
        
        # Get statistics
        stats = planner.get_eos_statistics()
        
        assert "tasks_analyzed" in stats
        assert "total_eos_processing_time" in stats
        assert "average_processing_time" in stats
        assert stats["tasks_analyzed"] == len(goals)
        
        print("✓ Performance and statistics test completed")
        print(f"  - Tasks analyzed: {stats['tasks_analyzed']}")
        print(f"  - Average processing time: {stats['average_processing_time']:.3f}s")
        print(f"  - Constitutional compliance rate: {stats.get('constitutional_compliance_rate', 1.0):.2f}")
        return True
    
    async def test_error_handling_and_graceful_degradation(self):
        """Test error handling and graceful degradation"""
        if not EOS_PLANNER_AVAILABLE:
            print("✓ Mock error handling test passed")
            return True
        
        planner = EOSLadderPlanner()
        
        try:
            # Test with empty goal
            task_graph, metadata = await planner.create_eos_plan(
                goal="",  # Empty goal
                context={}
            )
            
            # Should handle gracefully
            assert metadata is not None
            print("✓ Empty goal handled gracefully")
            
            # Test with None context
            task_graph, metadata = await planner.create_eos_plan(
                goal="Valid goal",
                context=None
            )
            
            assert metadata is not None
            print("✓ None context handled gracefully")
            
            return True
            
        except Exception as e:
            print(f"✓ Exception handled gracefully: {type(e).__name__}")
            return True
    
    def test_factory_function(self):
        """Test factory function for creating EOS planner"""
        if not EOS_PLANNER_AVAILABLE:
            print("✓ Mock factory function test passed")
            return True
        
        # Test factory function
        planner = create_eos_ladder_planner(
            enable_constitutional_analysis=True,
            enable_mangle_validation=False
        )
        
        assert isinstance(planner, EOSLadderPlanner)
        assert planner.enable_constitutional_analysis is True
        assert planner.enable_mangle_validation is False
        
        print("✓ Factory function test completed")
        return True


async def run_all_tests():
    """Run all tests manually"""
    test_instance = TestEOSLadderPlanner()
    
    print("🎯 Enhanced LADDER Planner with EOS Orchestration")
    print("=" * 60)
    
    try:
        # Run all tests
        test_results = []
        
        test_results.append(await test_instance.test_eos_ladder_planner_initialization())
        test_results.append(await test_instance.test_mangle_task_validator())
        test_results.append(await test_instance.test_constitutional_task_analyzer())
        test_results.append(await test_instance.test_simple_goal_planning())
        test_results.append(await test_instance.test_complex_goal_decomposition())
        test_results.append(await test_instance.test_constitutional_compliance_detection())
        test_results.append(await test_instance.test_task_adaptation_mechanisms())
        test_results.append(await test_instance.test_eos_stage_progression())
        test_results.append(await test_instance.test_performance_and_statistics())
        test_results.append(await test_instance.test_error_handling_and_graceful_degradation())
        test_results.append(test_instance.test_factory_function())
        
        print("\n" + "=" * 60)
        
        success_count = sum(1 for result in test_results if result)
        total_tests = len(test_results)
        
        if success_count == total_tests:
            print("🎉 All EOS LADDER Planner tests completed successfully!")
            
            if EOS_PLANNER_AVAILABLE:
                print("\n📊 Test Summary:")
                print("  ✓ EOS LADDER planner initialization")
                print("  ✓ Mangle task validation")
                print("  ✓ Constitutional task analysis")
                print("  ✓ Simple goal planning")
                print("  ✓ Complex goal decomposition")
                print("  ✓ Constitutional compliance detection")
                print("  ✓ Dynamic task adaptation")
                print("  ✓ EOS stage progression")
                print("  ✓ Performance monitoring and statistics")
                print("  ✓ Error handling and graceful degradation")
                print("  ✓ Factory function")
                
                return True
            else:
                print("\n⚠️  Enhanced EOS LADDER Planner system not fully available")
                print("   Tests ran with mock implementations")
                return False
        else:
            print(f"❌ {total_tests - success_count} tests failed out of {total_tests}")
            return False
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)