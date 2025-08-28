"""
Test script for Solo Developer Multi-Agent Orchestration System

This script validates that the new orchestration components work correctly.
"""

import asyncio
import logging
from datetime import UTC, datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_orchestration_system():
    """Test the complete orchestration system"""
    try:
        # Import the main orchestrator
        from src.orchestration.solo_dev_orchestrator import (
            SoloDevMultiAgentOrchestrator, 
            OrchestrationConfig
        )
        from src.orchestration.agent_task_router import TaskType
        
        logger.info("Testing Solo Developer Multi-Agent Orchestration System")
        
        # Create orchestrator with test configuration
        config = OrchestrationConfig(
            enable_cost_tracking=True,
            enable_performance_analytics=True,
            enable_capability_mapping=True,
            enable_handoff_protocol=True,
            cost_alert_threshold=10.0  # Low threshold for testing
        )
        
        orchestrator = SoloDevMultiAgentOrchestrator(config=config)
        
        # Start the orchestrator
        await orchestrator.start()
        logger.info("✅ Orchestrator started successfully")
        
        # Test 1: Submit a security scan task
        task_id_1 = await orchestrator.submit_task(
            task_description="Scan the authentication module for security vulnerabilities",
            task_type=TaskType.SECURITY_SCAN,
            priority=8,
            complexity=3,
            session_id="test_session",
            conversation_id="test_conversation"
        )
        logger.info(f"✅ Security scan task submitted: {task_id_1}")
        
        # Test 2: Submit a feature development task
        task_id_2 = await orchestrator.submit_task(
            task_description="Implement user registration API endpoint with validation",
            task_type=TaskType.FEATURE_DEVELOPMENT,
            priority=6,
            complexity=4,
            context={"file_path": "/api/users.py", "framework": "FastAPI"}
        )
        logger.info(f"✅ Feature development task submitted: {task_id_2}")
        
        # Test 3: Submit a testing task
        task_id_3 = await orchestrator.submit_task(
            task_description="Write unit tests for the payment processing module",
            task_type=TaskType.TESTING,
            priority=7,
            complexity=2
        )
        logger.info(f"✅ Testing task submitted: {task_id_3}")
        
        # Wait a moment for tasks to be processed
        await asyncio.sleep(1)
        
        # Test 4: Complete the security scan task
        await orchestrator.complete_task(
            task_id=task_id_1,
            success=True,
            cost=2.5,
            quality_score=0.9,
            user_satisfaction=0.85,
            outputs_generated=["security_report.md", "vulnerability_fixes.py"],
            files_modified=["auth/login.py", "auth/session.py"],
            security_issues_found=3,
            provider="openai",
            tokens_used=1500,
            api_requests=2
        )
        logger.info(f"✅ Completed security scan task: {task_id_1}")
        
        # Test 5: Complete the feature development task
        await orchestrator.complete_task(
            task_id=task_id_2,
            success=True,
            cost=4.2,
            quality_score=0.8,
            user_satisfaction=0.9,
            outputs_generated=["user_api.py", "user_schemas.py"],
            files_modified=["api/users.py", "models/user.py"],
            code_lines_changed=150,
            tests_passed=12,
            tests_failed=0,
            documentation_updated=True,
            provider="anthropic",
            tokens_used=2800,
            api_requests=3
        )
        logger.info(f"✅ Completed feature development task: {task_id_2}")
        
        # Test 6: Fail the testing task (to test failure handling)
        await orchestrator.complete_task(
            task_id=task_id_3,
            success=False,
            cost=1.1,
            quality_score=0.3,
            error_message="Unable to access payment module dependencies",
            provider="openai",
            tokens_used=800,
            api_requests=1
        )
        logger.info(f"✅ Completed testing task (failed): {task_id_3}")
        
        # Test 7: Create a multi-agent workflow
        workflow_steps = [
            {
                "agent_id": "architecture_agent",
                "description": "Design API architecture for billing system",
                "outputs_required": ["architecture_diagram", "api_specification"],
                "verification_criteria": ["Architecture is scalable", "APIs are RESTful"]
            },
            {
                "agent_id": "implementation_agent", 
                "description": "Implement billing API endpoints",
                "depends_on": ["step_1"],
                "outputs_required": ["billing_api.py", "billing_models.py"],
                "verification_criteria": ["All endpoints implemented", "Code follows standards"]
            },
            {
                "agent_id": "testing_agent",
                "description": "Write comprehensive tests for billing API",
                "depends_on": ["step_2"],
                "outputs_required": ["test_billing.py"],
                "verification_criteria": ["All endpoints tested", "95% code coverage"]
            }
        ]
        
        workflow_id = await orchestrator.create_multi_agent_workflow(
            workflow_description="Complete billing system implementation",
            steps=workflow_steps
        )
        logger.info(f"✅ Created multi-agent workflow: {workflow_id}")
        
        # Test 8: Get system status
        status = await orchestrator.get_system_status()
        logger.info(f"✅ System status retrieved: {status['orchestrator']['active_tasks']} active tasks")
        
        # Test 9: Get recommendations
        recommendations = await orchestrator.get_recommendations()
        logger.info(f"✅ Recommendations retrieved: {len(recommendations)} categories")
        
        # Test 10: Test individual components
        
        # Test agent router status
        agent_status = orchestrator.task_router.get_agent_status()
        logger.info(f"✅ Agent router: {len(agent_status)} agents tracked")
        
        # Test performance analytics
        performance_report = await orchestrator.performance_analytics.generate_performance_report()
        logger.info(f"✅ Performance analytics: {performance_report['report_type']} report generated")
        
        # Test cost dashboard
        cost_summary = orchestrator.cost_dashboard.get_cost_summary(1)  # Last day
        logger.info(f"✅ Cost dashboard: ${cost_summary['total_cost']:.2f} total cost")
        
        # Test capability mapping
        capability_analytics = orchestrator.capability_mapping.get_capability_analytics()
        logger.info(f"✅ Capability mapping: {capability_analytics['total_capabilities']} capabilities tracked")
        
        # Test budget status
        budget_status = orchestrator.cost_dashboard.get_budget_status()
        logger.info(f"✅ Budget tracking: {len(budget_status)} budget limits configured")
        
        # Test handoff protocol status
        if workflow_id in orchestrator.handoff_protocol.active_workflows:
            workflow_status = orchestrator.handoff_protocol.get_workflow_status(workflow_id)
            logger.info(f"✅ Handoff protocol: Workflow has {workflow_status['total_steps']} steps")
        
        await orchestrator.stop()
        logger.info("✅ Orchestrator stopped successfully")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("🎉 ALL TESTS PASSED - Solo Developer Multi-Agent Orchestration System is working!")
        logger.info("="*60)
        logger.info(f"Tasks processed: 3")
        logger.info(f"Workflows created: 1") 
        logger.info(f"Total cost tracked: ${cost_summary['total_cost']:.2f}")
        logger.info(f"Agents configured: {len(agent_status)}")
        logger.info(f"Capabilities tracked: {capability_analytics['total_capabilities']}")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    success = await test_orchestration_system()
    if success:
        print("\n✅ Solo Developer Multi-Agent Orchestration System test completed successfully!")
    else:
        print("\n❌ Solo Developer Multi-Agent Orchestration System test failed!")
        return 1
    return 0

if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(result)