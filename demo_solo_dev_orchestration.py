#!/usr/bin/env python3
"""
Solo Developer Multi-Agent Orchestration Demo

This script demonstrates how to use the new multi-agent orchestration system
for solo developers. It shows:

1. Agent task routing based on specialization
2. Cost tracking and budget management
3. Performance analytics 
4. Multi-agent workflows
5. Capability mapping and recommendations

Run with: python demo_solo_dev_orchestration.py
"""

import asyncio
import logging
from datetime import UTC, datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def demo_solo_dev_orchestration():
    """Demonstrate the solo developer multi-agent orchestration system"""
    print("🚀 Solo Developer Multi-Agent Orchestration Demo")
    print("=" * 60)
    
    try:
        # Import the orchestration system
        from src.orchestration.solo_dev_orchestrator import (
            SoloDevMultiAgentOrchestrator, 
            OrchestrationConfig
        )
        from src.orchestration.agent_task_router import TaskType
        
        # Create configuration optimized for solo developer
        config = OrchestrationConfig(
            enable_cost_tracking=True,
            enable_performance_analytics=True,
            enable_capability_mapping=True,
            enable_handoff_protocol=True,
            default_task_timeout_minutes=30,
            max_concurrent_tasks_per_agent=2,  # Limit for cost control
            cost_alert_threshold=25.0,  # $25/day budget
            performance_alert_threshold=0.75  # 75% success rate minimum
        )
        
        # Initialize the orchestrator
        orchestrator = SoloDevMultiAgentOrchestrator(config=config)
        await orchestrator.start()
        
        print("✅ Orchestrator started with solo developer configuration")
        print(f"   - Daily budget limit: ${config.cost_alert_threshold}")
        print(f"   - Performance threshold: {config.performance_alert_threshold:.0%}")
        print(f"   - Max tasks per agent: {config.max_concurrent_tasks_per_agent}")
        print()
        
        # Demo 1: Security-focused workflow
        print("🔒 Demo 1: Security Vulnerability Assessment")
        print("-" * 40)
        
        security_task = await orchestrator.submit_task(
            task_description="Analyze authentication system for security vulnerabilities and suggest fixes",
            task_type=TaskType.SECURITY_SCAN,
            priority=9,  # High priority
            complexity=4,
            context={
                "target_files": ["auth/login.py", "auth/session.py", "auth/middleware.py"],
                "security_frameworks": ["OWASP Top 10", "NIST"],
                "compliance_requirements": ["SOC2", "GDPR"]
            }
        )
        
        print(f"   📋 Task submitted: {security_task}")
        print("   🎯 Routed to: Security Agent (specialized in vulnerability analysis)")
        
        # Simulate completion with realistic results
        await orchestrator.complete_task(
            task_id=security_task,
            success=True,
            cost=3.75,  # API costs for analysis
            quality_score=0.92,
            user_satisfaction=0.88,
            outputs_generated=[
                "security_assessment_report.md",
                "vulnerability_fixes.patch", 
                "security_test_cases.py"
            ],
            files_modified=["auth/login.py", "auth/session.py"],
            security_issues_found=5,
            code_lines_changed=47,
            provider="openai",
            tokens_used=2100,
            api_requests=3
        )
        
        print("   ✅ Security assessment completed")
        print("   📊 Found 5 security issues with fixes provided")
        print()
        
        # Demo 2: Feature development workflow  
        print("🛠️ Demo 2: API Feature Development")
        print("-" * 40)
        
        feature_task = await orchestrator.submit_task(
            task_description="Implement REST API endpoints for user profile management with validation",
            task_type=TaskType.FEATURE_DEVELOPMENT,
            priority=7,
            complexity=3,
            context={
                "framework": "FastAPI",
                "database": "PostgreSQL", 
                "requirements": [
                    "CRUD operations for user profiles",
                    "Input validation with Pydantic",
                    "Authentication required",
                    "Rate limiting"
                ]
            }
        )
        
        print(f"   📋 Task submitted: {feature_task}")
        print("   🎯 Routed to: Implementation Agent (expert in Python/API development)")
        
        await orchestrator.complete_task(
            task_id=feature_task,
            success=True,
            cost=5.20,
            quality_score=0.85,
            user_satisfaction=0.91,
            outputs_generated=[
                "api/user_profiles.py",
                "models/user_profile.py",
                "schemas/user_profile.py",
                "tests/test_user_profiles.py"
            ],
            files_modified=[
                "api/user_profiles.py",
                "models/user_profile.py", 
                "main.py"
            ],
            code_lines_changed=178,
            tests_passed=15,
            tests_failed=0,
            documentation_updated=True,
            provider="anthropic",
            tokens_used=3200,
            api_requests=4
        )
        
        print("   ✅ API endpoints implemented with tests")
        print("   📊 178 lines added, 15 tests passing")
        print()
        
        # Demo 3: Multi-agent workflow for complex project
        print("🔄 Demo 3: Multi-Agent Workflow - E-commerce Payment System")
        print("-" * 40)
        
        workflow_steps = [
            {
                "agent_id": "architecture_agent",
                "description": "Design secure payment processing architecture",
                "outputs_required": [
                    "payment_system_architecture.md",
                    "security_requirements.md",
                    "api_specifications.yaml"
                ],
                "verification_criteria": [
                    "Architecture supports PCI DSS compliance",
                    "Scalable to 10k+ transactions/hour",
                    "Proper separation of concerns"
                ]
            },
            {
                "agent_id": "security_agent",
                "description": "Security review and hardening recommendations",
                "depends_on": ["step_1"],
                "outputs_required": [
                    "security_review_report.md",
                    "threat_model.md",
                    "security_test_plan.md"
                ],
                "verification_criteria": [
                    "PCI DSS requirements addressed",
                    "Threat model comprehensive",
                    "Security controls specified"
                ]
            },
            {
                "agent_id": "implementation_agent",
                "description": "Implement payment processing endpoints",
                "depends_on": ["step_2"], 
                "outputs_required": [
                    "payment_processor.py",
                    "payment_models.py",
                    "payment_validation.py"
                ],
                "verification_criteria": [
                    "All API endpoints implemented",
                    "Input validation comprehensive",
                    "Error handling robust"
                ]
            },
            {
                "agent_id": "testing_agent",
                "description": "Comprehensive testing including security tests",
                "depends_on": ["step_3"],
                "outputs_required": [
                    "test_payment_processor.py",
                    "test_payment_security.py",
                    "integration_tests.py"
                ],
                "verification_criteria": [
                    ">95% code coverage achieved",
                    "All security tests passing",
                    "Performance tests included"
                ]
            }
        ]
        
        workflow_id = await orchestrator.create_multi_agent_workflow(
            workflow_description="Complete e-commerce payment system with security focus",
            steps=workflow_steps
        )
        
        print(f"   📋 Workflow created: {workflow_id}")
        print("   🔗 4-step workflow: Architecture → Security → Implementation → Testing")
        print("   ⏱️ Estimated duration: 2-3 hours with handoffs")
        print()
        
        # Demo 4: System analytics and recommendations
        print("📊 Demo 4: System Analytics & Recommendations")
        print("-" * 40)
        
        # Get system status
        status = await orchestrator.get_system_status()
        print("   📈 System Status:")
        print(f"   - Active tasks: {status['orchestrator']['active_tasks']}")
        print(f"   - Total tasks processed: {status['orchestrator']['stats']['total_tasks_processed']}")
        print(f"   - Total cost: ${status['orchestrator']['stats']['total_cost']:.2f}")
        print()
        
        # Get cost analysis
        cost_summary = orchestrator.cost_dashboard.get_cost_summary(1)  # Last day
        print("   💰 Cost Analysis:")
        print(f"   - Total cost (last 24h): ${cost_summary['total_cost']:.2f}")
        print(f"   - Average per task: ${cost_summary.get('average_cost_per_entry', 0):.2f}")
        print(f"   - Projected monthly: ${cost_summary.get('projected_monthly', 0):.2f}")
        print()
        
        # Get performance analytics
        performance_report = await orchestrator.performance_analytics.generate_performance_report()
        if performance_report['comparative_analysis']['agents']:
            best_agent = performance_report['comparative_analysis']['agents'][0]
            print("   🏆 Top Performing Agent:")
            print(f"   - Agent: {best_agent['agent_id']}")
            print(f"   - Success rate: {best_agent['success_rate']:.1%}")
            print(f"   - Avg duration: {best_agent['average_duration']:.1f} min")
            print(f"   - Tasks completed: {best_agent['total_tasks']}")
        print()
        
        # Get recommendations
        recommendations = await orchestrator.get_recommendations()
        print("   💡 System Recommendations:")
        for category, recs in recommendations.items():
            if recs:
                print(f"   {category.title()}:")
                for rec in recs[:2]:  # Show top 2 per category
                    print(f"     • {rec}")
        print()
        
        # Demo 5: Capability mapping insights
        print("🧠 Demo 5: Agent Capability Insights") 
        print("-" * 40)
        
        capability_analytics = orchestrator.capability_mapping.get_capability_analytics()
        print("   🎯 Capability Coverage:")
        print(f"   - Total capabilities tracked: {capability_analytics['total_capabilities']}")
        print(f"   - Agents configured: {capability_analytics['total_agents']}")
        
        if capability_analytics['most_common_capabilities']:
            print("   - Most common capabilities:")
            for cap in capability_analytics['most_common_capabilities'][:3]:
                print(f"     • {cap['capability_id']}: {cap['agent_count']} agents")
        
        gaps = capability_analytics['capability_gaps']
        if gaps:
            print(f"   - Capability gaps found: {len(gaps)}")
            for gap in gaps[:2]:
                print(f"     • {gap['severity']}: {gap['capability_name']}")
        print()
        
        # Cleanup
        await orchestrator.stop()
        
        print("🎉 Demo completed successfully!")
        print("=" * 60)
        print()
        print("Key Benefits for Solo Developers:")
        print("• 🎯 Intelligent task routing to specialized agents")
        print("• 💰 Cost tracking and budget management") 
        print("• 📊 Performance analytics and optimization")
        print("• 🔄 Multi-agent workflows for complex projects")
        print("• 🧠 Capability-based agent selection")
        print("• ⚡ Automated handoffs with context preservation")
        print()
        print("Ready to integrate into your development workflow! 🚀")
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main demo function"""
    success = await demo_solo_dev_orchestration()
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(result)