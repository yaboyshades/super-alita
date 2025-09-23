"""Demo of LADDER-Cortex integration system."""

import asyncio

from src.core.event_bus import EventBus
from src.ladder.integration import LadderAdapter, LadderIntegrationConfig, PlanningMode
from src.orchestration.cortex_weaning import CortexWeaningOrchestrator


async def main():
    """Demonstrate LADDER-Cortex integration."""
    print("🚀 LADDER-Cortex Integration Demo")
    print("=" * 50)

    # Initialize components
    event_bus = EventBus()
    cortex_orchestrator = CortexWeaningOrchestrator()

    # Configure integration
    config = LadderIntegrationConfig(
        planning_mode=PlanningMode.ADAPTIVE,
        max_planning_horizon=4,
        use_cortex_for_decomposition=True,
        cortex_confidence_threshold=0.7,
        autonomous_task_threshold=0.8,
        max_concurrent_cortex_tasks=3,
    )

    # Create LADDER adapter
    adapter = LadderAdapter(
        event_bus=event_bus,
        cortex_orchestrator=cortex_orchestrator,
        config=config,
    )

    # Initialize async components
    await adapter.initialize()

    print("🔗 Integration initialized")
    print(f"📊 Current Cortex phase: {cortex_orchestrator.current_phase.value}")
    print(f"🎯 Planning mode: {config.planning_mode.value}")
    print()

    # Demo scenarios with different complexity levels
    scenarios = [
        {
            "goal": "Write a simple Python function to calculate factorial",
            "context": {"complexity": "low", "autonomy_confidence": 0.9},
        },
        {
            "goal": "Design and implement a REST API for user management",
            "context": {"complexity": "medium", "autonomy_confidence": 0.6},
        },
        {
            "goal": "Build a distributed microservices architecture",
            "context": {"complexity": "high", "autonomy_confidence": 0.3},
        },
    ]

    # Execute scenarios
    for i, scenario in enumerate(scenarios, 1):
        print(f"🎯 Scenario {i}: {scenario['goal']}")
        print(f"   Complexity: {scenario['context']['complexity']}")
        print(f"   Confidence: {scenario['context']['autonomy_confidence']}")

        # Execute planning and execution
        result = await adapter.plan_and_execute(
            goal=scenario["goal"],
            context=scenario["context"],
            session_id=f"demo_scenario_{i}",
        )

        print(f"   ✅ Success: {result.success}")
        print(f"   ⏱️  Execution time: {result.execution_time:.2f}s")
        print(f"   🔧 Tools used: {len(result.tools_used)}")
        print()

        # Show integration status after each scenario
        status = adapter.get_integration_status()
        print("📈 Integration Status:")
        print(f"   Active plans: {status['active_plans']}")
        print(f"   Success rate: {status['metrics']['success_rate']:.1%}")
        print(f"   Cortex dependency: {status['metrics']['cortex_dependency']:.1%}")
        print(f"   Avg planning time: {status['metrics']['avg_planning_time']:.2f}s")
        print()

        # Simulate autonomy improvement
        if i < len(scenarios):
            await cortex_orchestrator.advance_phase_if_ready(0.8)
            print(f"🔄 Cortex phase: {cortex_orchestrator.current_phase.value}")
            print()

    # Final integration metrics
    print("🏁 Final Integration Metrics")
    print("=" * 30)

    final_status = adapter.get_integration_status()
    metrics = final_status["metrics"]

    print(f"Total plans created: {metrics['total_plans']}")
    print(f"Overall success rate: {metrics['success_rate']:.1%}")
    print(f"Cortex dependency ratio: {metrics['cortex_dependency']:.1%}")
    print(f"Average planning time: {metrics['avg_planning_time']:.3f}s")
    print(f"Average execution time: {metrics['avg_execution_time']:.3f}s")

    planner_config = final_status["planner_config"]
    print(f"Final planning depth: {planner_config['max_depth']}")
    print(f"Shadow mode: {planner_config['shadow_mode']}")
    print(f"Max concurrent tasks: {planner_config['max_concurrent']}")

    print("\n🎉 LADDER-Cortex integration demo completed!")


async def demo_replanning():
    """Demonstrate dynamic replanning capabilities."""
    print("\n🔄 Replanning Demo")
    print("=" * 20)

    # Setup
    event_bus = EventBus()
    cortex_orchestrator = CortexWeaningOrchestrator()
    config = LadderIntegrationConfig(replanning_threshold=0.4)
    adapter = LadderAdapter(event_bus, cortex_orchestrator, config)

    # Initial context
    initial_context = {
        "framework": "Flask",
        "database": "SQLite",
        "frontend": "HTML/CSS",
        "team_size": 1,
    }

    # Start planning
    goal = "Build a web application for inventory management"
    print(f"🎯 Goal: {goal}")
    print(f"📋 Initial context: {initial_context}")

    # Simulate context changes
    changed_context = {
        "framework": "Django",  # Changed
        "database": "PostgreSQL",  # Changed
        "frontend": "React",  # Changed
        "team_size": 3,  # Changed
        "deployment": "Docker",  # New requirement
    }

    print(f"🔄 Changed context: {changed_context}")

    # Check if replanning is needed
    session_id = "replanning_demo"
    adapter.execution_contexts[session_id] = initial_context

    needs_replan = await adapter.replan_if_needed(session_id, changed_context)
    print(f"🤔 Replanning needed: {needs_replan}")

    if needs_replan:
        print("✅ Replanning would be triggered!")
    else:
        print("ℹ️  Current plan can accommodate changes")


async def demo_event_handling():
    """Demonstrate event-driven integration."""
    print("\n📡 Event Handling Demo")
    print("=" * 25)

    # Setup
    event_bus = EventBus()
    cortex_orchestrator = CortexWeaningOrchestrator()
    adapter = LadderAdapter(event_bus, cortex_orchestrator)

    # Demo event types
    events = [
        {
            "type": "autonomy_update",
            "data": {"score": 0.85, "phase": "supervised"},
        },
        {
            "type": "task_completion",
            "data": {"task_id": "task_123", "success": True},
        },
        {
            "type": "cortex_intervention",
            "data": {"session_id": "session_456", "type": "guidance"},
        },
        {
            "type": "planning_request",
            "data": {
                "goal": "Optimize database queries",
                "context": {"database": "PostgreSQL"},
                "session_id": "optimization_task",
            },
        },
    ]

    # Emit events and observe handling
    for event in events:
        print(f"📤 Emitting: {event['type']}")

        if event["type"] == "autonomy_update":
            await adapter._handle_autonomy_update(event["data"])
        elif event["type"] == "task_completion":
            await adapter._handle_task_completion(event["data"])
        elif event["type"] == "cortex_intervention":
            await adapter._handle_cortex_intervention(event["data"])
        elif event["type"] == "planning_request":
            await adapter._handle_planning_request(event["data"])

        print("✅ Event handled successfully")
        print()


if __name__ == "__main__":
    # Run all demos
    asyncio.run(main())
    asyncio.run(demo_replanning())
    asyncio.run(demo_event_handling())
