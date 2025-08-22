"""
Enhanced LADDER Planner Usage Example

This script demonstrates how to use the Enhanced LADDER Planner with
advanced features like multi-armed bandit learning, energy-based prioritization,
and shadow/active execution modes.
"""

import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate Enhanced LADDER Planner usage."""
    print("=" * 60)
    print("🚀 Enhanced LADDER Planner Demo")
    print("=" * 60)

    # Import here to avoid issues if dependencies aren't installed
    try:
        from cortex.config.planner_config import get_planner_config
        from cortex.planner.ladder_enhanced import EnhancedLadderPlanner

        # Import mock components for demonstration
        from tests.planner.test_ladder_enhanced import (
            MockBandit,
            MockKG,
            MockOrchestrator,
            MockTodoStore,
        )

        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return

    # Create mock components (in real usage, these would be your actual implementations)
    print("\n📦 Setting up components...")
    kg = MockKG()
    bandit = MockBandit()
    store = MockTodoStore()
    orchestrator = MockOrchestrator()

    # Create Enhanced LADDER Planner
    print("🧠 Creating Enhanced LADDER Planner...")
    planner = EnhancedLadderPlanner(
        kg=kg,
        bandit=bandit,
        store=store,
        orchestrator=orchestrator,
        mode="shadow",  # Start in safe shadow mode
    )

    # Display configuration
    config = get_planner_config()
    print("⚙️  Configuration:")
    print(f"   Mode: {planner.mode}")
    print(f"   Exploration Rate: {planner.exploration_rate}")
    print(f"   Max Tasks: {config.max_tasks}")
    print(f"   Energy Threshold: {config.energy_threshold}")

    # Example 1: Test automation
    print("\n" + "=" * 50)
    print("📋 Example 1: Test Automation")
    print("=" * 50)

    test_event = type(
        "UserEvent",
        (),
        {
            "payload": {
                "query": "Run comprehensive tests with coverage reporting",
                "context": "Need to run full test suite for Super Alita project with coverage metrics",
            }
        },
    )()

    start_time = datetime.now()
    test_plan = await planner.plan_from_user_event(test_event)
    execution_time = (datetime.now() - start_time).total_seconds()

    print(f"✅ Test plan created in {execution_time:.2f}s")
    print(f"   Plan ID: {test_plan.id}")
    print(f"   Title: {test_plan.title}")
    print(f"   Energy: {test_plan.energy:.2f}")
    print(f"   Children: {len(test_plan.children_ids)}")

    # Show children tasks
    if test_plan.children_ids:
        print("   Subtasks:")
        for i, child_id in enumerate(test_plan.children_ids, 1):
            child = store.get(child_id)
            if child:
                print(
                    f"     {i}. {child.title} (energy: {child.energy:.2f}, tool: {child.tool_hint})"
                )

    # Example 2: Code Quality Pipeline
    print("\n" + "=" * 50)
    print("🔧 Example 2: Code Quality Pipeline")
    print("=" * 50)

    quality_event = type(
        "UserEvent",
        (),
        {
            "payload": {
                "query": "Format, lint, and check code quality",
                "context": "Prepare code for production deployment",
            }
        },
    )()

    start_time = datetime.now()
    quality_plan = await planner.plan_from_user_event(quality_event)
    execution_time = (datetime.now() - start_time).total_seconds()

    print(f"✅ Quality plan created in {execution_time:.2f}s")
    print(f"   Plan ID: {quality_plan.id}")
    print(f"   Total Energy: {quality_plan.energy:.2f}")

    # Example 3: Switch to Active Mode and Execute
    print("\n" + "=" * 50)
    print("⚡ Example 3: Active Mode Execution")
    print("=" * 50)

    print("🔄 Switching to active mode...")
    planner.set_mode("active")
    print(f"   Current mode: {planner.mode}")

    simple_event = type(
        "UserEvent",
        (),
        {
            "payload": {
                "query": "Simple formatting task",
                "context": "Format a single Python file",
            }
        },
    )()

    start_time = datetime.now()
    simple_plan = await planner.plan_from_user_event(simple_event)
    execution_time = (datetime.now() - start_time).total_seconds()

    print(f"✅ Simple plan executed in {execution_time:.2f}s")
    print(f"   Orchestrator calls: {len(orchestrator.executions)}")

    # Example 4: Bandit Learning Statistics
    print("\n" + "=" * 50)
    print("📊 Example 4: Learning Statistics")
    print("=" * 50)

    bandit_stats = planner.get_bandit_stats()
    kb_summary = planner.get_knowledge_base_summary()

    print("🤖 Bandit Statistics:")
    if bandit_stats:
        for tool, stats in bandit_stats.items():
            attempts = stats.get("attempts", 0)
            wins = stats.get("wins", 0)
            success_rate = (wins / attempts * 100) if attempts > 0 else 0
            print(f"   {tool}: {wins}/{attempts} ({success_rate:.1f}% success)")
    else:
        print("   No bandit statistics available yet")

    print("🧠 Knowledge Base:")
    print(f"   Size: {kb_summary.get('size', 0)} entries")
    print(f"   Success Rate: {kb_summary.get('success_rate', 0.0):.1%}")
    print(f"   Average Reward: {kb_summary.get('average_reward', 0.0):.2f}")

    # Example 5: Event Stream Analysis
    print("\n" + "=" * 50)
    print("📡 Example 5: Event Stream Analysis")
    print("=" * 50)

    events = orchestrator.event_bus.events
    print(f"📨 Total events emitted: {len(events)}")

    # Group events by type
    event_counts = {}
    for event in events:
        event_type = event[1]  # event[1] is the event type
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

    print("   Event breakdown:")
    for event_type, count in sorted(event_counts.items()):
        print(f"     {event_type}: {count}")

    # Performance Summary
    print("\n" + "=" * 60)
    print("🏁 Performance Summary")
    print("=" * 60)

    total_plans = 3
    total_tasks = sum(
        len(plan.children_ids) for plan in [test_plan, quality_plan, simple_plan]
    )

    print("📈 Execution Summary:")
    print(f"   Plans Created: {total_plans}")
    print(f"   Total Tasks: {total_tasks}")
    print(f"   Events Emitted: {len(events)}")
    print(f"   Knowledge Entries: {kb_summary.get('size', 0)}")
    print(f"   Bandit Tools: {len(bandit_stats)}")

    print("\n✨ Enhanced LADDER Planner demo completed successfully!")
    print("\n🚀 Key Features Demonstrated:")
    print("   ✅ Advanced task decomposition strategies")
    print("   ✅ Multi-armed bandit tool selection")
    print("   ✅ Energy-based task prioritization")
    print("   ✅ Shadow/Active execution modes")
    print("   ✅ Knowledge base learning")
    print("   ✅ Comprehensive event tracking")

    print("\n📚 Next Steps:")
    print("   1. Integrate with your actual KG, Bandit, and Orchestrator")
    print("   2. Set up FastAPI endpoints for web access")
    print("   3. Configure environment variables for production")
    print("   4. Run comprehensive tests with: pytest tests/planner/")
    print("   5. Monitor performance with the provided metrics")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
