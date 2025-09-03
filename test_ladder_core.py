"""
Simple LADDER Core Test

This tests just the core LADDER planne        print("✅ Execution completed!")
        print(f"   Success: {result.success}")
        result_text = str(result.result) if result.result else "No result"
        print(f"   Result: {result_text[:100]}...")unctionality without integration.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ladder.planner import LadderPlanner, PlannerConfig
from src.ladder.policies.bandit import UCB1Policy


async def main():
    """Test LADDER core functionality."""
    print("🚀 Testing LADDER Core Components")
    print("=" * 40)

    # Create UCB1 bandit policy
    bandit_policy = UCB1Policy(
        tools=["WebSearchTool", "CalculatorTool", "LLMTool"], exploration_factor=1.414
    )
    print("✅ UCB1 Bandit Policy created")

    # Create configuration for shadow mode
    config = PlannerConfig(
        shadow_mode=True,  # Safe testing mode
        debug_mode=True,  # More detailed output
    )

    # Create LADDER planner
    planner = LadderPlanner(bandit_policy=bandit_policy, config=config)
    print("✅ LADDER Planner created in shadow mode")

    # Test planning with a simple task
    user_query = "Research quantum computing and create a summary"

    print(f"\n📋 User Query: {user_query}")
    print("\n🔄 Creating plan with LADDER...")

    # Create a plan
    try:
        plan = await planner.create_plan(
            goal=user_query, context={"priority": "high", "domain": "technology"}
        )

        print("✅ Plan created successfully!")
        print(f"   Plan ID: {plan.name}")
        print(f"   Total Tasks: {len(plan.get_all_task_ids())}")

        # Show plan structure
        all_tasks = plan.get_all_task_ids()
        for task_id in all_tasks[:3]:  # Show first 3 tasks
            task = plan.get_task(task_id)
            if task:
                print(f"   Task: {task.description[:50]}...")

        if len(all_tasks) > 3:
            print(f"   ... and {len(all_tasks) - 3} more tasks")

        # Test execution in shadow mode
        print("\n🔄 Executing plan in shadow mode...")
        result = await planner.execute_plan(plan)

        print("✅ Execution completed!")
        print(f"   Success: {result.success}")
        result_text = str(result.result) if result.result else "No result"
        print(f"   Result: {result_text[:100]}...")

        # Show bandit learning
        print("\n🎯 Bandit Policy Status:")
        metrics = bandit_policy.get_tool_stats()
        for tool_name, stats in metrics.items():
            print(f"   {tool_name}: {stats.get('uses', 0)} uses")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n✅ LADDER Core Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
