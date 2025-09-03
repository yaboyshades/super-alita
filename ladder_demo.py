"""Demo of the LADDER planner sys    # Generate the plan
    print("📋 Creating hierarchical plan...")
    task_graph = await planner.create_plan(
        goal=goal,
        template=TASK_TEMPLATES["implement"],
        context={
            "framework": "FastAPI",
            "database": "SQLite",
            "frontend": "React",
        }
    )import asyncio

from src.ladder import (
    ExecutionStrategy,
    LadderPlanner,
    PlannerConfig,
)
from src.ladder.models.task import TASK_TEMPLATES


async def main():
    """Run a simple LADDER planner demo."""
    print("🪜 LADDER Planner Demo")
    print("=" * 50)

    # Configure the planner
    config = PlannerConfig(
        max_decomposition_depth=3,
        max_concurrent_tasks=2,
        shadow_mode=True,  # Safe testing mode
        debug_mode=True,
    )

    # Create planner instance
    planner = LadderPlanner(config=config)

    # Create a plan for a complex goal
    goal = "Create a web application for task management"
    print(f"🎯 Goal: {goal}")
    print()

    # Generate the plan
    print("📋 Creating hierarchical plan...")
    task_graph = await planner.create_plan(
        goal=goal,
        template=TaskTemplate.DEVELOPMENT,
        context={
            "framework": "FastAPI",
            "database": "SQLite",
            "frontend": "React",
        },
    )

    # Show plan structure
    print(f"✅ Plan created with {len(task_graph.get_all_task_ids())} tasks")

    # Display execution order
    execution_order = task_graph.get_execution_order()
    print(f"📊 Execution phases: {len(execution_order)}")

    for i, phase in enumerate(execution_order, 1):
        print(f"  Phase {i}: {len(phase)} parallel tasks")
        for task_id in phase:
            task = task_graph.get_task(task_id)
            if task:
                print(f"    - {task.description[:60]}...")

    print()

    # Execute the plan
    print("🚀 Executing plan...")
    result = await planner.execute_plan(
        task_graph=task_graph,
        strategy=ExecutionStrategy.PARALLEL_SAFE,
    )

    # Show results
    print(f"✅ Execution completed: {result.success}")
    print(f"⏱️  Total time: {result.execution_time:.2f}s")
    print(f"🔧 Tools used: {len(result.tools_used)}")

    # Show metrics
    metrics = planner.get_metrics()
    print()
    print("📈 Execution Metrics:")
    print(f"  Total tasks: {metrics.total_tasks}")
    print(f"  Completed tasks: {metrics.completed_tasks}")
    print(f"  Completion rate: {metrics.completion_rate:.1%}")
    print(f"  Total energy: {metrics.total_energy}")
    if metrics.execution_time > 0:
        print(f"  Execution time: {metrics.execution_time:.2f}s")

    # Show planner status
    status = planner.get_status()
    print()
    print("🎛️  Planner Status:")
    print(f"  Total tasks created: {status['total_tasks']}")
    print(f"  Tasks completed: {status['completed_tasks']}")
    print(f"  Decomposition depth: {status['decomposition_depth']}")
    print(f"  Shadow mode: {status['shadow_mode']}")


if __name__ == "__main__":
    asyncio.run(main())
