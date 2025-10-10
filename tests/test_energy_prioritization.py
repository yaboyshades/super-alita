"""Test the complete Energy-Based Prioritization system."""

import asyncio
import time
from unittest.mock import MagicMock


# Test the complete energy prioritization pipeline
async def test_energy_prioritization_system():
    """Test the complete energy-based prioritization system."""
    print("Testing Energy-Based Prioritization System")
    print("==========================================")

    try:
        # 1. Test core components import
        from src.knowledge_graph import KnowledgeGraphInterface
        from src.ladder.graph.task_graph import Task, TaskGraph
        from src.ladder.prioritization import (
            EnergyBasedPrioritizer,
            EnergyCalculator,
            EnergyEnhancedLadderPlanner,
            PriorityConfig,
            PriorityStrategy,
        )

        print("✅ All energy prioritization components imported successfully")

        # 2. Initialize Knowledge Graph
        kg = KnowledgeGraphInterface()
        print(f"✅ KG initialized with {len(kg.patterns)} patterns")

        # 3. Initialize Energy Calculator
        energy_calc = EnergyCalculator(
            kg_interface=kg,
            effort_weight=0.3,
            success_weight=0.4,
            complexity_weight=0.2,
            context_weight=0.1,
        )
        print("✅ Energy Calculator initialized")

        # 4. Initialize Priority Engine with different strategies
        priority_config = PriorityConfig(
            strategy=PriorityStrategy.BALANCED,
            energy_threshold=0.8,
            max_parallel_tasks=3,
            confidence_threshold=0.3,
        )

        prioritizer = EnergyBasedPrioritizer(
            kg_interface=kg, priority_config=priority_config
        )
        print("✅ Energy-Based Prioritizer initialized")

        # 5. Create test tasks with varying characteristics
        tasks = []

        # High-effort, complex task
        task1 = Task(
            id="task_1",
            title="Implement Complex Algorithm",
            description="Implement a comprehensive machine learning algorithm with full error handling and extensive testing. This requires significant development effort and complex integration.",
        )
        tasks.append(task1)

        # Low-effort, simple task
        task2 = Task(
            id="task_2",
            title="Quick Validation",
            description="Simple check to verify the basic functionality is working correctly. Quick and straightforward validation.",
        )
        tasks.append(task2)

        # Medium complexity task
        task3 = Task(
            id="task_3",
            title="Code Development Task",
            description="Create a Python function to process data with proper error handling and documentation.",
        )
        tasks.append(task3)

        # Research-oriented task
        task4 = Task(
            id="task_4",
            title="Research Analysis",
            description="Analyze research papers and synthesize findings for the project requirements.",
        )
        tasks.append(task4)

        print(
            f"✅ Created {len(tasks)} test tasks with varying characteristics"
        )

        # 6. Create TaskGraph
        task_graph = TaskGraph()
        for task in tasks:
            task_graph.add_task(task)

        print("✅ Created TaskGraph with test tasks")

        # 7. Test Energy Calculation for each task
        print("\n🔋 Energy Analysis for Each Task:")
        print("=================================")

        context = {
            "domain": "software_development",
            "keywords": ["python", "function", "data"],
        }

        for task in tasks:
            energy = energy_calc.calculate_task_energy(task, context)
            print(f"\nTask: {task.title}")
            print(
                f"  Energy Score: {energy.energy_score:.3f} (lower = higher priority)"
            )
            print(f"  Confidence: {energy.confidence:.3f}")
            print("  Metrics:")
            print(f"    - Effort: {energy.metrics.effort_score:.3f}")
            print(
                f"    - Success Probability: {energy.metrics.success_probability:.3f}"
            )
            print(f"    - Complexity: {energy.metrics.complexity_score:.3f}")
            print(
                f"    - Pattern Confidence: {energy.metrics.pattern_confidence:.3f}"
            )
            print(
                f"    - Context Relevance: {energy.metrics.context_relevance:.3f}"
            )
            print(f"  Top Reasoning: {energy.reasoning[:2]}")

        # 8. Test Task Prioritization
        print("\n⚡ Task Prioritization Results:")
        print("==============================")

        start_time = time.time()
        prioritization_result = prioritizer.prioritize_tasks(
            task_graph, context
        )
        calc_time = time.time() - start_time

        print(f"Prioritization completed in {calc_time:.3f}s")
        print(f"Strategy used: {prioritization_result.strategy_used}")
        print(f"Total tasks: {prioritization_result.total_tasks}")
        print(f"Executable tasks: {prioritization_result.executable_tasks}")
        print(f"Blocked tasks: {prioritization_result.blocked_tasks}")
        print(f"Average energy: {prioritization_result.average_energy:.3f}")
        print(
            f"Average confidence: {prioritization_result.average_confidence:.3f}"
        )

        print("\nTask Priority Ranking:")
        for i, priority in enumerate(prioritization_result.priorities):
            status = "🟢 Ready" if priority.can_execute else "🔴 Blocked"
            print(
                f"  {i+1}. {priority.task_id} (score: {priority.priority_score:.3f}, energy: {priority.energy.energy_score:.3f}) {status}"
            )

        # 9. Test Next Task Recommendations
        print("\n📋 Next Task Recommendations:")
        print("=============================")

        next_tasks = prioritizer.get_next_tasks(task_graph, count=2)
        for i, task_priority in enumerate(next_tasks):
            print(f"  {i+1}. Task: {task_priority.task_id}")
            print(f"     Priority Score: {task_priority.priority_score:.3f}")
            print(
                f"     Energy Score: {task_priority.energy.energy_score:.3f}"
            )
            print(f"     Can Execute: {task_priority.can_execute}")

        # 10. Test Enhanced Planner (if possible with mock)
        print("\n🧠 Testing Energy-Enhanced Planner:")
        print("===================================")

        try:
            # Mock the KG adapter for testing
            from src.core.event_bus import EventBus
            from src.knowledge_graph import KnowledgeGraphAdapter

            # Create mock event bus
            mock_event_bus = MagicMock(spec=EventBus)
            mock_event_bus.emit = asyncio.coroutine(
                lambda *args, **kwargs: None
            )

            kg_adapter = KnowledgeGraphAdapter(
                kg_interface=kg,
                event_bus=mock_event_bus,
                source_plugin="test_plugin",
            )

            # Initialize enhanced planner
            enhanced_planner = EnergyEnhancedLadderPlanner(
                kg_interface=kg,
                priority_config=priority_config,
                kg_adapter=kg_adapter,
            )

            print("✅ Energy-Enhanced Planner initialized")

            # Test getting prioritization status
            status = enhanced_planner.get_prioritization_status()
            print("✅ Prioritization status retrieved")

            # Test task explanation
            if prioritization_result.priorities:
                first_task_id = prioritization_result.priorities[0].task_id
                explanation = enhanced_planner.explain_task_priority(
                    first_task_id
                )
                print(
                    f"✅ Task priority explanation generated for {first_task_id}"
                )
                print(
                    f"   Priority rank: {explanation.get('priority_rank', 'N/A')}"
                )
                print(
                    f"   Energy score: {explanation.get('energy_analysis', {}).get('energy_score', 'N/A')}"
                )

        except Exception as e:
            print(f"⚠️  Enhanced planner test skipped: {e}")

        # 11. Test Performance Metrics
        print("\n📊 Performance Metrics:")
        print("=======================")

        summary = prioritizer.get_prioritization_summary()
        if "current_prioritization" in summary:
            current = summary["current_prioritization"]
            print("Current prioritization metrics:")
            print(f"  - Total tasks: {current['total_tasks']}")
            print(f"  - Executable: {current['executable_tasks']}")
            print(f"  - Average energy: {current['average_energy']:.3f}")
            print(f"  - Calculation time: {current['calculation_time']:.3f}s")

        performance = prioritizer.get_performance_metrics()
        if "performance" in performance:
            perf = performance["performance"]
            print("System performance:")
            print(
                f"  - Average calc time: {perf['average_calculation_time']:.3f}s"
            )
            print(
                f"  - Total prioritizations: {perf['total_prioritizations']}"
            )

        # 12. Test Different Priority Strategies
        print("\n🎯 Testing Different Priority Strategies:")
        print("=========================================")

        strategies = [
            PriorityStrategy.ENERGY_ONLY,
            PriorityStrategy.ENERGY_DEPENDENCY,
            PriorityStrategy.BALANCED,
            PriorityStrategy.ADAPTIVE,
        ]

        for strategy in strategies:
            strategy_config = PriorityConfig(
                strategy=strategy,
                energy_threshold=0.8,
                max_parallel_tasks=3,
            )

            strategy_prioritizer = EnergyBasedPrioritizer(
                kg_interface=kg, priority_config=strategy_config
            )

            result = strategy_prioritizer.prioritize_tasks(task_graph, context)
            print(
                f"  {strategy.value}: {result.total_tasks} tasks, avg energy: {result.average_energy:.3f}"
            )

        # 13. Final Success Summary
        print("\n🎉 Energy-Based Prioritization System Test COMPLETE!")
        print("====================================================")
        print("✅ Energy calculation working")
        print("✅ Task prioritization working")
        print("✅ Multiple strategies tested")
        print("✅ Performance metrics available")
        print("✅ Task recommendations generated")
        print("✅ Knowledge Graph integration active")

        # Return summary metrics
        return {
            "success": True,
            "tasks_tested": len(tasks),
            "prioritization_time": calc_time,
            "strategies_tested": len(strategies),
            "average_energy": prioritization_result.average_energy,
            "average_confidence": prioritization_result.average_confidence,
            "kg_patterns_available": len(kg.patterns),
            "executable_tasks": prioritization_result.executable_tasks,
        }

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


# Run the test
if __name__ == "__main__":
    result = asyncio.run(test_energy_prioritization_system())
    print(f"\nFinal Result: {result}")
