"""Energy-enhanced LADDER planner with prioritization."""

import time
from typing import Any

from src.knowledge_graph import KnowledgeGraphInterface
from src.ladder.graph.task_graph import TaskGraph
from src.ladder.kg_enhanced_planner import KGEnhancedLadderPlanner
from src.ladder.planner import ExecutionResult

from .energy_prioritizer import EnergyBasedPrioritizer, PrioritizationResult
from .priority_engine import PriorityConfig


class EnergyEnhancedLadderPlanner(KGEnhancedLadderPlanner):
    """
    LADDER planner enhanced with energy-based task prioritization.

    This planner combines:
    - Knowledge Graph context (from KGEnhancedLadderPlanner)
    - Energy-based task prioritization
    - Intelligent task ordering based on success patterns
    - Resource-aware scheduling
    """

    def __init__(
        self,
        kg_interface: KnowledgeGraphInterface,
        priority_config: PriorityConfig | None = None,
        **kwargs,
    ):
        """Initialize the energy-enhanced planner.

        Args:
            kg_interface: Knowledge graph interface
            priority_config: Configuration for prioritization
            **kwargs: Arguments passed to parent planners
        """
        # Initialize parent with KG capabilities
        super().__init__(**kwargs)

        # Store KG interface
        self.kg_interface = kg_interface

        # Initialize energy-based prioritizer
        self.prioritizer = EnergyBasedPrioritizer(
            kg_interface=kg_interface,
            priority_config=priority_config,
        )

        # Enable energy prioritization in config
        self.config.enable_energy_prioritization = True

        # Track prioritization metrics
        self.prioritization_metrics = {
            "total_prioritizations": 0,
            "average_prioritization_time": 0.0,
            "energy_improvements": 0,
        }

    async def create_plan(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
        template=None,
    ) -> TaskGraph:
        """Create an energy-enhanced hierarchical plan."""
        context = context or {}

        # Create base plan with KG enhancement
        plan = await super().create_plan(goal, context, template)

        # Apply energy-based prioritization to the plan
        if plan.get_all_tasks():
            prioritization_result = self._prioritize_plan_tasks(plan, context)

            # Store prioritization info in plan metadata
            plan.metadata = plan.metadata or {}
            plan.metadata["energy_prioritization"] = {
                "total_tasks": prioritization_result.total_tasks,
                "executable_tasks": prioritization_result.executable_tasks,
                "blocked_tasks": prioritization_result.blocked_tasks,
                "average_energy": prioritization_result.average_energy,
                "average_confidence": prioritization_result.average_confidence,
                "strategy_used": prioritization_result.strategy_used,
                "calculation_time": prioritization_result.calculation_time,
            }

            # Add priority information to tasks
            self._annotate_tasks_with_priorities(plan, prioritization_result)

            # Log prioritization if debug mode
            if self.config.debug_mode:
                self._log_prioritization_results(prioritization_result)

        return plan

    def _prioritize_plan_tasks(
        self, plan: TaskGraph, context: dict[str, Any]
    ) -> PrioritizationResult:
        """Apply energy-based prioritization to plan tasks."""
        start_time = time.time()

        # Enhance context with domain information if available
        enhanced_context = context.copy()
        if hasattr(self, "execution_context"):
            domain = self.execution_context.get_fact("domain")
            if domain:
                enhanced_context["domain"] = domain

        # Calculate priorities
        result = self.prioritizer.prioritize_tasks(plan, enhanced_context)

        # Update metrics
        calculation_time = time.time() - start_time
        self.prioritization_metrics["total_prioritizations"] += 1

        # Update average calculation time
        current_avg = self.prioritization_metrics["average_prioritization_time"]
        total_count = self.prioritization_metrics["total_prioritizations"]
        new_avg = (current_avg * (total_count - 1) + calculation_time) / total_count
        self.prioritization_metrics["average_prioritization_time"] = new_avg

        return result

    def _annotate_tasks_with_priorities(
        self, plan: TaskGraph, prioritization_result: PrioritizationResult
    ):
        """Add priority information to tasks in the plan."""
        priority_map = {p.task_id: p for p in prioritization_result.priorities}

        for task in plan.get_all_tasks():
            if task.id in priority_map:
                priority_info = priority_map[task.id]

                # Add priority metadata to task
                if not hasattr(task, "metadata") or task.metadata is None:
                    task.metadata = {}

                task.metadata["energy_priority"] = {
                    "priority_score": priority_info.priority_score,
                    "rank": priority_info.rank,
                    "energy_score": priority_info.energy.energy_score,
                    "confidence": priority_info.energy.confidence,
                    "can_execute": priority_info.can_execute,
                    "blocked_by": priority_info.blocked_by,
                    "estimated_start_time": priority_info.estimated_start_time,
                }

                # Store detailed energy metrics for analysis
                task.metadata["energy_metrics"] = {
                    "effort_score": priority_info.energy.metrics.effort_score,
                    "success_probability": priority_info.energy.metrics.success_probability,
                    "complexity_score": priority_info.energy.metrics.complexity_score,
                    "pattern_confidence": priority_info.energy.metrics.pattern_confidence,
                    "dependency_score": priority_info.energy.metrics.dependency_score,
                    "recency_bonus": priority_info.energy.metrics.recency_bonus,
                    "context_relevance": priority_info.energy.metrics.context_relevance,
                }

    def _log_prioritization_results(self, result: PrioritizationResult):
        """Log prioritization results for debugging."""
        print("\n⚡ Energy-Based Prioritization Results")
        print("========================================")
        print(f"Total tasks: {result.total_tasks}")
        print(f"Executable: {result.executable_tasks}")
        print(f"Blocked: {result.blocked_tasks}")
        print(f"Average energy: {result.average_energy:.3f}")
        print(f"Average confidence: {result.average_confidence:.3f}")
        print(f"Strategy: {result.strategy_used}")
        print(f"Calculation time: {result.calculation_time:.3f}s")

        print("\nTop 5 Priority Tasks:")
        for i, priority in enumerate(result.priorities[:5]):
            status = "🟢 Ready" if priority.can_execute else "🔴 Blocked"
            print(
                f"  {i+1}. {priority.task_id[:8]}... "
                f"(score: {priority.priority_score:.3f}, "
                f"energy: {priority.energy.energy_score:.3f}) {status}"
            )

    async def execute_plan(
        self, plan: TaskGraph, context: dict[str, Any] | None = None
    ) -> ExecutionResult:
        """Execute plan with energy-aware task ordering."""
        context = context or {}

        # Get prioritized task execution order
        next_tasks = self.prioritizer.get_next_tasks(
            plan, count=self.prioritizer.priority_engine.config.max_parallel_tasks
        )

        if next_tasks and self.config.debug_mode:
            print("\n⚡ Next tasks to execute (energy-optimized):")
            for task_priority in next_tasks:
                print(
                    f"  - {task_priority.task_id} "
                    f"(priority: {task_priority.priority_score:.3f})"
                )

        # Execute with parent's execution logic
        result = await super().execute_plan(plan, context)

        # Learn from execution results
        await self._learn_from_execution(plan, result)

        return result

    async def _learn_from_execution(self, plan: TaskGraph, result: ExecutionResult):
        """Learn from execution results to improve future prioritization."""
        if not hasattr(result, "task_results"):
            return

        # Track which tasks completed successfully
        for task_id, task_result in result.task_results.items():
            if task_result.get("status") == "completed":
                # Mark task as completed in prioritizer
                completion_time = task_result.get("completion_time", time.time())
                self.prioritizer.mark_task_completed(task_id, completion_time)

                # Check if this task had better/worse energy prediction
                priority_info = self.prioritizer.get_task_priority_info(task_id)
                if priority_info:
                    predicted_energy = priority_info.energy.energy_score
                    actual_success = 1.0  # Task completed successfully

                    # Simple learning: if high energy task succeeded,
                    # it was over-penalized
                    if predicted_energy > 0.7 and actual_success > 0.8:
                        self.prioritization_metrics["energy_improvements"] += 1

    def get_prioritization_status(self) -> dict[str, Any]:
        """Get current prioritization status and metrics."""
        status = self.prioritizer.get_prioritization_summary()
        status["planner_metrics"] = self.prioritization_metrics

        return status

    def explain_task_priority(self, task_id: str) -> dict[str, Any]:
        """Get detailed explanation of task prioritization."""
        return self.prioritizer.explain_task_priority(task_id)

    def rebalance_priorities(
        self, plan: TaskGraph, context: dict[str, Any] | None = None
    ) -> bool:
        """Manually trigger priority rebalancing.

        Args:
            plan: Current task graph
            context: Planning context

        Returns:
            True if rebalancing was performed
        """
        if self.prioritizer._should_recalculate():
            self._prioritize_plan_tasks(plan, context or {})
            return True
        return False

    def get_next_recommended_tasks(
        self, plan: TaskGraph, count: int = 3
    ) -> list[dict[str, Any]]:
        """Get recommended next tasks with detailed information.

        Args:
            plan: Current task graph
            count: Number of recommendations

        Returns:
            List of task recommendations with priority details
        """
        next_tasks = self.prioritizer.get_next_tasks(plan, count)

        recommendations = []
        for priority in next_tasks:
            # Get the actual task object
            task = plan.get_task(priority.task_id)
            if not task:
                continue

            recommendation = {
                "task_id": priority.task_id,
                "task_title": getattr(task, "title", "Unknown"),
                "task_description": getattr(task, "description", ""),
                "priority_score": priority.priority_score,
                "rank": priority.rank,
                "energy_score": priority.energy.energy_score,
                "confidence": priority.energy.confidence,
                "can_execute": priority.can_execute,
                "blocked_by": priority.blocked_by,
                "estimated_start_time": priority.estimated_start_time,
                "why_recommended": self._generate_recommendation_reasoning(priority),
            }
            recommendations.append(recommendation)

        return recommendations

    def _generate_recommendation_reasoning(self, priority) -> str:
        """Generate human-readable reasoning for task recommendation."""
        reasons = []

        # Priority-based reasoning
        if priority.priority_score > 0.8:
            reasons.append("Very high priority task")
        elif priority.priority_score > 0.6:
            reasons.append("High priority task")
        else:
            reasons.append("Medium priority task")

        # Energy-based reasoning
        if priority.energy.energy_score < 0.3:
            reasons.append("low energy cost")
        elif priority.energy.energy_score < 0.6:
            reasons.append("moderate energy cost")
        else:
            reasons.append("high energy cost")

        # Success probability reasoning
        success_prob = priority.energy.metrics.success_probability
        if success_prob > 0.7:
            reasons.append("high success probability")
        elif success_prob > 0.4:
            reasons.append("moderate success probability")
        else:
            reasons.append("uncertain success probability")

        # Execution readiness
        if priority.can_execute:
            reasons.append("ready to execute immediately")
        else:
            reasons.append(f"blocked by {len(priority.blocked_by)} dependencies")

        return "; ".join(reasons)
