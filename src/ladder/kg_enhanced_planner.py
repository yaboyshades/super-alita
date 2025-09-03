"""KG-enhanced LADDER planner with knowledge graph integration."""

import time
from typing import Any

from src.knowledge_graph import KnowledgeGraphAdapter

from .graph.task_graph import TaskGraph
from .planner import ExecutionResult, LadderPlanner


class KGEnhancedLadderPlanner(LadderPlanner):
    """
    Knowledge Graph enhanced LADDER planner.

    This planner extends the base LadderPlanner with KG integration:
    - Uses KG context for better task decomposition
    - Learns from planning patterns
    - Provides historical knowledge for similar goals
    """

    def __init__(self, kg_adapter: KnowledgeGraphAdapter | None = None, **kwargs):
        """Initialize the KG-enhanced planner.

        Args:
            kg_adapter: Knowledge graph adapter for context
            **kwargs: Arguments passed to base LadderPlanner
        """
        super().__init__(**kwargs)
        self.kg_adapter = kg_adapter

        # Enable KG in config if adapter provided
        if self.kg_adapter:
            self.config.enable_knowledge_graph = True

    async def create_plan(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
        template=None,
    ) -> TaskGraph:
        """Create a KG-enhanced hierarchical plan."""
        context = context or {}

        # Get KG context if available
        if self.kg_adapter:
            kg_context = self.kg_adapter.get_planning_context(goal, context)

            # Merge KG context with user context
            context["kg_context"] = kg_context

            # Add domain information to execution context
            if kg_context.get("domain"):
                self.execution_context.add_fact("domain", kg_context["domain"])

            # Add relevant patterns to execution context
            if kg_context.get("relevant_patterns"):
                self.execution_context.add_fact(
                    "relevant_patterns", kg_context["relevant_patterns"]
                )

                # Log pattern usage for debugging
                if self.config.debug_mode:
                    print(
                        f"🧠 KG found {len(kg_context['relevant_patterns'])} relevant patterns"
                    )
                    for pattern in kg_context["relevant_patterns"]:
                        print(
                            f"   - {pattern['name']} (success: {pattern['success_rate']:.1%})"
                        )

        # Create plan using enhanced context
        return await super().create_plan(goal, context, template)

    def _should_decompose_task(self, task) -> bool:
        """Enhanced decomposition decision using KG patterns."""
        base_decision = super()._should_decompose_task(task)

        # If base says no, stick with that
        if not base_decision or not self.kg_adapter:
            return base_decision

        # Check KG patterns for decomposition guidance
        relevant_patterns = self.execution_context.get_fact("relevant_patterns", [])

        for pattern in relevant_patterns:
            # If we have successful patterns with multiple steps, decompose
            if (
                pattern.get("success_rate", 0) > 0.5
                and len(pattern.get("steps", [])) > 1
            ):
                return True

        return base_decision

    async def execute_plan(
        self, plan: TaskGraph, context: dict[str, Any] | None = None
    ) -> ExecutionResult:
        """Execute a plan with KG learning."""
        context = context or {}
        start_time = time.time()

        # Execute the plan
        result = await super().execute_plan(plan, context)

        # Update execution time
        result.execution_time = time.time() - start_time

        # Learn from execution if KG is available
        if self.kg_adapter:
            await self._learn_from_execution(plan, result, context)

        return result

    async def _learn_from_execution(
        self, plan: TaskGraph, result: ExecutionResult, context: dict[str, Any]
    ) -> None:
        """Learn from execution results and update KG."""
        # This would be called by the EventBus integration
        # The actual learning happens in the KG adapter via events
        pass

    def get_enhanced_context(
        self, goal: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Get enhanced planning context with KG knowledge."""
        if not self.kg_adapter:
            return context

        kg_context = self.kg_adapter.get_planning_context(goal, context)

        enhanced_context = {
            **context,
            "kg_enhanced": True,
            "domain": kg_context.get("domain", "general"),
            "patterns_found": len(kg_context.get("relevant_patterns", [])),
            "similar_goals_found": len(kg_context.get("similar_goals", [])),
            "historical_outcomes": len(kg_context.get("historical_outcomes", [])),
        }

        return enhanced_context

    def get_kg_statistics(self) -> dict[str, Any]:
        """Get knowledge graph statistics."""
        if not self.kg_adapter:
            return {"kg_enabled": False}

        return {"kg_enabled": True, **self.kg_adapter.kg.get_statistics()}
