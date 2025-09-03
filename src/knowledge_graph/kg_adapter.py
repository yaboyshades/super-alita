"""Knowledge Graph EventBus adapter for learning from planning events."""

import time
from typing import Any

from src.core.event_bus import EventBus
from src.core.events import BaseEvent

from .kg_interface import KnowledgeGraphInterface
from .models import (
    EntityType,
    KnowledgeEntity,
    KnowledgeRelation,
    PlanningContext,
    RelationType,
)


class KnowledgeGraphAdapter:
    """
    EventBus adapter that learns from planning events and updates the KG.

    This adapter:
    - Listens to planning events from LADDER
    - Extracts knowledge from planning outcomes
    - Updates the KG with new entities, relations, and patterns
    - Provides KG context for future planning sessions
    """

    def __init__(
        self,
        kg_interface: KnowledgeGraphInterface,
        event_bus: EventBus,
        source_plugin: str = "kg_adapter",
    ):
        """Initialize the KG adapter.

        Args:
            kg_interface: Knowledge graph interface
            event_bus: Event bus for system communication
            source_plugin: Plugin identifier for events
        """
        self.kg = kg_interface
        self.event_bus = event_bus
        self.source_plugin = source_plugin
        self._initialized = False
        self._active_sessions: dict[str, dict[str, Any]] = {}

    async def setup(self) -> None:
        """Setup event subscriptions."""
        if self._initialized:
            return

        # Subscribe to planning events
        await self.event_bus.subscribe(
            "planning_started", self._handle_planning_started
        )
        await self.event_bus.subscribe(
            "planning_completed", self._handle_planning_completed
        )
        await self.event_bus.subscribe("planning_error", self._handle_planning_error)

        # Emit KG ready event
        await self.event_bus.emit(
            "kg_initialized",
            source_plugin=self.source_plugin,
            statistics=self.kg.get_statistics(),
        )

        self._initialized = True

    async def _handle_planning_started(self, event: BaseEvent) -> None:
        """Handle planning started events."""
        event_data = event.model_dump() if hasattr(event, "model_dump") else {}

        session_id = event_data.get("session_id", "")
        goal = event_data.get("goal", "")
        context = event_data.get("context", {})

        if session_id and goal:
            # Store session info for learning
            self._active_sessions[session_id] = {
                "goal": goal,
                "context": context,
                "start_time": time.time(),
                "domain": self._extract_domain(goal, context),
            }

            # Create goal entity if not exists
            goal_entity = KnowledgeEntity(
                entity_type=EntityType.GOAL,
                name=goal,
                description=f"Planning goal: {goal}",
                properties={
                    "domain": self._extract_domain(goal, context),
                    "context": context,
                },
            )
            self.kg.add_entity(goal_entity)

    async def _handle_planning_completed(self, event: BaseEvent) -> None:
        """Handle planning completed events."""
        event_data = event.model_dump() if hasattr(event, "model_dump") else {}

        session_id = event_data.get("session_id", "")
        success = event_data.get("success", False)
        execution_time = event_data.get("execution_time", 0.0)
        tasks_completed = event_data.get("tasks_completed", 0)

        if session_id in self._active_sessions:
            session_info = self._active_sessions[session_id]

            # Create planning context for learning
            planning_context = PlanningContext(
                session_id=session_id,
                goal=session_info["goal"],
                domain=session_info["domain"],
                user_context=session_info["context"],
                success=success,
                execution_time=execution_time,
                task_count=tasks_completed,
            )

            # Add to KG for learning
            self.kg.add_context(planning_context)

            # Create outcome entity
            outcome_entity = KnowledgeEntity(
                entity_type=EntityType.OUTCOME,
                name=f"Planning outcome: {'success' if success else 'failure'}",
                description=f"Result of planning session {session_id}",
                properties={
                    "success": success,
                    "execution_time": execution_time,
                    "task_count": tasks_completed,
                    "session_id": session_id,
                },
            )
            outcome_entity_id = self.kg.add_entity(outcome_entity)

            # Find goal entity and create relation
            goal_entities = [
                e
                for e in self.kg.entities.values()
                if e.entity_type == EntityType.GOAL and e.name == session_info["goal"]
            ]

            if goal_entities:
                goal_entity = goal_entities[0]  # Take most recent
                relation = KnowledgeRelation(
                    source_entity_id=goal_entity.id,
                    target_entity_id=outcome_entity_id,
                    relation_type=(
                        RelationType.SUCCEEDED_BY
                        if success
                        else RelationType.FAILED_WITH
                    ),
                    strength=1.0,
                    context={"session_id": session_id},
                )
                self.kg.add_relation(relation)

            # Emit KG learning event
            await self.event_bus.emit(
                "kg_learned",
                source_plugin=self.source_plugin,
                session_id=session_id,
                success=success,
                domain=session_info["domain"],
                statistics=self.kg.get_statistics(),
            )

            # Clean up session
            del self._active_sessions[session_id]

    async def _handle_planning_error(self, event: BaseEvent) -> None:
        """Handle planning error events."""
        event_data = event.model_dump() if hasattr(event, "model_dump") else {}

        session_id = event_data.get("session_id", "")
        error = event_data.get("error", "")

        if session_id in self._active_sessions:
            session_info = self._active_sessions[session_id]

            # Create error context
            planning_context = PlanningContext(
                session_id=session_id,
                goal=session_info["goal"],
                domain=session_info["domain"],
                user_context=session_info["context"],
                success=False,
                execution_time=0.0,
                task_count=0,
            )

            # Add to KG for learning
            self.kg.add_context(planning_context)

            # Create error entity
            error_entity = KnowledgeEntity(
                entity_type=EntityType.OUTCOME,
                name=f"Planning error: {error[:50]}...",
                description=f"Error in planning session {session_id}: {error}",
                properties={
                    "error": error,
                    "session_id": session_id,
                    "success": False,
                },
            )
            self.kg.add_entity(error_entity)

            # Clean up session
            del self._active_sessions[session_id]

    def _extract_domain(self, goal: str, context: dict[str, Any]) -> str:
        """Extract domain from goal and context."""
        goal_lower = goal.lower()

        # Check for common domains
        if any(
            word in goal_lower
            for word in ["code", "function", "class", "program", "software"]
        ):
            return "software_development"
        elif any(
            word in goal_lower
            for word in ["research", "study", "investigate", "analyze"]
        ):
            return "research"
        elif any(
            word in goal_lower for word in ["write", "document", "report", "article"]
        ):
            return "documentation"
        elif any(
            word in goal_lower for word in ["test", "debug", "fix", "troubleshoot"]
        ):
            return "testing_debugging"
        elif any(
            word in goal_lower
            for word in ["design", "architecture", "plan", "blueprint"]
        ):
            return "design"

        # Check context for domain hints
        if "language" in context:
            return "software_development"
        elif "domain" in context:
            return context["domain"]

        return "general"

    def get_planning_context(
        self, goal: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Get relevant planning context from the KG."""
        from .models import EntityType, KnowledgeQuery

        # Extract domain
        domain = self._extract_domain(goal, context)

        # Query KG for relevant knowledge
        query = KnowledgeQuery(
            goal=goal,
            domain=domain,
            context=context,
            entity_types={EntityType.PATTERN, EntityType.GOAL, EntityType.OUTCOME},
            max_results=5,
            include_patterns=True,
        )

        result = self.kg.query(query)

        # Format context for LADDER
        planning_context = {
            "domain": domain,
            "relevant_patterns": [
                {
                    "name": pattern.pattern_name,
                    "steps": pattern.decomposition_steps,
                    "success_rate": pattern.success_rate,
                    "complexity": pattern.complexity_level,
                    "tools": pattern.required_tools,
                }
                for pattern in result.patterns[:3]  # Top 3 patterns
            ],
            "similar_goals": [
                {
                    "name": entity.name,
                    "description": entity.description,
                    "confidence": entity.confidence,
                }
                for entity in result.entities
                if entity.entity_type == EntityType.GOAL
            ][
                :3
            ],  # Top 3 similar goals
            "historical_outcomes": [
                {
                    "description": entity.description,
                    "success": entity.properties.get("success", False),
                    "execution_time": entity.properties.get("execution_time", 0.0),
                }
                for entity in result.entities
                if entity.entity_type == EntityType.OUTCOME
            ][
                :3
            ],  # Top 3 outcomes
            "kg_statistics": self.kg.get_statistics(),
        }

        return planning_context
