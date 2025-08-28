"""
Strategic Planner Plugin

This plugin implements the Strategic Layer of the three-layer OaK reasoning architecture.
It receives high-level user goals and decomposes them into manageable sub-goals,
emitting subgoal_defined events for the OaK Tactical Layer to process.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from uuid import uuid4

from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class StrategicPlannerPlugin(PluginInterface):
    """
    Strategic layer that decomposes user goals into actionable sub-goals.
    
    This plugin listens for goal_received events and breaks them down into
    smaller, more manageable sub-goals that can be handled by the OaK tactical layer.
    """

    @property
    def name(self) -> str:
        return "strategic_planner"

    def __init__(self) -> None:
        super().__init__()
        self.goal_decomposition_patterns = [
            # Simple task patterns that can be decomposed
            (r"analyze.*and.*", ["analyze_primary_component", "analyze_secondary_component", "synthesize_results"]),
            (r"create.*with.*", ["understand_requirements", "design_solution", "implement_solution"]),
            (r"find.*information.*about.*", ["search_primary_sources", "gather_relevant_data", "summarize_findings"]),
            (r"help.*with.*", ["understand_problem", "identify_solution_approach", "provide_assistance"]),
            (r"write.*code.*", ["understand_requirements", "design_architecture", "implement_code", "test_solution"]),
            (r"research.*", ["identify_research_scope", "gather_information", "analyze_findings"]),
        ]

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        """Initialize the strategic planner with event bus and configuration."""
        await super().setup(event_bus, store, config)
        self.max_subgoals = self.get_config("max_subgoals", 3)
        self.min_goal_length = self.get_config("min_goal_length", 10)

    async def start(self) -> None:
        """Start the strategic planner and subscribe to goal events."""
        await super().start()
        await self.subscribe("goal_received", self._handle_goal_received)
        await self.subscribe("TaskStarted", self._handle_task_started)  # Also listen to TaskStarted events
        self.log("info", "Strategic planner started and listening for goals")

    async def shutdown(self) -> None:
        """Clean up strategic planner resources."""
        self.log("info", "Strategic planner shutting down")
        await super().shutdown()

    async def _handle_goal_received(self, event: Any) -> None:
        """Handle incoming goal_received events by decomposing them into sub-goals."""
        try:
            goal = self._extract_goal_from_event(event)
            if not goal or len(goal) < self.min_goal_length:
                self.log("debug", f"Goal too short or empty, skipping decomposition: {goal}")
                return

            session_id = self._extract_session_id_from_event(event)
            parent_goal_id = self._extract_goal_id_from_event(event)
            
            sub_goals = self._decompose_goal(goal)
            
            self.log("info", f"Decomposed goal into {len(sub_goals)} sub-goals: {goal[:50]}...")
            
            # Emit subgoal_defined events for each sub-goal
            for i, sub_goal in enumerate(sub_goals):
                await self._emit_subgoal(sub_goal, parent_goal_id, session_id, i)
                
        except Exception as e:
            self.log("error", f"Failed to handle goal_received event: {e}")

    async def _handle_task_started(self, event: Any) -> None:
        """Handle TaskStarted events as an alternative entry point."""
        try:
            # Extract goal from TaskStarted event
            if hasattr(event, 'goal'):
                goal = event.goal
            elif isinstance(event, dict) and 'goal' in event:
                goal = event['goal']
            else:
                return  # No goal to process
                
            if not goal or len(goal) < self.min_goal_length:
                return
                
            correlation_id = getattr(event, 'correlation_id', None) or (event.get('correlation_id') if isinstance(event, dict) else None)
            session_id = correlation_id.split('-')[0] if correlation_id and '-' in correlation_id else 'default'
            parent_goal_id = f"goal_{correlation_id}" if correlation_id else f"goal_{uuid4()}"
            
            sub_goals = self._decompose_goal(goal)
            
            if len(sub_goals) > 1:  # Only decompose if we get multiple sub-goals
                self.log("info", f"Decomposed TaskStarted goal into {len(sub_goals)} sub-goals")
                
                for i, sub_goal in enumerate(sub_goals):
                    await self._emit_subgoal(sub_goal, parent_goal_id, session_id, i)
                    
        except Exception as e:
            self.log("error", f"Failed to handle TaskStarted event: {e}")

    def _extract_goal_from_event(self, event: Any) -> str | None:
        """Extract goal description from event object."""
        if hasattr(event, 'goal'):
            return event.goal
        elif hasattr(event, 'description'):
            return event.description
        elif isinstance(event, dict):
            return event.get('goal') or event.get('description') or event.get('message')
        return None

    def _extract_session_id_from_event(self, event: Any) -> str:
        """Extract session ID from event object."""
        if hasattr(event, 'session_id'):
            return event.session_id
        elif isinstance(event, dict):
            return event.get('session_id', 'default')
        return 'default'

    def _extract_goal_id_from_event(self, event: Any) -> str:
        """Extract or generate goal ID from event object."""
        if hasattr(event, 'goal_id'):
            return event.goal_id
        elif isinstance(event, dict) and 'goal_id' in event:
            return event['goal_id']
        else:
            return f"goal_{uuid4()}"

    def _decompose_goal(self, goal: str) -> list[str]:
        """
        Decompose a high-level goal into actionable sub-goals.
        
        Uses pattern matching and heuristics to break down complex goals.
        """
        goal_lower = goal.lower().strip()
        
        # Check against known patterns
        for pattern, sub_goal_templates in self.goal_decomposition_patterns:
            if re.search(pattern, goal_lower):
                return self._generate_subgoals_from_template(goal, sub_goal_templates)
        
        # Default decomposition for unknown patterns
        return self._default_decomposition(goal)

    def _generate_subgoals_from_template(self, original_goal: str, templates: list[str]) -> list[str]:
        """Generate sub-goals from templates, contextualizing them to the original goal."""
        sub_goals = []
        
        # Extract key entities from the original goal for contextualization
        entities = self._extract_key_entities(original_goal)
        
        for template in templates[:self.max_subgoals]:
            # Simple template substitution
            sub_goal = template.replace("_", " ").title()
            
            # Add context from original goal
            if entities:
                sub_goal = f"{sub_goal} for: {entities[0]}"
            
            sub_goals.append(sub_goal)
            
        return sub_goals

    def _extract_key_entities(self, goal: str) -> list[str]:
        """Extract key entities/subjects from the goal text."""
        # Simple heuristic: look for words after common prepositions
        import re
        
        patterns = [
            r"about\s+(\w+(?:\s+\w+)*)",
            r"for\s+(\w+(?:\s+\w+)*)",
            r"with\s+(\w+(?:\s+\w+)*)",
            r"on\s+(\w+(?:\s+\w+)*)",
            r"(?:analyze|research|find|create|write|help)\s+(\w+(?:\s+\w+)*)",
        ]
        
        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, goal.lower())
            entities.extend(matches)
            
        # Return first few entities, clean them up
        return [entity.strip() for entity in entities[:2] if entity.strip()]

    def _default_decomposition(self, goal: str) -> list[str]:
        """
        Default decomposition strategy for goals that don't match known patterns.
        
        Breaks goals based on common task phases.
        """
        # For very short goals, don't decompose
        if len(goal.split()) < 4:
            return [goal]
            
        # Generic decomposition
        return [
            f"Understand the requirements: {goal}",
            f"Execute the main task: {goal}",
            f"Verify and finalize: {goal}"
        ]

    async def _emit_subgoal(self, sub_goal_description: str, parent_goal_id: str, session_id: str, index: int) -> None:
        """Emit a subgoal_defined event for the OaK tactical layer."""
        subgoal_id = f"subgoal_{parent_goal_id}_{index}"
        
        await self.emit_event(
            "subgoal_defined",
            subgoal={
                "description": sub_goal_description,
                "parent_goal_id": parent_goal_id,
                "subgoal_id": subgoal_id,
                "index": index,
            },
            session_id=session_id,
            source_plugin=self.name,
        )
        
        self.log("debug", f"Emitted subgoal {index}: {sub_goal_description}")

    async def health_check(self) -> dict[str, Any]:
        """Return health status for the strategic planner."""
        base_health = await super().health_check()
        base_health.update({
            "max_subgoals": self.max_subgoals,
            "min_goal_length": self.min_goal_length,
            "decomposition_patterns": len(self.goal_decomposition_patterns),
        })
        return base_health