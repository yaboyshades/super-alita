#!/usr/bin/env python3
"""
V2 Planner Plugin for Super Alita
This plugin acts as the Strategic Layer in the OaK architecture.
It decomposes high-level goals into sub-goals for the Tactical Layer.
"""

import logging
from typing import Any

from src.core.events import Subgoal, SubgoalDefinedEvent
from src.core.plan_executor import PlanExecutor
from src.core.plugin_interface import PluginInterface
from src.core.prompt_manager import get_prompt_manager

logger = logging.getLogger(__name__)


class PlannerPluginV2(PluginInterface):
    """
    V2 Plugin that converts user goals into sub-goals for the OaK Tactical Layer.
    """

    def __init__(self):
        super().__init__()
        self.active_plans = {}
        self.plan_counter = 0
        self.executor = None  # Will be initialized in setup
        self._handled_tool_calls: set[str] = set()  # Added deduplication cache
        self.gemini_client = None  # Will be initialized in setup
        self.prompt_manager = get_prompt_manager()  # Initialize prompt manager

    @property
    def name(self) -> str:
        return "planner_v2"

    async def setup(self, event_bus, store, config: dict[str, Any]) -> None:
        """Initialize the planner plugin."""
        await super().setup(event_bus, store, config)

        # Initialize closed-loop executor
        self.executor = PlanExecutor(event_bus, store)

        # Initialize Gemini client for intent detection
        try:
            from src.core.gemini_pilot import GeminiPilotClient

            self.gemini_client = GeminiPilotClient()
            logger.info(
                "Gemini client initialized for natural language intent detection"
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize Gemini client: {e} - falling back to pattern matching"
            )

        logger.info("PlannerPluginV2 setup complete.")

    async def start(self) -> None:
        """Start the planner plugin."""
        await super().start()

        # Subscribe to goal events from conversation plugin
        await self.subscribe("goal_received", self._create_plan)
        # The rest of the subscriptions are for the old planner logic, we can remove them
        # if they are not needed for the V2 planner. For now, I will leave them commented out.
        # await self.subscribe("tool_result", self._handle_tool_result)
        # await self.subscribe("user_message", self._handle_user_message)
        # await self.subscribe("atom_ready", self._handle_atom_ready)

        logger.info("PlannerPluginV2 started - ready to create plans from goals.")

    async def shutdown(self) -> None:
        """Shutdown the planner plugin."""
        # Cancel any active plans
        for plan_id in list(self.active_plans.keys()):
            plan = self.active_plans[plan_id]
            if plan["status"] == "executing":
                plan["status"] = "cancelled"

        logger.info(
            f"PlannerPluginV2 shutdown - cancelled {len(self.active_plans)} active plans"
        )
        self.active_plans.clear()

    async def _create_plan(self, event):
        """
        Takes a goal and emits a subgoal_defined event for the OaK Tactical Layer.
        This is the new primary function of the planner in the OaK architecture.
        """
        try:
            goal_description = event.goal
            session_id = event.session_id
            parent_goal_id = getattr(event, "parent_goal_id", f"goal_{session_id}")
            subgoal_id = f"subgoal_{session_id}_{self.plan_counter}"
            self.plan_counter += 1

            logger.info(f"🎯 Decomposing goal into subgoal: {goal_description}")

            subgoal = Subgoal(
                description=goal_description,
                parent_goal_id=parent_goal_id,
                subgoal_id=subgoal_id,
            )

            await self.event_bus.publish(
                SubgoalDefinedEvent(
                    source_plugin=self.name,
                    subgoal=subgoal,
                    session_id=session_id,
                )
            )

            logger.info(f"✅ Emitted subgoal_defined event: {subgoal_id}")

        except Exception as e:
            logger.error(f"Failed to emit subgoal_defined event: {e}", exc_info=True)
            await self.emit_event(
                "agent_reply",
                text=f"❌ Failed to process goal: {e!s}",
                session_id=event.session_id,
            )

    # The following methods are part of the old planner's logic and are not needed
    # for the V2 planner which only emits subgoals. I am removing them to keep the
    # new plugin clean.

    # async def _handle_user_message(self, event): ...
    # async def _handle_tool_gap(self, gap_description: str, ...): ...
    # async def _emit_chat_response(self, text: str, ...): ...
    # async def _emit_tool_call(self, tool_name: str, ...): ...
    # async def _handle_tool_result(self, event): ...
    # async def _handle_atom_ready(self, event): ...
    # async def _get_dynamic_tools(self) -> dict[str, str]: ...
    # async def get_status(self) -> dict[str, Any]: ...

    async def get_status(self) -> dict[str, Any]:
        """Get plugin status."""
        base_status = {
            "active_plans": len(self.active_plans),
            "total_subgoals_created": self.plan_counter,
            "planning_active": True,
        }
        return base_status
