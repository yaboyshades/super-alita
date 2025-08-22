"""Dispatcher for executing actions based on router output."""

import asyncio
import logging
from typing import Any

from src.core.events import (
    AgentReplyEvent,
    AtomGapEvent,
    ShowCreatedToolRequest,
    ToolCallEvent,
)

logger = logging.getLogger(__name__)


class Dispatcher:
    """Execute actions based on ActionRoute from router."""

    def __init__(
        self,
        event_bus,
        session_id: str | None = None,
        conversation_id: str | None = None,
    ):
        self.event_bus = event_bus
        self.session_id = session_id or "default_session"
        self.conversation_id = conversation_id or self.session_id

    async def dispatch_gap(self, description: str) -> None:
        """Dispatch GAP action - create new capability."""
        logger.info(f"Dispatching GAP: {description}")

        gap_event = AtomGapEvent(
            source_plugin="conversation_plugin",
            conversation_id=self.conversation_id,
            missing_tool="dynamic_tool",
            description=description,
            session_id=self.session_id,
            gap_id=f"gap_{asyncio.get_event_loop().time()}",
        )

        await self.event_bus.publish(gap_event)
        logger.info(f"Published AtomGapEvent for: {description}")

    async def dispatch_tool(self, tool_name: str, params: dict[str, Any]) -> None:
        """Dispatch TOOL action - call specific tool."""
        logger.info(f"Dispatching TOOL: {tool_name} with params {params}")

        tool_event = ToolCallEvent(
            source_plugin="conversation_plugin",
            conversation_id=self.session_id,
            session_id=self.session_id,
            tool_name=tool_name,
            parameters=params,
            tool_call_id=f"call_{asyncio.get_event_loop().time()}",
        )

        await self.event_bus.publish(tool_event)
        logger.info(f"Published ToolCallEvent for: {tool_name}")

    async def dispatch_show_created(self, tool_name: str, user_message: str) -> None:
        """Dispatch SHOW_CREATED action - show proof of tool creation."""
        logger.info(f"Dispatching SHOW_CREATED: {tool_name}")

        show_request = ShowCreatedToolRequest(
            source_plugin="conversation_plugin",
            tool_name=tool_name,
            session_id=self.session_id,
            conversation_id=self.conversation_id,
        )

        await self.event_bus.publish(show_request)
        logger.info(f"Published ShowCreatedToolRequest for: {tool_name}")

    async def dispatch_sot_executed(self, response: str) -> None:
        """Dispatch SOT_EXECUTED action - Script-of-Thought has already processed the request."""
        logger.info(f"Dispatching SOT_EXECUTED: {response}")

        response_event = AgentReplyEvent(
            source_plugin="conversation_plugin",
            text=response,  # Required field
            message=response,  # Backward compatibility
            session_id=self.session_id,
            conversation_id=self.conversation_id,
        )

        await self.event_bus.publish(response_event)
        logger.info(f"Published SoT execution result: {response}")

    async def dispatch_none(self, response: str) -> None:
        """Dispatch NONE action - direct response to user with structured event."""
        logger.info(f"Dispatching NONE: {response}")

        response_event = AgentReplyEvent(
            source_plugin="conversation_plugin",
            text=response,  # Required field
            message=response,  # Backward compatibility
            session_id=self.session_id,
            conversation_id=self.conversation_id,
        )

        await self.event_bus.publish(response_event)
        logger.info(f"Published structured agent reply: {response}")

    async def dispatch_action(self, action_route) -> None:
        """Main dispatch function - route ActionRoute to appropriate handler."""
        action_type = action_route.action_type
        params = action_route.params
        user_message = params.get("user_message", "")

        logger.info(f"Dispatching action: {action_route}")

        if action_type == "GAP":
            description = params.get("description", "Unknown capability gap")
            await self.dispatch_gap(description)

        elif action_type == "NONE":
            response = params.get(
                "response", "I understand, but I'm not sure how to help with that."
            )
            await self.dispatch_none(response)

        elif action_type == "SOT_EXECUTED":
            response = params.get(
                "response", "Task completed via Script-of-Thought execution."
            )
            await self.dispatch_sot_executed(response)

        elif action_type == "TOOL":
            tool_name = action_route.target
            tool_params = params.copy()
            # Remove metadata from tool params
            tool_params.pop("user_message", None)
            await self.dispatch_tool(tool_name, tool_params)

        elif action_type == "SHOW_CREATED":
            tool_name = action_route.target
            await self.dispatch_show_created(tool_name, user_message)

        else:
            logger.error(f"Unknown action type: {action_type}")
            await self.dispatch_none(
                f"I'm not sure how to handle that request: {user_message}"
            )
