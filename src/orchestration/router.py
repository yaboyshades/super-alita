"""Router for parsing planner output and determining action type."""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class ActionRoute:
    """Represents a parsed action from planner output."""

    def __init__(
        self,
        action_type: str,
        target: str | None = None,
        params: dict[str, Any] | None = None,
    ):
        self.action_type = action_type  # "GAP", "NONE", or tool name
        self.target = target  # Tool name for tool actions
        self.params = params or {}

    def __repr__(self):
        if self.action_type == "GAP":
            return (
                f"ActionRoute(GAP -> {self.params.get('description', 'Unknown gap')})"
            )
        if self.action_type == "NONE":
            return f"ActionRoute(NONE -> {self.params.get('response', 'No action')})"
        return f"ActionRoute(TOOL -> {self.target})"


class Router:
    """Parse planner output and route to appropriate action."""

    def __init__(self):
        # Patterns to parse planner output
        self.gap_pattern = re.compile(r"GAP\s+(.+)", re.IGNORECASE)
        self.none_pattern = re.compile(r"NONE\s+(.+)", re.IGNORECASE)
        self.tool_pattern = re.compile(r"TOOL\s+(\w+)(?:\s+(.+))?", re.IGNORECASE)
        self.sot_executed_pattern = re.compile(r"SOT_EXECUTED\s+(.+)", re.IGNORECASE)

        # Pattern for "show me you created it" requests
        self.show_created_pattern = re.compile(
            r"(?:show|prove|display|demonstrate).+(?:created?|made|built|generated).+tool",
            re.IGNORECASE,
        )

    def parse_planner_output(self, planner_output: str) -> ActionRoute:
        """Parse planner output string and return ActionRoute."""
        planner_output = planner_output.strip()

        logger.debug(f"Parsing planner output: {planner_output}")

        # Check for SOT_EXECUTED action (SoT has already processed the request)
        sot_match = self.sot_executed_pattern.match(planner_output)
        if sot_match:
            response = sot_match.group(1).strip()
            logger.info(f"Parsed SOT_EXECUTED action: {response}")
            return ActionRoute("SOT_EXECUTED", params={"response": response})

        # Check for GAP action
        gap_match = self.gap_pattern.match(planner_output)
        if gap_match:
            description = gap_match.group(1).strip()
            logger.info(f"Parsed GAP action: {description}")
            return ActionRoute("GAP", params={"description": description})

        # Check for NONE action
        none_match = self.none_pattern.match(planner_output)
        if none_match:
            response = none_match.group(1).strip()
            logger.info(f"Parsed NONE action: {response}")
            return ActionRoute("NONE", params={"response": response})

        # Check for TOOL action
        tool_match = self.tool_pattern.match(planner_output)
        if tool_match:
            tool_name = tool_match.group(1).strip()
            tool_params_str = tool_match.group(2)
            params = {}
            if tool_params_str:
                # Basic parameter parsing - can be enhanced later
                params = {"params_str": tool_params_str.strip()}
            logger.info(f"Parsed TOOL action: {tool_name} with params {params}")
            return ActionRoute("TOOL", target=tool_name, params=params)

        # Fallback - treat as NONE
        logger.warning(
            f"Could not parse planner output, treating as NONE: {planner_output}"
        )
        return ActionRoute("NONE", params={"response": planner_output})

    def route_user_message(self, user_message: str, planner_output: str) -> ActionRoute:
        """Main routing function: user message + planner output -> ActionRoute."""
        logger.info(f"Routing user message: {user_message}")

        # Check for "show me you created it" requests first
        if self.show_created_pattern.search(user_message):
            # Try to extract tool name from the message
            tool_name_pattern = re.compile(
                r"(?:tool|function)\s+['\"]?(\w+)['\"]?", re.IGNORECASE
            )
            tool_match = tool_name_pattern.search(user_message)
            tool_name = tool_match.group(1) if tool_match else "unknown"

            logger.info(f"Detected 'show created tool' request for: {tool_name}")
            return ActionRoute(
                "SHOW_CREATED", target=tool_name, params={"user_message": user_message}
            )

        # Parse the planner output
        route = self.parse_planner_output(planner_output)

        # Add original user message to context
        route.params["user_message"] = user_message

        logger.info(f"Routed to: {route}")
        return route
