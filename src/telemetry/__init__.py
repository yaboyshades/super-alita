"""
Super Alita Telemetry Module

This module provides real-time telemetry broadcasting to MCP servers
for monitoring and debugging agent behavior through Copilot Chat.
"""

from .mcp_broadcaster import (
    EventTypes,
    MCPTelemetryBroadcaster,
    TelemetryEvent,
    broadcast_agent_event,
    get_broadcaster,
)
from .plugin_wrapper import wrap_plugin_for_telemetry

# ---------------------------------------------------------------------------
# Copilot context helpers
# ---------------------------------------------------------------------------


def build_copilot_context(user_message: str, session_id: str) -> str:
    """Construct a lightweight context string for Copilot prompts.

    The function intentionally keeps the structure simple – the goal is to
    surface user/session details into telemetry without imposing a specific
    prompt format.  Callers may prepend this context to system prompts or use it
    in event metadata.

    Args:
        user_message: Raw user message for the current turn.
        session_id: Identifier for the active conversation/session.

    Returns:
        A single string embedding the session and user message.
    """

    return f"session={session_id} user={user_message}"


__all__ = [
    "EventTypes",
    "MCPTelemetryBroadcaster",
    "TelemetryEvent",
    "broadcast_agent_event",
    "get_broadcaster",
    "wrap_plugin_for_telemetry",
    "build_copilot_context",
]
