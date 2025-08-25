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

__all__ = [
    "EventTypes",
    "MCPTelemetryBroadcaster",
    "TelemetryEvent",
    "broadcast_agent_event",
    "get_broadcaster",
    "wrap_plugin_for_telemetry",
]
