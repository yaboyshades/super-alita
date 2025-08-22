"""
MCP Telemetry Broadcaster - Real-time agent event streaming to MCP server.

This module broadcasts agent events and telemetry data to the MCP server
for real-time monitoring and debugging through Copilot Chat.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TelemetryEvent:
    """Structured telemetry event for MCP broadcasting."""

    timestamp: float
    event_type: str
    source: str
    data: dict[str, Any]
    session_id: str | None = None
    conversation_id: str | None = None
    metadata: dict[str, Any] | None = None


class MCPTelemetryBroadcaster:
    """
    Broadcasts agent telemetry to MCP server for real-time monitoring.

    This component:
    1. Captures agent events and metrics
    2. Formats them for MCP transmission
    3. Provides real-time streaming to Copilot Chat
    4. Maintains event history for analysis
    """

    def __init__(self, max_events: int = 1000):
        self.max_events = max_events
        self.events: list[TelemetryEvent] = []
        self.event_counts: dict[str, int] = {}
        self.start_time = time.time()
        self.is_active = False

        # Event filtering - what to broadcast
        self.broadcast_event_types = {
            "tool_call",
            "tool_result",
            "conversation",
            "cognitive_turn",
            "memory_operation",
            "plugin_startup",
            "plugin_error",
            "validation_result",
            "performance_metric",
        }

    async def start(self):
        """Start the telemetry broadcaster."""
        self.is_active = True
        self.start_time = time.time()
        logger.info("🔄 MCP Telemetry Broadcaster started")

        # Broadcast startup event
        await self.broadcast_event(
            event_type="broadcaster_startup",
            source="mcp_telemetry_broadcaster",
            data={"status": "active", "start_time": self.start_time},
        )

    async def stop(self):
        """Stop the telemetry broadcaster."""
        self.is_active = False
        logger.info("🛑 MCP Telemetry Broadcaster stopped")

        # Broadcast shutdown event
        await self.broadcast_event(
            event_type="broadcaster_shutdown",
            source="mcp_telemetry_broadcaster",
            data={"status": "inactive", "runtime": time.time() - self.start_time},
        )

    async def broadcast_event(
        self,
        event_type: str,
        source: str,
        data: dict[str, Any],
        session_id: str | None = None,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Broadcast a telemetry event to MCP server.

        Args:
            event_type: Type of event (tool_call, conversation, etc.)
            source: Source component/plugin name
            data: Event data payload
            session_id: Optional session identifier
            conversation_id: Optional conversation identifier
            metadata: Optional additional metadata
        """
        if not self.is_active:
            return

        try:
            # Create telemetry event
            event = TelemetryEvent(
                timestamp=time.time(),
                event_type=event_type,
                source=source,
                data=data,
                session_id=session_id,
                conversation_id=conversation_id,
                metadata=metadata or {},
            )

            # Add to event history
            self.events.append(event)
            self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1

            # Maintain max events limit
            if len(self.events) > self.max_events:
                self.events.pop(0)

            # Filter events for broadcasting
            if event_type in self.broadcast_event_types:
                await self._send_to_mcp(event)

            logger.debug(f"📡 Broadcasted {event_type} event from {source}")

        except Exception as e:
            logger.error(f"Failed to broadcast telemetry event: {e}")

    async def _send_to_mcp(self, event: TelemetryEvent):
        """Send event to MCP server (placeholder for actual implementation)."""
        try:
            # Format event for MCP transmission
            mcp_data = {
                "type": "telemetry_event",
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "source": event.source,
                "data": event.data,
                "session_id": event.session_id,
                "conversation_id": event.conversation_id,
                "metadata": event.metadata,
            }

            # TODO: Implement actual MCP transmission
            # For now, log to stderr so MCP server can capture it
            print(f"TELEMETRY: {json.dumps(mcp_data)}", flush=True)

        except Exception as e:
            logger.error(f"Failed to send event to MCP: {e}")

    def get_telemetry_summary(self) -> dict[str, Any]:
        """Get comprehensive telemetry summary."""
        runtime = time.time() - self.start_time

        return {
            "broadcaster_status": "active" if self.is_active else "inactive",
            "runtime_seconds": runtime,
            "total_events": len(self.events),
            "event_counts_by_type": dict(self.event_counts),
            "events_per_second": len(self.events) / runtime if runtime > 0 else 0,
            "recent_events": [
                {
                    "timestamp": event.timestamp,
                    "type": event.event_type,
                    "source": event.source,
                    "has_data": bool(event.data),
                }
                for event in self.events[-10:]  # Last 10 events
            ],
        }

    def get_event_history(
        self, event_type: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get filtered event history."""
        events = self.events

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Return recent events first
        events = events[-limit:]
        events.reverse()

        return [asdict(event) for event in events]


# Global broadcaster instance
_global_broadcaster: MCPTelemetryBroadcaster | None = None


def get_broadcaster() -> MCPTelemetryBroadcaster:
    """Get the global telemetry broadcaster instance."""
    global _global_broadcaster
    if _global_broadcaster is None:
        _global_broadcaster = MCPTelemetryBroadcaster()
    return _global_broadcaster


async def broadcast_agent_event(
    event_type: str,
    source: str,
    data: dict[str, Any],
    session_id: str | None = None,
    conversation_id: str | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Convenience function to broadcast agent events."""
    broadcaster = get_broadcaster()
    await broadcaster.broadcast_event(
        event_type=event_type,
        source=source,
        data=data,
        session_id=session_id,
        conversation_id=conversation_id,
        metadata=metadata,
    )


# Event types for easy reference
class EventTypes:
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CONVERSATION = "conversation"
    COGNITIVE_TURN = "cognitive_turn"
    MEMORY_OPERATION = "memory_operation"
    PLUGIN_STARTUP = "plugin_startup"
    PLUGIN_ERROR = "plugin_error"
    VALIDATION_RESULT = "validation_result"
    PERFORMANCE_METRIC = "performance_metric"
