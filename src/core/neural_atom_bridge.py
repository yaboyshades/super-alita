#!/usr/bin/env python3
"""
Bridge between EventBus and Neural Atom system
"""

import json
import logging

from ..neural.atom import Atom
from ..neural.mcp_server import MCPServer

logger = logging.getLogger(__name__)


class NeuralAtomBridge:
    """Bridge between EventBus and Neural Atom system"""

    def __init__(self, mcp_server: MCPServer):
        self.mcp_server = mcp_server
        self.event_mappings = {
            "user_message": self._map_user_message,
            "tool_created": self._map_tool_created,
            "sot_executed": self._map_sot_executed,
            "state_transition": self._map_state_transition,
            "tool_call": self._map_tool_call,
            "tool_response": self._map_tool_response,
        }
        logger.info("Neural Atom Bridge initialized")

    async def handle_event(self, event):
        """Convert EventBus event to Neural Atom and process"""
        try:
            # Check if we have a mapping for this event type
            mapper = self.event_mappings.get(event.event_type)
            if mapper:
                atom = await mapper(event)
                if atom:
                    # Process through MCP server
                    await self.mcp_server.handle_event("AtomCreated", atom.to_dict())
                    logger.debug(
                        f"Converted event {event.event_type} to atom {atom.atom_id}"
                    )
        except Exception as e:
            logger.error(f"Error bridging event {event.event_type}: {e}")

    async def _map_user_message(self, event) -> Atom:
        """Map user message event to atom"""
        return Atom(
            atom_type="USER_INTERACTION",
            title=f"User Message: {event.data.get('text', '')[:50]}...",
            content=json.dumps(event.data),
            meta={
                "event_id": event.event_id,
                "correlation_id": getattr(event, "correlation_id", None),
                "timestamp": event.timestamp,
                "source": "user_message",
            },
        )

    async def _map_tool_created(self, event) -> Atom:
        """Map tool creation event to atom"""
        tool_name = event.data.get("name", "Unknown Tool")
        return Atom(
            atom_type="TOOL_CAPABILITY",
            title=f"Tool Created: {tool_name}",
            content=json.dumps(event.data),
            meta={
                "event_id": event.event_id,
                "correlation_id": getattr(event, "correlation_id", None),
                "timestamp": event.timestamp,
                "source": "tool_creation",
            },
        )

    async def _map_sot_executed(self, event) -> Atom:
        """Map Script-of-Thought execution event to atom"""
        return Atom(
            atom_type="REASONING_TRACE",
            title="Script-of-Thought Execution",
            content=event.data.get("reasoning_trace", ""),
            meta={
                "event_id": event.event_id,
                "correlation_id": getattr(event, "correlation_id", None),
                "timestamp": event.timestamp,
                "steps": event.data.get("steps", []),
                "source": "sot_execution",
            },
        )

    async def _map_state_transition(self, event) -> Atom:
        """Map state transition event to atom"""
        from_state = event.data.get("from_state", "unknown")
        to_state = event.data.get("to_state", "unknown")
        return Atom(
            atom_type="STATE_TRANSITION",
            title=f"State Transition: {from_state} → {to_state}",
            content=json.dumps(event.data),
            meta={
                "event_id": event.event_id,
                "correlation_id": getattr(event, "correlation_id", None),
                "timestamp": event.timestamp,
                "source": "state_machine",
            },
        )

    async def _map_tool_call(self, event) -> Atom:
        """Map tool call event to atom"""
        tool_id = event.data.get("tool_id", "unknown")
        return Atom(
            atom_type="TOOL_CALL",
            title=f"Tool Call: {tool_id}",
            content=json.dumps(event.data),
            meta={
                "event_id": event.event_id,
                "correlation_id": getattr(event, "correlation_id", None),
                "timestamp": event.timestamp,
                "source": "tool_execution",
            },
        )

    async def _map_tool_response(self, event) -> Atom:
        """Map tool response event to atom"""
        tool_id = event.data.get("tool_id", "unknown")
        status = event.data.get("status", "unknown")
        return Atom(
            atom_type="TOOL_RESPONSE",
            title=f"Tool Response: {tool_id} ({status})",
            content=json.dumps(event.data),
            meta={
                "event_id": event.event_id,
                "correlation_id": getattr(event, "correlation_id", None),
                "timestamp": event.timestamp,
                "source": "tool_execution",
            },
        )
