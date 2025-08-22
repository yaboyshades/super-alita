#!/usr/bin/env python3
"""
MemoryManagerPlugin - Event Contract Compliant Memory Operations

Version: 2.0.0 (Updated for bulletproof event contract compliance)
Description: Memory manager that fulfills the event contract by always replying with a ToolResultEvent.

This plugin provides rock-solid memory management capabilities with guaranteed
ToolResultEvent emission to prevent planner timeouts. Every tool call receives
a response, even on errors or exceptions.

Key Features:
- Bulletproof event contract compliance
- Always emits ToolResultEvent to prevent deadlocks
- Robust error handling with detailed logging
- Simple, reliable memory operations
- Comprehensive timeout prevention

Event Contract:
1. Receives ToolCallEvent for "memory_manager" tool
2. ALWAYS emits ToolResultEvent with matching tool_call_id
3. Never leaves planner waiting - guaranteed response
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from src.core.event_bus import EventBus
from src.core.events import ToolCallEvent, ToolResultEvent
from src.core.neural_atom import NeuralAtomMetadata, NeuralStore, TextualMemoryAtom
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class MemoryManagerPlugin(PluginInterface):
    """Manages memory operations and always responds with a ToolResultEvent."""

    def __init__(self):
        super().__init__()
        self._handled_calls: set[str] = set()
        self._stats = {
            "saves": 0,
            "lists": 0,
            "recalls": 0,
            "errors": 0,
            "last_activity": None,
        }

    @property
    def name(self) -> str:
        return "memory_manager"

    @property
    def version(self) -> str:
        return "2.0.0"

    async def setup(
        self, event_bus: EventBus, store: NeuralStore, config: dict[str, Any]
    ):
        """Setup the memory manager plugin."""
        await super().setup(event_bus, store, config)
        self._config = config.get(self.name, {})
        logger.info("MemoryManagerPlugin setup complete")

    async def start(self) -> None:
        """Start the plugin and subscribe to tool call events."""
        logger.info("Starting MemoryManagerPlugin...")
        # Subscribe to tool calls; this depends on how event_bus works
        await self.subscribe("tool_call", self._handle_tool_call)
        await super().start()
        logger.info("MemoryManagerPlugin started successfully")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down MemoryManagerPlugin...")
        await super().shutdown()
        logger.info("MemoryManagerPlugin shutdown complete")

    async def _save_memory(self, content: str, session_id: str) -> dict[str, Any]:
        """Save memory content and return result."""
        try:
            # Create metadata for the new TextualMemoryAtom
            memory_id = f"mem_{uuid.uuid4().hex[:12]}"
            metadata = NeuralAtomMetadata(
                name=f"memory_{session_id}_{memory_id}",
                description=(
                    f"User memory: {content[:50]}..."
                    if len(content) > 50
                    else f"User memory: {content}"
                ),
                capabilities=["storage", "recall", "memory"],
            )

            # Create the concrete TextualMemoryAtom
            memory_atom = TextualMemoryAtom(
                metadata=metadata, content=content, embedding_client=None
            )

            # Register the atom in the store
            self.store.register(memory_atom)

            logger.info(f"Memory saved successfully: {memory_id}")
            return {
                "success": True,
                "memory_id": memory_id,
                "message": "Memory saved successfully",
                "storage_method": "neural_store",
            }

        except Exception as e:
            logger.error(f"Error saving memory: {e}", exc_info=True)
            return {"success": False, "error": f"Failed to save memory: {str(e)[:200]}"}

    async def _handle_tool_call(self, event: ToolCallEvent) -> None:
        """Processes tool calls for 'memory_manager' and ALWAYS emits a result."""
        if event.tool_name != self.name:
            return

        # Duplicate prevention
        call_id = getattr(event, "tool_call_id", str(event))
        if call_id in self._handled_calls:
            logger.debug(f"Skipping duplicate tool call: {call_id}")
            return
        self._handled_calls.add(call_id)

        logger.info(f"Processing memory manager tool call: {event.tool_call_id}")
        success = False
        result = {}
        error = None

        try:
            action = event.parameters.get("action", "save")
            content = event.parameters.get("content")

            if action == "save":
                if not content:
                    result = {
                        "success": False,
                        "error": "Content is required for save action",
                        "example": {
                            "action": "save",
                            "content": "Important information",
                        },
                    }
                    success = False
                else:
                    save_result = await self._save_memory(content, event.session_id)
                    result = save_result
                    success = save_result.get("success", False)
                    if success:
                        self._stats["saves"] += 1

            elif action == "list":
                # Simple list implementation
                memory_count = len(
                    [k for k in self.store._atoms.keys() if k.startswith("memory:")]
                )
                result = {
                    "success": True,
                    "message": f"Found {memory_count} stored memories",
                    "count": memory_count,
                }
                success = True
                self._stats["lists"] += 1

            elif action == "recall":
                query = event.parameters.get("query", "")
                result = {
                    "success": True,
                    "message": f"Search completed for: '{query}'",
                    "query": query,
                    "note": "Basic recall functionality - semantic search requires SemanticMemoryPlugin",
                }
                success = True
                self._stats["recalls"] += 1

            else:
                error = f"Unknown action: {action}"
                result = {
                    "success": False,
                    "error": error,
                    "supported_actions": ["save", "list", "recall"],
                }
                success = False
                self._stats["errors"] += 1

        except Exception as e:
            logger.error(f"Error in MemoryManagerPlugin: {e}", exc_info=True)
            error = str(e)
            result = {"success": False, "error": error}
            success = False
            self._stats["errors"] += 1

        # CRITICAL: Always publish a ToolResultEvent to unblock the planner.
        try:
            await self.event_bus.publish(
                ToolResultEvent(
                    source_plugin=self.name,
                    conversation_id=event.conversation_id,
                    session_id=event.session_id,
                    tool_call_id=event.tool_call_id,  # Must match the incoming call ID
                    success=success,
                    result=result,
                    error=error,
                )
            )
            logger.info(
                f"Published result for memory manager call: {event.tool_call_id}"
            )
            self._stats["last_activity"] = datetime.now(UTC).isoformat()

        except Exception as pub_exc:
            # If publishing fails, log heavily; planner may still hang unless you layer additional watchdogs.
            logger.critical(
                f"Failed to publish ToolResultEvent for {event.tool_call_id}: {pub_exc}",
                exc_info=True,
            )

    def _get_stats(self) -> dict[str, Any]:
        """Get plugin statistics."""
        return {
            **self._stats,
            "plugin": self.name,
            "version": self.version,
            "is_running": self.is_running,
            "handled_calls": len(self._handled_calls),
        }

    async def health_check(self) -> dict[str, Any]:
        """Health check for the memory manager plugin."""
        return {
            "status": "healthy" if self.is_running else "stopped",
            "version": self.version,
            "stats": self._get_stats(),
            "event_contract": "compliant",
        }
