"""
Atom for high-level memory operations.
Version: 1.0.3 - Fixed syntax errors and duplicated error handling code
"""

import logging
from typing import Any

from src.core.events import ToolResultEvent
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)

# Constants
MAX_TEXT_PREVIEW_LENGTH = 100
MAX_CONTENT_PREVIEW_LENGTH = 50


class MemoryManagerAtom(PluginInterface):
    """
    Atom that handles high-level memory operations like saving and recalling information.

    Features:
    - Save user notes and information to semantic memory
    - List and recall stored memories
    - Handles memory-related tool calls from the planner
    """

    @property
    def name(self) -> str:
        return "memory_manager"

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        """Initialize the memory manager atom."""
        await super().setup(event_bus, store, config)
        logger.info("MemoryManagerAtom setup complete")

    async def start(self) -> None:
        """Start the memory manager atom."""
        await super().start()
        await self.subscribe("tool_call", self._handle_tool_call)
        logger.info("MemoryManagerAtom started - ready to handle memory operations")

    async def shutdown(self) -> None:
        """Shutdown the memory manager atom."""
        logger.info("MemoryManagerAtom shutdown complete")

    async def _handle_tool_call(self, event: Any) -> None:
        """Handle tool calls for memory operations."""
        # Initialize result and success flag
        result: dict[str, Any] = {}
        success = False

        # Extract tool info defensively from different event formats
        tool_name: str | None = None
        tool_call_id = "unknown"
        conversation_id = "default"
        session_id = "default"
        params: dict[str, Any] = {}

        try:
            if hasattr(event, "tool") and isinstance(event.tool, dict):
                tool_name = event.tool.get("name")
                tool_call_id = event.tool.get("call_id", "unknown")
                conversation_id = event.tool.get("conversation_id", "default")
                session_id = event.tool.get("session_id", "default")
                params = getattr(event, "params", {})
            else:
                # Fallback for other event formats
                tool_name = getattr(event, "tool_name", None)
                tool_call_id = getattr(event, "tool_call_id", "unknown")
                conversation_id = getattr(event, "conversation_id", "default")
                session_id = getattr(event, "session_id", "default")
                params = getattr(event, "parameters", getattr(event, "params", {}))

            if tool_name != "memory_manager":
                return  # Not for us

            logger.info(f"🧠 MemoryManager handling call: {tool_call_id}")

            # Process the memory operation inside try-except block
            try:
                action = params.get("action", "list")
                result = {"tool": "memory_manager", "action": action}

                if action == "save":
                    content = params.get("content", "")
                    if not content.strip():
                        result["error"] = "No content provided to save"
                    elif self.store is not None and hasattr(self.store, "upsert"):
                        memory_id = await self.store.upsert(
                            content={"text": content, "type": "user_note"},
                            hierarchy_path=["memory", "user_notes"],
                            owner_plugin=self.name,
                        )
                        result["status"] = "saved"
                        result["summary"] = (
                            f"Saved memory: {content[:MAX_CONTENT_PREVIEW_LENGTH]}..."
                        )
                        result["memory_id"] = memory_id
                        success = True
                        logger.info(
                            f"💾 Saved user note: {content[:MAX_CONTENT_PREVIEW_LENGTH]}..."
                        )
                    else:
                        result["error"] = "Memory storage not available"

                elif action in {"list", "recall"}:
                    memories = []
                    if self.store is not None and hasattr(self.store, "query"):
                        # Try semantic search for user notes
                        query_text = params.get("query", "user notes memories")
                        search_results = await self.store.query(
                            query_text=query_text,
                            hierarchy_filter=["memory", "user_notes"],
                            top_k=5,
                        )
                        for result_item in search_results:
                            content = result_item.get("content", {})
                            if isinstance(content, dict):
                                text = content.get("text", str(content))
                            else:
                                text = str(content)
                            memories.append(
                                text[:MAX_TEXT_PREVIEW_LENGTH] + "..."
                                if len(text) > MAX_TEXT_PREVIEW_LENGTH
                                else text
                            )

                    result["memories"] = memories
                    result["count"] = len(memories)
                    result["summary"] = f"Found {len(memories)} memories"
                    success = True
                    logger.info(f"🔍 Retrieved {len(memories)} memories")

                else:
                    result["error"] = f"Unknown action: {action}"
                    logger.warning(f"Unknown memory action: {action}")

            except Exception as e:
                logger.error(f"Failed to process memory operation: {e}", exc_info=True)
                result["error"] = (
                    f"Operation failed: {str(e)[:MAX_TEXT_PREVIEW_LENGTH]}"
                )
                success = False

            # CRITICAL: Always publish a ToolResultEvent in response to a ToolCallEvent
            if self.event_bus is not None and hasattr(self.event_bus, "publish"):
                try:
                    await self.event_bus.publish(
                        ToolResultEvent(
                            source_plugin=self.name,
                            conversation_id=conversation_id,
                            tool_call_id=tool_call_id,
                            session_id=session_id,
                            success=success,
                            result=result,
                            error=""
                            if success
                            else result.get("error", "Unknown error"),
                        )
                    )
                    logger.debug(
                        f"Published ToolResultEvent for tool_call_id {tool_call_id} (success={success})"
                    )
                except Exception as e:
                    logger.critical(
                        f"Failed to publish ToolResultEvent for tool_call_id {tool_call_id}: {e}",
                        exc_info=True,
                    )
            else:
                logger.warning("Event bus not available for publishing ToolResultEvent")

            logger.info(f"✅ MemoryManager completed: {action}")

        except Exception as e:
            logger.error(f"Error in memory manager tool call: {e}", exc_info=True)

            # CRITICAL: Always emit a ToolResultEvent even on failure
            # Extract IDs defensively for error case with improved logic
            conversation_id = "default"
            session_id = "default"
            tool_call_id = "unknown"

            # Primary extraction from event attributes
            if hasattr(event, "conversation_id"):
                conversation_id = event.conversation_id
            if hasattr(event, "session_id"):
                session_id = event.session_id
            if hasattr(event, "tool_call_id"):
                tool_call_id = event.tool_call_id

            # Secondary extraction from nested tool structure
            if hasattr(event, "tool") and isinstance(event.tool, dict):
                conversation_id = event.tool.get("conversation_id", conversation_id)
                session_id = event.tool.get("session_id", session_id)
                tool_call_id = event.tool.get("call_id", tool_call_id)

            # Ensure we have meaningful IDs for debugging
            if tool_call_id == "unknown":
                tool_call_id = f"error_{id(event)}"  # Use object ID as fallback

            error_message = f"Memory manager error: {e!s}"
            if len(error_message) > MAX_TEXT_PREVIEW_LENGTH:
                error_message = error_message[: MAX_TEXT_PREVIEW_LENGTH - 3] + "..."

            if self.event_bus is not None and hasattr(self.event_bus, "publish"):
                try:
                    await self.event_bus.publish(
                        ToolResultEvent(
                            source_plugin=self.name,
                            conversation_id=conversation_id,
                            tool_call_id=tool_call_id,
                            session_id=session_id,
                            success=False,
                            result={"error": error_message},
                            error=error_message,
                        )
                    )
                    logger.debug(
                        f"Published error ToolResultEvent for tool_call_id {tool_call_id}"
                    )
                except Exception as publish_error:
                    logger.critical(
                        f"Failed to publish error ToolResultEvent: {publish_error}",
                        exc_info=True,
                    )
            else:
                logger.warning(
                    "Event bus not available for publishing error ToolResultEvent"
                )
