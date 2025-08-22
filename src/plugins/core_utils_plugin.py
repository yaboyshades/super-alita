"""
CoreUtils Plugin for Super Alita.
Provides safe arithmetic calculation and string manipulation tools.
"""

import logging
from typing import Any

from src.core.events import ToolCallEvent
from src.core.plugin_interface import PluginInterface
from src.tools.core_utils import CoreUtils

logger = logging.getLogger(__name__)


class CoreUtilsPlugin(PluginInterface):
    """Plugin providing core utility tools: calculator and string operations."""

    @property
    def name(self) -> str:
        return "core_utils"

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        """Initialize the CoreUtils plugin."""
        await super().setup(event_bus, store, config)
        logger.info(
            "🧮 CoreUtils Plugin initialized - calculator and string tools ready"
        )

    async def start(self) -> None:
        """Start the CoreUtils plugin and register event handlers."""
        await super().start()

        # Subscribe to tool call events
        await self.subscribe("tool_call", self._handle_tool_call)

        logger.info(
            "🚀 CoreUtils Plugin started - listening for core.calculate and core.reverse"
        )

    async def _handle_tool_call(self, event: ToolCallEvent) -> None:
        """Handle tool call events for core utilities."""
        try:
            tool_name = event.tool_name.strip().lower()

            # Only handle our tools
            if tool_name not in ("core.calculate", "core.reverse"):
                return

            if tool_name == "core.calculate":
                await self._handle_calculate(event)
            elif tool_name == "core.reverse":
                await self._handle_reverse(event)

        except Exception as e:
            logger.exception(f"CoreUtils tool call failed: {e}")
            await self._emit_error_result(event, str(e))

    async def _handle_calculate(self, event: ToolCallEvent) -> None:
        """Handle arithmetic calculation requests."""
        try:
            expression = event.parameters.get("expression", "")
            if not expression:
                raise ValueError("expression parameter is required")

            # Use CoreUtils to safely calculate
            result = CoreUtils.calculate(str(expression))

            # Emit successful result
            await self.emit_event(
                "tool_result",
                tool_call_id=event.tool_call_id,
                conversation_id=event.conversation_id,
                session_id=event.session_id,
                success=True,
                result={"value": result, "expression": expression},
            )

            logger.info(f"✅ Calculated: {expression} = {result}")

        except Exception as e:
            logger.error(
                f"❌ Calculation failed for '{event.parameters.get('expression', '')}': {e}"
            )
            await self._emit_error_result(event, f"Calculation error: {e}")

    async def _handle_reverse(self, event: ToolCallEvent) -> None:
        """Handle string reversal requests."""
        try:
            text = event.parameters.get("text", "")
            if not text:
                raise ValueError("text parameter is required")

            # Use CoreUtils to reverse string
            reversed_text = CoreUtils.reverse_string(str(text))

            # Emit successful result
            await self.emit_event(
                "tool_result",
                tool_call_id=event.tool_call_id,
                conversation_id=event.conversation_id,
                session_id=event.session_id,
                success=True,
                result={"text": reversed_text, "original": text},
            )

            logger.info(f"🔄 Reversed: '{text}' → '{reversed_text}'")

        except Exception as e:
            logger.error(
                f"❌ String reversal failed for '{event.parameters.get('text', '')}': {e}"
            )
            await self._emit_error_result(event, f"String reversal error: {e}")

    async def _emit_error_result(
        self, event: ToolCallEvent, error_message: str
    ) -> None:
        """Emit an error result for a failed tool call."""
        await self.emit_event(
            "tool_result",
            tool_call_id=event.tool_call_id,
            conversation_id=event.conversation_id,
            session_id=event.session_id,
            success=False,
            result={},
            error=error_message,
        )

    async def shutdown(self) -> None:
        """Gracefully shutdown the CoreUtils plugin."""
        await super().shutdown()
        logger.info("📴 CoreUtils Plugin shutdown complete")
