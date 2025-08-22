#!/usr/bin/env python3
"""
🧠 ADAPTIVE NEURAL ATOM PLUGIN - Integration wrapper for Super Alita
Provides adaptive learning capabilities as a plugin service
"""

import logging
from typing import Any

from src.core.adaptive_neural_atom import AdaptiveMemoryAtom, AdaptiveTextProcessor
from src.core.events import BaseEvent, ToolCallEvent
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class AdaptiveNeuralAtomPlugin(PluginInterface):
    """Plugin wrapper for adaptive neural atom capabilities"""

    @property
    def name(self) -> str:
        return "adaptive_neural_atom_plugin"

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)

        # Initialize adaptive components
        self.text_processor = AdaptiveTextProcessor()

        # Register neural atoms in store
        await self._register_adaptive_atoms()

        logger.info("🧠 Adaptive Neural Atom Plugin initialized")

    async def start(self) -> None:
        await super().start()

        # Subscribe to relevant events
        await self.subscribe("adaptive_text_request", self._handle_text_processing)
        await self.subscribe("adaptive_memory_request", self._handle_memory_creation)
        await self.subscribe("tool_call", self._handle_tool_call)

        logger.info("🧠 Adaptive Neural Atom Plugin started")

    async def shutdown(self) -> None:
        logger.info("🧠 Adaptive Neural Atom Plugin shutting down")

    async def _register_adaptive_atoms(self) -> None:
        """Register adaptive neural atoms in the store"""
        try:
            # Register text processor with store
            await self.store.register(self.text_processor)

            logger.info("✅ Registered adaptive neural atoms in store")
        except Exception:
            logger.exception("❌ Failed to register adaptive atoms")

    async def _handle_text_processing(self, event: BaseEvent) -> None:
        """Handle adaptive text processing requests"""
        try:
            text_input = event.data.get("text", "")

            # Process with adaptive text processor
            result = await self.text_processor.execute(text_input)

            # Emit result event
            await self.emit_event(
                "text_processing_result",
                source_plugin=self.name,
                conversation_id=event.conversation_id,
                original_text=text_input,
                processed_result=result,
                success=True,
            )

            logger.info(f"🧠 Processed text: {len(text_input)} chars")

        except Exception as e:
            logger.exception("❌ Text processing error")
            await self.emit_event(
                "text_processing_result",
                source_plugin=self.name,
                conversation_id=event.conversation_id,
                success=False,
                error=str(e),
            )

    async def _handle_memory_creation(self, event: BaseEvent) -> None:
        """Handle adaptive memory creation requests"""
        try:
            content = event.data.get("content", "")

            # Create adaptive memory atom with content
            memory_atom = AdaptiveMemoryAtom(content=content)

            # Register in store - use await for async method
            await self.store.register(memory_atom)

            # Emit result event with proper parameter structure
            await self.emit_event(
                "memory_creation_result",
                source_plugin=self.name,
                conversation_id=event.conversation_id,
                memory_key=memory_atom.key,
                content_length=len(content),
                success=True,
            )

            logger.info(f"🧠 Created adaptive memory: {memory_atom.key}")

        except Exception as e:
            logger.exception("❌ Memory creation error")
            await self.emit_event(
                "memory_creation_result",
                source_plugin=self.name,
                conversation_id=event.conversation_id,
                success=False,
                error=str(e),
            )

    async def _handle_tool_call(self, event: ToolCallEvent) -> None:
        """Handle tool calls for adaptive neural atom operations"""
        if event.tool_name != "adaptive_neural_atom":
            return

        try:
            # Extract parameters
            operation = event.parameters.get("operation")
            content = event.parameters.get("content", "")

            result = None

            if operation == "process_text":
                result = await self.text_processor.execute(content)
            elif operation == "create_memory":
                # Create memory atom with content
                memory_atom = AdaptiveMemoryAtom(content=content)
                await self.store.register(memory_atom)
                result = {"memory_key": memory_atom.key, "status": "created"}
            else:
                # Return error for unknown operation
                await self.emit_event(
                    "tool_result",
                    tool_call_id=event.tool_call_id,
                    success=False,
                    error=f"Unknown operation: {operation}",
                    conversation_id=event.conversation_id,
                )
                return

            # Emit successful result
            await self.emit_event(
                "tool_result",
                tool_call_id=event.tool_call_id,
                success=True,
                result=result,
                conversation_id=event.conversation_id,
            )

        except Exception as e:
            logger.exception("❌ Tool call error")
            # Ensure tool result is always emitted even on error
            await self.emit_event(
                "tool_result",
                tool_call_id=event.tool_call_id,
                success=False,
                error=str(e),
                conversation_id=event.conversation_id,
            )
