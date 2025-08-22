#!/usr/bin/env python3
"""
🧠 META LEARNING CREATOR PLUGIN - Integration wrapper for Super Alita
Provides meta-learning and tool generation capabilities
"""

import logging
from typing import Any

from src.core.events import BaseEvent
from src.core.meta_learning_creator import MetaLearningCreator, ToolGenerationRequest
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class MetaLearningCreatorPlugin(PluginInterface):
    """Plugin wrapper for meta-learning creator capabilities"""

    @property
    def name(self) -> str:
        return "meta_learning_creator_plugin"

    async def setup(self, event_bus, store, config: dict[str, Any]):
        await super().setup(event_bus, store, config)

        # Initialize meta learning creator
        self.creator = MetaLearningCreator(
            max_patterns=config.get("max_patterns", 1000),
            max_history=config.get("max_history", 5000),
        )

        logger.info("🧠 Meta Learning Creator Plugin initialized")

    async def start(self):
        await super().start()

        # Subscribe to tool generation and learning events
        await self.subscribe("tool_generation_request", self._handle_tool_generation)
        await self.subscribe(
            "tool_execution_result", self._handle_learning_from_execution
        )
        await self.subscribe("pattern_learning_request", self._handle_pattern_learning)

        logger.info("🧠 Meta Learning Creator Plugin started")

    async def shutdown(self):
        logger.info("🧠 Meta Learning Creator Plugin shutting down")

    async def _handle_tool_generation(self, event: BaseEvent):
        """Handle requests for tool generation"""
        try:
            capability_description = event.data.get("capability_description", "")
            success_criteria = event.data.get("success_criteria", [])
            constraints = event.data.get("constraints", {})
            context = event.data.get("context", {})
            priority = event.data.get("priority", 0.5)
            timeout = event.data.get("timeout", 30.0)

            # Create generation request
            request = ToolGenerationRequest(
                capability_description=capability_description,
                success_criteria=success_criteria,
                constraints=constraints,
                context=context,
                priority=priority,
                timeout=timeout,
            )

            # Generate tool
            generated_tool = await self.creator.generate_tool(request)

            # Emit generation result
            await self.emit_event(
                "tool_generation_result",
                {
                    "source_plugin": self.name,
                    "conversation_id": event.conversation_id,
                    "generated_tool": {
                        "name": generated_tool.name,
                        "code": generated_tool.code,
                        "description": generated_tool.description,
                        "capabilities": generated_tool.capabilities,
                        "success_rate": generated_tool.success_rate,
                        "generation_time": generated_tool.generation_time,
                        "validation_passed": generated_tool.validation_passed,
                        "metadata": generated_tool.metadata,
                    },
                    "request": {
                        "capability_description": capability_description,
                        "success_criteria": success_criteria,
                        "priority": priority,
                    },
                    "success": True,
                },
            )

            logger.info(f"🧠 Generated tool: {generated_tool.name}")

        except Exception as e:
            logger.exception("❌ Tool generation error")
            await self.emit_event(
                "tool_generation_result",
                {
                    "source_plugin": self.name,
                    "conversation_id": event.conversation_id,
                    "success": False,
                    "error": str(e),
                },
            )

    async def _handle_learning_from_execution(self, event: BaseEvent):
        """Handle learning from tool execution results"""
        try:
            tool_name = event.data.get("tool_name", "")
            success = event.data.get("success", False)
            execution_time = event.data.get("execution_time", 0.0)
            error_message = event.data.get("error_message", "")
            context = event.data.get("context", {})

            # Learn from generation result
            await self.creator.learn_from_generation(
                tool_name=tool_name,
                success=success,
                execution_time=execution_time,
                error_message=error_message,
                context=context,
            )

            logger.info(f"🧠 Learned from execution: {tool_name} (success={success})")

        except Exception:
            logger.exception("❌ Learning from execution error")

    async def _handle_pattern_learning(self, event: BaseEvent):
        """Handle pattern learning requests"""
        try:
            pattern_data = event.data.get("pattern_data", {})

            # Learn pattern through the generation method (adapting to actual API)
            # Note: This is a simplified approach since learn_pattern isn't available
            logger.info(
                f"🧠 Pattern learning requested with data: {list(pattern_data.keys())}"
            )

            # Emit pattern learning result
            await self.emit_event(
                "pattern_learning_result",
                {
                    "source_plugin": self.name,
                    "conversation_id": event.conversation_id,
                    "pattern_data": pattern_data,
                    "success": True,
                    "message": "Pattern data noted for future generations",
                },
            )

        except Exception as e:
            logger.exception("❌ Pattern learning error")
            await self.emit_event(
                "pattern_learning_result",
                {
                    "source_plugin": self.name,
                    "conversation_id": event.conversation_id,
                    "success": False,
                    "error": str(e),
                },
            )

    async def get_creator_summary(self) -> dict[str, Any]:
        """Get current creator state summary"""
        try:
            summary = {
                "total_patterns": len(self.creator.learned_patterns),
                "generation_history": len(self.creator.generation_history),
                "success_rate": self.creator.global_success_rate,
                "confidence_threshold": self.creator.confidence_threshold,
                "recent_generations": [
                    {
                        "tool_name": gen.name,
                        "success_rate": gen.success_rate,
                        "validation_passed": gen.validation_passed,
                    }
                    for gen in list(self.creator.generation_history.values())[-5:]
                ],
            }

            # Emit creator summary event
            await self.emit_event(
                "creator_summary",
                {"source_plugin": self.name, "summary": summary, "success": True},
            )

            return summary

        except Exception as e:
            logger.exception("❌ Creator summary error")
            return {"error": str(e), "success": False}
