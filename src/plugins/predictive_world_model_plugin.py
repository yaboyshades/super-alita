#!/usr/bin/env python3
"""
🔮 PREDICTIVE WORLD MODEL PLUGIN - Integration wrapper for Super Alita
Provides predictive modeling and strategic planning capabilities
"""

import logging
from typing import Any

from src.core.events import BaseEvent
from src.core.plugin_interface import PluginInterface
from src.core.predictive_world_model import PredictiveWorldModel

logger = logging.getLogger(__name__)


class PredictiveWorldModelPlugin(PluginInterface):
    """Plugin wrapper for predictive world model capabilities"""

    @property
    def name(self) -> str:
        return "predictive_world_model_plugin"

    async def setup(self, event_bus, store, config: dict[str, Any]):
        await super().setup(event_bus, store, config)

        # Get predictive world model configuration
        pwm_config = config.get("predictive_world_model", {})

        # Initialize predictive world model with enhanced startup integration
        self.world_model = PredictiveWorldModel(
            max_history=pwm_config.get("max_history", 10000),
            event_bus=event_bus,
            config=pwm_config,
        )

        logger.info(
            "🔮 Predictive World Model Plugin initialized with enhanced integration"
        )

    async def start(self):
        await super().start()

        # Initialize and start the world model
        startup_success = await self.world_model.startup()
        if not startup_success:
            logger.error("❌ Failed to start Predictive World Model")
            return

        # Subscribe to prediction and learning events
        await self.subscribe("prediction_request", self._handle_prediction_request)
        await self.subscribe("execution_result", self._handle_execution_learning)
        await self.subscribe("tool_result", self._handle_tool_result_learning)

        # Emit startup status
        await self.emit_event(
            "predictive_model_status",
            source_plugin=self.name,
            status="started",
            startup_status=self.world_model.get_startup_status(),
            health_status=self.world_model.get_health_status(),
            success=True,
        )

        logger.info("🔮 Predictive World Model Plugin started with enhanced features")

    async def shutdown(self):
        logger.info("🔮 Predictive World Model Plugin shutting down")

        # Gracefully shutdown the world model
        if hasattr(self, "world_model"):
            await self.world_model.shutdown()

        await super().shutdown()

    async def _handle_prediction_request(self, event: BaseEvent):
        """Handle requests for outcome prediction"""
        try:
            current_state = event.data.get("current_state", {})
            proposed_action = event.data.get("proposed_action", "")
            context = event.data.get("context", {})

            # Generate prediction
            prediction = await self.world_model.predict_outcome(
                current_state=current_state,
                proposed_action=proposed_action,
                context=context,
            )

            # Emit prediction result
            await self.emit_event(
                "prediction_result",
                source_plugin=self.name,
                conversation_id=event.conversation_id,
                prediction={
                    "success_probability": prediction.success_probability,
                    "expected_duration": prediction.expected_duration,
                    "confidence": prediction.confidence,
                    "risk_factors": prediction.risk_factors,
                    "alternative_strategies": prediction.alternative_strategies,
                },
                current_state=current_state,
                proposed_action=proposed_action,
                success=True,
            )

            logger.info(f"🔮 Generated prediction for action: {proposed_action}")

        except Exception as e:
            logger.exception(f"❌ Prediction error: {e}")
            await self.emit_event(
                "prediction_result",
                source_plugin=self.name,
                conversation_id=getattr(event, "conversation_id", None),
                success=False,
                error=str(e),
            )

    async def _handle_execution_learning(self, event: BaseEvent):
        """Handle execution results for learning"""
        try:
            initial_state = event.data.get("initial_state", {})
            action = event.data.get("action", "")
            final_state = event.data.get("final_state", {})
            success = event.data.get("success", False)
            duration = event.data.get("duration", 0.0)
            context = event.data.get("context", {})

            # Learn from execution
            await self.world_model.learn_from_execution(
                initial_state=initial_state,
                action=action,
                final_state=final_state,
                success=success,
                duration=duration,
                context=context,
            )

            logger.info(f"🔮 Learned from execution: {action} (success={success})")

        except Exception as e:
            logger.exception(f"❌ Learning error: {e}")

    async def _handle_tool_result_learning(self, event: BaseEvent):
        """Handle tool results for automatic learning"""
        try:
            # Extract learning data from tool results
            tool_name = event.data.get("tool_name", "unknown")
            success = event.data.get("success", False)
            duration = getattr(event, "execution_time", 0.0)

            # Create simplified state representation
            initial_state = {
                "tool_type": tool_name,
                "complexity": "medium",  # Could be enhanced with actual complexity analysis
            }

            final_state = {"completed": success, "tool_type": tool_name}

            context = {
                "learning_source": "tool_result",
                "conversation_id": event.conversation_id,
            }

            # Learn from tool execution
            await self.world_model.learn_from_execution(
                initial_state=initial_state,
                action=f"execute_{tool_name}",
                final_state=final_state,
                success=success,
                duration=duration,
                context=context,
            )

            logger.debug(
                f"🔮 Learned from tool result: {tool_name} (success={success})"
            )

        except Exception as e:
            logger.exception(f"❌ Tool result learning error: {e}")

    async def get_model_summary(self) -> dict[str, Any]:
        """Get current model state summary"""
        try:
            summary = self.world_model.get_model_summary()

            # Emit model summary event
            await self.emit_event(
                "model_summary",
                source_plugin=self.name,
                summary=summary,
                success=True,
            )

            return summary

        except Exception as e:
            logger.exception(f"❌ Model summary error: {e}")
            return {"error": str(e), "success": False}
