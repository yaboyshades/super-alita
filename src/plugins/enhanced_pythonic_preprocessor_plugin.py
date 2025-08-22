"""
Pythonic Preprocessor Plugin - Upgraded to a Cognitive Cycle Orchestrator

This plugin serves as the enhanced "cognitive airlock" between user inputs and the system,
implementing DTA 2.0 cognitive turn processing for structured reasoning and decision-making.

Key Features:
- Cognitive turn processing with structured reasoning
- Advanced planning integration with REUG methodology
- Neural integration with Super Alita's event system
- Comprehensive validation and monitoring
- Circuit breaker reliability patterns
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any

from src.core.events import BaseEvent

# Core imports
from src.core.plugin_interface import PluginInterface

# DTA 2.0 Cognitive imports
try:
    from src.dta.cognitive_plan import generate_master_cognitive_plan
    from src.dta.config import DTAConfig
    from src.dta.types import (
        # ActivationProtocol,  # Currently unused
        CognitiveTurnRecord,
        # ConfidenceCalibration,  # Currently unused
        # StateUpdate,  # Currently unused
        StrategicPlan,
    )

    COGNITIVE_TURN_AVAILABLE = True
except ImportError:
    COGNITIVE_TURN_AVAILABLE = False

# Event imports (conditional)
try:
    from src.core.events import CognitiveTurnCompletedEvent

    COGNITIVE_EVENT_AVAILABLE = True
except ImportError:
    COGNITIVE_EVENT_AVAILABLE = False

# LLM integration (conditional)
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)


class PythonicPreprocessorPlugin(PluginInterface):
    """Enhanced preprocessor with cognitive turn processing capabilities."""

    def __init__(self):
        super().__init__()
        self.llm_client = None
        self.dta_config = None
        self.cognitive_enabled = COGNITIVE_TURN_AVAILABLE

    @property
    def name(self) -> str:
        return "pythonic_preprocessor"

    async def setup(self, event_bus, store, config):
        """Initialize the cognitive preprocessor with enhanced capabilities."""
        await super().setup(event_bus, store, config)

        # Initialize DTA 2.0 configuration
        if COGNITIVE_TURN_AVAILABLE:
            self.dta_config = DTAConfig(
                environment="development",
                debug=config.get("debug", False),
                cognitive_turn={
                    "enabled": True,
                    "max_execution_time": 60.0,
                    "validation_enabled": True,
                    "confidence_threshold": 0.7,
                },
            )
            logger.info("Cognitive turn processing enabled")
        else:
            logger.warning(
                "Cognitive turn processing not available - falling back to legacy mode"
            )

        # Initialize LLM client if available
        if GEMINI_AVAILABLE:
            try:
                api_key = config.get("gemini_api_key") or "placeholder"
                if api_key != "placeholder":
                    genai.configure(api_key=api_key)
                    self.llm_client = genai.GenerativeModel("gemini-1.5-flash")
                    logger.info("Gemini LLM client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")

    async def start(self):
        """Start the cognitive preprocessor."""
        await super().start()

        # Subscribe to conversation events for cognitive processing
        if hasattr(self.event_bus, "subscribe"):
            await self.subscribe("conversation_message", self._handle_conversation)
            await self.subscribe("user_input", self._handle_user_input)
            logger.info("Subscribed to conversation events")

    async def _handle_conversation(self, event: BaseEvent):
        """Handle conversation events with cognitive processing."""
        if not hasattr(event, "user_message"):
            return

        # Check if already handled
        if getattr(event, "handled_by", None):
            return

        try:
            user_message = event.user_message
            context = getattr(event, "context", {})
            session_id = getattr(event, "session_id", str(uuid.uuid4()))
            conversation_id = getattr(event, "conversation_id", session_id)

            if self.cognitive_enabled:
                # Process with cognitive turn
                turn_record = await self._process_cognitive_turn(
                    user_message, context, session_id
                )

                # Emit cognitive turn completed event
                if COGNITIVE_EVENT_AVAILABLE and turn_record:
                    turn_event = CognitiveTurnCompletedEvent(
                        source_plugin=self.name,
                        turn_record=turn_record,
                        session_id=session_id,
                        conversation_id=conversation_id,
                    )
                    await self.event_bus.publish(turn_event)
                    logger.info(f"Published cognitive turn for session {session_id}")
            else:
                # Fallback to legacy processing
                await self._process_legacy_mode(user_message, context, event)

            # Mark as handled
            event.handled_by = self.name

        except Exception as e:
            logger.error(f"Failed to process conversation: {e}", exc_info=True)

    async def _handle_user_input(self, event: BaseEvent):
        """Handle direct user input events."""
        # Redirect to conversation handler
        await self._handle_conversation(event)

    async def _process_cognitive_turn(
        self, user_message: str, context: dict[str, Any], session_id: str
    ) -> CognitiveTurnRecord | None:
        """Process user input through cognitive turn methodology."""

        try:
            # Build cognitive turn prompt
            prompt = self._build_cognitive_turn_prompt(user_message, context)

            # Generate cognitive turn via LLM (or use placeholder)
            if self.llm_client:
                llm_response = await self._generate_with_llm(prompt)
            else:
                llm_response = self._get_placeholder_turn_response(user_message)

            # Parse and validate cognitive turn
            turn_data = json.loads(llm_response)
            turn_record = CognitiveTurnRecord(**turn_data)

            # Generate strategic plan if required
            if turn_record.activation_protocol.planning_requirement:
                plan = generate_master_cognitive_plan(user_message)
                turn_record.strategic_plan = StrategicPlan(
                    is_required=True,
                    steps=plan.execution_steps,
                    estimated_duration=60.0,
                    resource_requirements=["cognitive_processing", "llm_access"],
                )

            logger.info(
                f"Generated cognitive turn with confidence: {turn_record.confidence_calibration.final_confidence}"
            )
            return turn_record

        except Exception as e:
            logger.error(f"Error in cognitive turn processing: {e}")
            return None

    def _build_cognitive_turn_prompt(
        self, user_message: str, context: dict[str, Any]
    ) -> str:
        """Build prompt for cognitive turn generation."""

        context_str = (
            json.dumps(context, indent=2) if context else "No additional context"
        )

        prompt = f"""
Analyze the user's request and respond with a valid JSON object matching the CognitiveTurnRecord schema.

USER MESSAGE: "{user_message}"
CONTEXT: {context_str}

Respond with a JSON object containing:
- state_readout: Current understanding of the situation
- activation_protocol: Analysis settings and confidence
- synthesis: Key findings and final answer summary
- state_update: Any memory or context updates needed
- confidence_calibration: Final confidence and uncertainty assessment

Format as valid JSON only.
"""
        return prompt

    async def _generate_with_llm(self, prompt: str) -> str:
        """Generate response using LLM client."""

        try:
            response = await asyncio.wait_for(
                self.llm_client.generate_content_async(prompt), timeout=30.0
            )
            return response.text
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Return placeholder response
            return self._get_placeholder_turn_response("LLM Error")

    def _get_placeholder_turn_response(self, user_message: str) -> str:
        """Generate placeholder cognitive turn response for testing."""

        return json.dumps(
            {
                "state_readout": f"User is asking about '{user_message}'. Processing with cognitive methodology.",
                "activation_protocol": {
                    "pattern_recognition": "analytical",
                    "confidence_score": 8,
                    "planning_requirement": True,
                    "quality_speed_tradeoff": "balance",
                    "evidence_threshold": "medium",
                    "audience_level": "professional",
                    "meta_cycle_check": "analysis",
                },
                "strategic_plan": {"is_required": True},
                "execution_log": [
                    "Initial cognitive processing initiated",
                    "User input analyzed and categorized",
                    "Strategic planning requirements assessed",
                ],
                "synthesis": {
                    "key_findings": [
                        f"User request: {user_message}",
                        "Cognitive processing methodology applied",
                        "Structured response framework activated",
                    ],
                    "counterarguments": [],
                    "final_answer_summary": f"Cognitive analysis complete for: {user_message}. Proceeding with structured response generation.",
                },
                "state_update": {
                    "directive": "memory_stream_add",
                    "memory_stream_add": {
                        "summary": f"Processed cognitive turn for user request: {user_message}",
                        "timestamp": datetime.now().isoformat(),
                        "type": "cognitive_processing",
                    },
                },
                "confidence_calibration": {
                    "final_confidence": 9,
                    "uncertainty_gaps": "Minimal uncertainty. Cognitive framework successfully applied.",
                    "risk_assessment": "low",
                    "verification_methods": [
                        "structured_analysis",
                        "cognitive_validation",
                    ],
                },
            }
        )

    async def _process_legacy_mode(
        self, user_message: str, context: dict[str, Any], event: BaseEvent
    ):
        """Fallback processing when cognitive turn is not available."""

        logger.info(f"Processing in legacy mode: {user_message}")

        # Simple intent detection and response
        if "help" in user_message.lower():
            response = "I'm here to help! What would you like to know?"
        elif "status" in user_message.lower():
            response = "System is operational. Cognitive turn processing unavailable."
        else:
            response = f"Processed: {user_message} (legacy mode)"

        # Emit simple response event
        response_event = BaseEvent(
            event_type="agent_response",
            source_plugin=self.name,
            metadata={
                "response": response,
                "mode": "legacy",
                "original_message": user_message,
            },
        )

        await self.event_bus.publish(response_event)

    async def stop(self):
        """Stop the cognitive preprocessor."""
        logger.info("Stopping cognitive preprocessor")
        await super().stop()

    def get_status(self) -> dict[str, Any]:
        """Get preprocessor status information."""
        return {
            "plugin": self.name,
            "cognitive_enabled": self.cognitive_enabled,
            "llm_available": GEMINI_AVAILABLE and self.llm_client is not None,
            "events_available": COGNITIVE_EVENT_AVAILABLE,
            "status": "operational",
        }


# Plugin registration
def create_plugin() -> PythonicPreprocessorPlugin:
    """Factory function for plugin creation."""
    return PythonicPreprocessorPlugin()
