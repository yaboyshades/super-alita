#!/usr/bin/env python3
"""
Enhanced Protocol Integration Plugin for Super Alita
Integrates the Enhanced Security Research Protocol v2.0 with the existing agent architecture.
"""

import logging
from dataclasses import asdict
from typing import Any

from src.core.enhanced_protocol import (
    CognitiveRequest,
    EnhancedProtocolEngine,
    ExpansionTier,
    ReasoningMode,
    create_enhanced_protocol_engine,
)
from src.core.events import BaseEvent, ConversationEvent, ConversationMessageEvent
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class EnhancedProtocolPlugin(PluginInterface):
    """
    Enhanced Protocol Plugin
    Provides advanced cognitive processing capabilities using the Enhanced Security Research Protocol v2.0.
    """

    def __init__(self):
        """Initialize the Enhanced Protocol Plugin."""
        self._name = "enhanced_protocol"
        self.protocol_engine: EnhancedProtocolEngine | None = None
        self.processing_stats = {
            "requests_processed": 0,
            "average_confidence": 0.0,
            "compliance_rate": 0.0,
        }

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return self._name

    async def setup(self, event_bus, store, config):
        """Initialize the enhanced protocol plugin."""
        await super().setup(event_bus, store, config)

        # Initialize the enhanced protocol engine
        self.protocol_engine = create_enhanced_protocol_engine()

        logger.info("Enhanced Protocol Plugin initialized")
        logger.info(f"Protocol Status: {self.protocol_engine.get_protocol_status()}")

    async def start(self):
        """Start the enhanced protocol plugin."""
        await super().start()

        # Subscribe to conversation events for enhanced processing
        await self.subscribe("conversation", self._handle_conversation_with_protocol)
        await self.subscribe(
            "enhanced_cognitive_request", self._handle_enhanced_request
        )

        logger.info(
            "Enhanced Protocol Plugin started - subscribing to cognitive events"
        )

    async def shutdown(self):
        """Shutdown the enhanced protocol plugin."""
        if self.protocol_engine:
            status = self.protocol_engine.get_protocol_status()
            logger.info(f"Enhanced Protocol Plugin shutdown - Final stats: {status}")

        await super().shutdown()

    async def _handle_conversation_with_protocol(self, event: ConversationEvent):
        """Handle conversation events using enhanced protocol processing."""
        try:
            if not self.protocol_engine:
                logger.error("Protocol engine not initialized")
                return

            # Determine if this requires enhanced processing
            if await self._should_use_enhanced_processing(event.user_message):
                await self._process_with_enhanced_protocol(event)

        except Exception as e:
            logger.error(f"Error in enhanced conversation handling: {e}")

    async def _handle_enhanced_request(self, event: BaseEvent):
        """Handle direct enhanced cognitive processing requests."""
        try:
            if not hasattr(event, "cognitive_request"):
                logger.warning(
                    "Enhanced request event missing cognitive_request attribute"
                )
                return

            # Process through enhanced protocol
            request = event.cognitive_request
            response = await self.protocol_engine.process_request(request)

            # Emit enhanced response event
            await self._emit_enhanced_response(event, response)

        except Exception as e:
            logger.error(f"Error in enhanced request processing: {e}")

    async def _should_use_enhanced_processing(self, user_message: str) -> bool:
        """Determine if message requires enhanced protocol processing."""
        # Enhanced processing triggers
        enhanced_keywords = [
            "analyze",
            "assess",
            "evaluate",
            "complex",
            "comprehensive",
            "detailed",
            "thorough",
            "in-depth",
            "research",
            "investigate",
            "capabilities",
            "abilities",
            "functions",
            "features",
            "security",
            "compliance",
            "protocol",
            "reasoning",
        ]

        message_lower = user_message.lower()

        # Check for enhanced processing keywords
        if any(keyword in message_lower for keyword in enhanced_keywords):
            return True

        # Check message length (longer messages may benefit from enhanced processing)
        if len(user_message.strip()) > 100:
            return True

        # Check for question complexity (multiple question marks, complex structure)
        if message_lower.count("?") > 1:
            return True

        return False

    async def _process_with_enhanced_protocol(self, event: ConversationEvent):
        """Process conversation event through enhanced protocol."""
        try:
            # Create cognitive request
            request = CognitiveRequest(
                user_input=event.user_message,
                session_id=getattr(event, "session_id", "default"),
                expansion_tier=self._determine_expansion_tier(event.user_message),
                required_modes=self._determine_required_modes(event.user_message),
                confidence_threshold=0.7,
                compliance_required=True,
                metadata={
                    "source_event": type(event).__name__,
                    "conversation_id": getattr(event, "conversation_id", ""),
                    "timestamp": getattr(event, "timestamp", ""),
                },
            )

            # Process through enhanced protocol
            response = await self.protocol_engine.process_request(request)

            # Update statistics
            self._update_processing_stats(response)

            # Create and emit enhanced conversation response
            await self._emit_enhanced_conversation_response(event, response)

        except Exception as e:
            logger.error(f"Error in enhanced protocol processing: {e}")

    def _determine_expansion_tier(self, user_message: str) -> ExpansionTier:
        """Determine appropriate expansion tier based on message characteristics."""
        message_lower = user_message.lower()

        # Maximal tier triggers
        if any(
            keyword in message_lower
            for keyword in [
                "comprehensive",
                "detailed",
                "thorough",
                "complete",
                "full analysis",
            ]
        ):
            return ExpansionTier.MAXIMAL

        # Extended tier triggers
        if any(
            keyword in message_lower
            for keyword in ["analyze", "evaluate", "assess", "examine", "investigate"]
        ):
            return ExpansionTier.EXTENDED

        # Standard tier triggers
        if any(
            keyword in message_lower
            for keyword in ["explain", "describe", "tell me about", "what is"]
        ):
            return ExpansionTier.STANDARD

        # Default to minimal for simple queries
        return ExpansionTier.MINIMAL

    def _determine_required_modes(self, user_message: str) -> list[ReasoningMode]:
        """Determine required reasoning modes based on message content."""
        message_lower = user_message.lower()
        modes = []

        # Always include analytical
        modes.append(ReasoningMode.ANALYTICAL)

        # Dialectical mode triggers
        if any(
            keyword in message_lower
            for keyword in ["compare", "contrast", "versus", "alternative", "different"]
        ):
            modes.append(ReasoningMode.DIALECTICAL)

        # Critical mode triggers
        if any(
            keyword in message_lower
            for keyword in ["risk", "problem", "issue", "concern", "security", "safety"]
        ):
            modes.append(ReasoningMode.CRITICAL)

        # Speculative mode triggers
        if any(
            keyword in message_lower
            for keyword in ["future", "predict", "forecast", "what if", "scenario"]
        ):
            modes.append(ReasoningMode.SPECULATIVE)

        return modes

    def _update_processing_stats(self, response):
        """Update processing statistics."""
        self.processing_stats["requests_processed"] += 1

        # Update average confidence
        total_confidence = (
            self.processing_stats["average_confidence"]
            * (self.processing_stats["requests_processed"] - 1)
            + response.confidence_score
        )
        self.processing_stats["average_confidence"] = (
            total_confidence / self.processing_stats["requests_processed"]
        )

        # Update compliance rate
        if response.compliance_verified:
            compliance_count = (
                self.processing_stats["compliance_rate"]
                * (self.processing_stats["requests_processed"] - 1)
                + 1
            )
            self.processing_stats["compliance_rate"] = (
                compliance_count / self.processing_stats["requests_processed"]
            )

    async def _emit_enhanced_conversation_response(
        self, original_event: ConversationEvent, protocol_response
    ):
        """Emit enhanced conversation response event."""
        try:
            # Format enhanced response for conversation
            formatted_response = self._format_protocol_response(protocol_response)

            # Create conversation message event
            response_event = ConversationMessageEvent(
                source_plugin=self.name,
                conversation_id=getattr(original_event, "conversation_id", ""),
                session_id=getattr(original_event, "session_id", ""),
                user_message=original_event.user_message,
                assistant_message=formatted_response,
                metadata={
                    "enhanced_protocol": True,
                    "protocol_version": "2.0",
                    "confidence_score": protocol_response.confidence_score,
                    "expansion_tier": protocol_response.expansion_tier.value,
                    "active_modes": [
                        mode.value for mode in protocol_response.active_modes
                    ],
                    "compliance_verified": protocol_response.compliance_verified,
                    "processing_time": protocol_response.processing_time,
                },
            )

            await self.event_bus.publish(response_event)

        except Exception as e:
            logger.error(f"Error emitting enhanced conversation response: {e}")

    async def _emit_enhanced_response(
        self, original_event: BaseEvent, protocol_response
    ):
        """Emit enhanced protocol response event."""
        try:
            # Create enhanced response event (would need to define this event type)
            response_data = {
                "source_plugin": self.name,
                "original_event_id": getattr(original_event, "id", ""),
                "protocol_response": asdict(protocol_response),
                "processing_stats": self.processing_stats,
            }

            # For now, log the response (would emit actual event in full implementation)
            logger.info(f"Enhanced protocol response: {response_data}")

        except Exception as e:
            logger.error(f"Error emitting enhanced response: {e}")

    def _format_protocol_response(self, protocol_response) -> str:
        """Format protocol response for conversation output."""
        try:
            # Start with supplemental banner
            formatted = (
                f"{protocol_response.metadata.get('supplemental_banner', '')}\n\n"
            )

            # Add executive summary
            if protocol_response.executive_summary:
                formatted += (
                    f"## Executive Summary\n{protocol_response.executive_summary}\n\n"
                )

            # Add reasoning trace if available
            if protocol_response.reasoning_trace:
                formatted += "## Reasoning Trace\n"
                for trace_item in protocol_response.reasoning_trace:
                    formatted += f"{trace_item}\n"
                formatted += "\n"

            # Add confidence and compliance information
            formatted += "## Processing Details\n"
            formatted += (
                f"- **Confidence Score:** {protocol_response.confidence_score:.2f}\n"
            )
            formatted += f"- **Compliance Verified:** {'✅' if protocol_response.compliance_verified else '❌'}\n"
            formatted += (
                f"- **Expansion Tier:** {protocol_response.expansion_tier.value}\n"
            )
            formatted += f"- **Active Reasoning Modes:** {', '.join(mode.value for mode in protocol_response.active_modes)}\n"
            formatted += (
                f"- **Processing Time:** {protocol_response.processing_time:.3f}s\n"
            )

            # Add metadata if available
            if protocol_response.metadata:
                formatted += "\n## Additional Analysis\n"
                for key, value in protocol_response.metadata.items():
                    if key not in [
                        "supplemental_banner",
                        "protocol_version",
                        "legal_authority",
                    ]:
                        formatted += f"- **{key.replace('_', ' ').title()}:** {value}\n"

            return formatted

        except Exception as e:
            logger.error(f"Error formatting protocol response: {e}")
            return f"Enhanced protocol processing completed with confidence {protocol_response.confidence_score:.2f}"

    async def get_plugin_status(self) -> dict[str, Any]:
        """Get current plugin status and statistics."""
        status = {
            "plugin_name": self.name,
            "protocol_engine_status": (
                "initialized" if self.protocol_engine else "not_initialized"
            ),
            "processing_stats": self.processing_stats,
        }

        if self.protocol_engine:
            status["protocol_status"] = self.protocol_engine.get_protocol_status()

        return status
