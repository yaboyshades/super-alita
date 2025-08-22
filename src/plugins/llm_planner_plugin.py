#!/usr/bin/env python3
"""
LLMPlannerPlugin - Enhanced with Cognitive Turn Processing

This plugin provides sophisticated decision-making for tool routing by
leveraging LLM reasoning to analyze user requests and intelligently select
the appropriate tools and actions. Now enhanced with DTA 2.0 cognitive
turn capabilities for structured reasoning and planning.

Key Features:
- LLM-powered semantic analysis of user requests
- Dynamic tool discovery and selection
- Contextual decision making with conversation history
- Cognitive turn processing for enhanced reasoning
- Fallback handling for ambiguous requests
- Integration with existing tool ecosystem

Architecture:
1. User request → ConversationEvent
2. LLMPlannerPlugin analyzes intent and context
3. Processes through cognitive turn methodology
4. Makes intelligent routing decision: TOOL vs NONE
5. Emits ToolCallEvent for tool use or passes to conversation
6. Tracks success/failure for continuous improvement

Version: 2.0.0 (DTA 2.0 Enhanced)
Author: Super Alita Agent
"""

import asyncio
import logging
import os
from datetime import UTC, datetime
from typing import Any

try:
    import google.generativeai as genai

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    genai = None

from src.core.event_bus import EventBus
from src.core.events import (
    AgentResponseEvent,
    PreprocessedActionEvent,
    ToolCallEvent,
)
from src.core.neural_atom import NeuralStore
from src.core.plugin_interface import PluginInterface

# DTA 2.0 Cognitive Turn imports
try:
    from src.dta.cognitive_plan import generate_master_cognitive_plan

    # from src.dta.config import DTAConfig  # Currently unused
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

# Event imports (including cognitive turns)
try:
    from src.core.events import CognitiveTurnCompletedEvent

    COGNITIVE_EVENT_AVAILABLE = True
except ImportError:
    COGNITIVE_EVENT_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMPlannerPlugin(PluginInterface):
    """
    Provides intelligent tool routing through LLM-based semantic analysis.

    This plugin acts as the "smart dispatcher" that analyzes user requests
    and makes contextual decisions about which tools to invoke. It uses
    sophisticated prompt engineering to enable the LLM to understand the
    available tools and make intelligent routing decisions.

    Decision Flow:
    1. Receive ConversationEvent with user message
    2. Analyze message context and intent using LLM
    3. Consider available tools and their capabilities
    4. Make routing decision: TOOL (with parameters) or NONE
    5. Emit appropriate events and track performance

    LLM Integration:
    - Uses Gemini for semantic analysis and decision making
    - Sophisticated prompt engineering for tool awareness
    - Context injection from conversation history
    - Error handling with graceful fallbacks
    """

    def __init__(self):
        super().__init__()
        self._config: dict[str, Any] = {}
        self._llm_client = None
        self._available_tools: dict[str, dict[str, Any]] = {}

        # Performance tracking
        self._stats = {
            "decisions_made": 0,
            "tools_selected": 0,
            "conversations_handled": 0,
            "errors": 0,
            "tool_success_rate": {},
            "last_activity": None,
        }

        # Tool definitions for LLM awareness
        self._tool_definitions = {
            "memory_manager": {
                "name": "memory_manager",
                "description": "Manages user memories, notes, and knowledge storage",
                "actions": {
                    "save": "Store new information or memories",
                    "list": "Retrieve and list stored memories",
                    "recall": "Search memories by semantic similarity",
                },
                "parameters": {
                    "action": "Required: 'save', 'list', or 'recall'",
                    "content": "Required for save: the content to store",
                    "query": "Required for recall: search query",
                    "hierarchy": "Optional: categorization path",
                    "limit": "Optional: max results for list/recall",
                },
                "examples": [
                    "Remember that John likes coffee",
                    "What do I know about machine learning?",
                    "List my recent notes",
                ],
            },
            "web_agent": {
                "name": "web_agent",
                "description": "Searches the web and GitHub for current information",
                "actions": {
                    "search": "Search the web for current information",
                    "github_search": "Search GitHub repositories and code",
                },
                "parameters": {
                    "query": "Required: search query",
                    "search_type": "Optional: 'web' or 'github'",
                    "max_results": "Optional: maximum results to return",
                },
                "examples": [
                    "What's the latest news about AI?",
                    "Find Python libraries for data visualization",
                    "Search for React component examples",
                ],
            },
            "calculator": {
                "name": "calculator",
                "description": "Evaluates mathematical expressions and performs calculations",
                "actions": {"calculate": "Evaluate arithmetic expressions safely"},
                "parameters": {
                    "expression": "Required: mathematical expression to evaluate",
                    "expr": "Alternative: same as expression",
                    "input": "Alternative: same as expression",
                },
                "examples": [
                    "Calculate 2 + 3 * 4",
                    "What is 15% of 200?",
                    "Compute the square root of 64",
                    "Evaluate (10 + 5) * 2",
                ],
            },
        }

    @property
    def name(self) -> str:
        return "llm_planner"

    @property
    def version(self) -> str:
        return "1.0.0"

    async def setup(
        self, event_bus: EventBus, store: NeuralStore, config: dict[str, Any]
    ):
        """Setup the LLM planner plugin with Gemini configuration."""
        await super().setup(event_bus, store, config)
        self._config = config.get(self.name, {})

        # Apply configuration defaults
        self._config.setdefault("llm_model", "gemini-2.5-pro")
        self._config.setdefault("max_tokens", 1000)
        self._config.setdefault(
            "temperature", 0.1
        )  # Low temperature for consistent decisions
        self._config.setdefault("enable_context_injection", True)
        self._config.setdefault("max_context_messages", 5)
        self._config.setdefault("llm_timeout", 10.0)  # 10 second timeout for LLM calls

        # Configure Gemini API
        if not HAS_GEMINI:
            logger.error(
                "google-generativeai not available. LLM planner will be disabled."
            )
            return

        # Get API key with robust template handling (matches conversation_plugin.py)
        api_key = self._config.get("gemini_api_key")

        # Handle environment variable substitution
        if api_key and api_key.startswith("${") and api_key.endswith("}"):
            env_var_name = api_key[2:-1]  # Remove ${ and }
            api_key = os.environ.get(env_var_name)

        # Also check directly from environment if not in config
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            logger.error("Gemini API key not configured. LLM planner will be disabled.")
            return

        try:
            genai.configure(api_key=api_key)
            self._llm_client = genai.GenerativeModel(self._config["llm_model"])
            logger.info(
                f"LLM planner configured with model: {self._config['llm_model']}"
            )

            # Test LLM connection
            await self._test_llm_connection()

        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")

        logger.info("LLMPlannerPlugin setup complete")

    async def start(self) -> None:
        """Start the plugin and subscribe to preprocessed action events."""
        logger.info("Starting LLMPlannerPlugin...")

        # Note: In the new preprocessor architecture, we don't need the LLM client
        # since all complex reasoning is handled by the PythonicPreprocessorPlugin.
        # This plugin now just executes simple routing based on structured actions.

        # Subscribe to preprocessed actions - this is our primary event source
        await self.subscribe("preprocessed_action", self._handle_preprocessed_action)

        await super().start()

        if self._llm_client:
            logger.info(
                "LLMPlannerPlugin started successfully - ready to execute preprocessed actions (LLM available)"
            )
        else:
            logger.info(
                "LLMPlannerPlugin started successfully - ready to execute preprocessed actions (LLM not available, but not required)"
            )

    async def shutdown(self) -> None:
        """Graceful shutdown with statistics export."""
        logger.info("Shutting down LLMPlannerPlugin...")

        # Log final statistics
        final_stats = self._get_stats()
        logger.info(f"Final LLM planner statistics: {final_stats}")

        self._llm_client = None

        logger.info("LLMPlannerPlugin shutdown complete")

    async def _test_llm_connection(self):
        """Test LLM connection with a simple query."""
        try:
            test_prompt = "Respond with exactly 'OK' if you can understand this."
            response = await asyncio.wait_for(
                asyncio.to_thread(self._llm_client.generate_content, test_prompt),
                timeout=10.0,
            )

            if response and response.text:
                logger.info("LLM connection test successful")
            else:
                logger.warning("LLM connection test returned empty response")

        except Exception as e:
            logger.error(f"LLM connection test failed: {e}")

    async def _handle_preprocessed_action(self, event: PreprocessedActionEvent) -> None:
        """
        Enhanced handler for preprocessed actions with cognitive turn processing.

        This method receives structured action objects and routes them to appropriate
        tools while incorporating cognitive turn insights for enhanced decision-making.
        """
        logger.info(f"Processing preprocessed action for session {event.session_id}")

        try:
            # Extract action details from the action dictionary
            action_data = event.action
            action_type = action_data.get("action_type")

            logger.info(f"Action type: {action_type}")

            # Process through cognitive turn if available
            if COGNITIVE_TURN_AVAILABLE and hasattr(event, "user_message"):
                cognitive_turn = await self._process_cognitive_turn_for_action(
                    event.user_message, action_data, event.session_id
                )

                # Emit cognitive turn completed event
                if (
                    COGNITIVE_EVENT_AVAILABLE
                    and cognitive_turn
                    and hasattr(self.event_bus, "publish")
                ):
                    turn_event = CognitiveTurnCompletedEvent(
                        source_plugin=self.name,
                        turn_record=cognitive_turn,
                        session_id=event.session_id,
                        conversation_id=getattr(
                            event, "conversation_id", event.session_id
                        ),
                    )
                    await self.event_bus.publish(turn_event)
                    logger.info(
                        f"Published cognitive turn for action routing in session {event.session_id}"
                    )

            # Route based on action type
            if action_type == "use_tool":
                # Direct tool execution with preprocessed parameters
                tool_name = action_data.get("tool_name")
                parameters = action_data.get("parameters", {})

                await self._emit_tool_call(
                    session_id=event.session_id,
                    tool_name=tool_name,
                    parameters=parameters,
                )
                self._stats["tools_selected"] += 1

                # Track tool usage for success rate analysis
                if tool_name not in self._stats["tool_success_rate"]:
                    self._stats["tool_success_rate"][tool_name] = {
                        "calls": 0,
                        "successes": 0,
                    }
                self._stats["tool_success_rate"][tool_name]["calls"] += 1

            elif action_type == "need_tool":
                # Tool gap detected - emit gap event for CREATOR pipeline
                gap_description = action_data.get(
                    "description",
                    action_data.get("parameters", {}).get(
                        "description", "Unknown capability gap"
                    ),
                )
                logger.info(f"🔧 Tool gap detected by preprocessor: {gap_description}")

                # Import here to avoid circular imports
                import uuid

                from src.core.events import AtomGapEvent

                gap_event = AtomGapEvent(
                    source_plugin=self.name,
                    missing_tool=gap_description,
                    description=gap_description,
                    session_id=event.session_id,
                    conversation_id=event.conversation_id,
                    gap_id=f"gap_{uuid.uuid4().hex[:8]}",
                )

                await self.event_bus.publish(gap_event)
                self._stats["gaps_detected"] = self._stats.get("gaps_detected", 0) + 1

                # Also send a response to the user
                await self._reply_to_user(
                    session_id=event.session_id,
                    message=f"I understand you need: {gap_description}\n\nI don't currently have a tool for that, but I'm working on creating one! This might take a moment while I design, generate, and test the code. I'll let you know once it's ready.",
                )

            elif action_type == "chat":
                # Handle conversationally - emit response event
                response_message = action_data.get(
                    "response",
                    action_data.get("parameters", {}).get(
                        "response", "I understand. How can I help you with that?"
                    ),
                )
                await self._reply_to_user(
                    session_id=event.session_id, message=response_message
                )
                self._stats["conversations_handled"] += 1

            elif action_type == "complex_task":
                # Handle complex multi-step tasks (future enhancement)
                task_description = action_data.get("description", "Complex task")
                logger.info(f"Complex task detected: {task_description}")
                await self._reply_to_user(
                    session_id=event.session_id,
                    message="I recognize this as a complex task. Multi-step task handling is coming soon!",
                )

            else:
                # Unknown action type - handle conversationally
                logger.warning(f"Unknown action type: {action_type}")
                await self._reply_to_user(
                    session_id=event.session_id,
                    message="I processed your request but encountered an unexpected action type. Let me handle this conversationally.",
                )

            self._stats["decisions_made"] += 1
            self._stats["last_activity"] = datetime.now(UTC).isoformat()

        except Exception as e:
            logger.error(f"Error processing preprocessed action: {e}", exc_info=True)
            self._stats["errors"] += 1

            # Graceful fallback response
            await self._reply_to_user(
                session_id=event.session_id,
                message="I encountered an issue processing your request. Please try again.",
            )

    async def _process_cognitive_turn_for_action(
        self, user_message: str, action_data: dict[str, Any], session_id: str
    ) -> CognitiveTurnRecord | None:
        """Process action routing through cognitive turn methodology."""

        try:
            action_type = action_data.get("action_type", "unknown")
            tool_name = action_data.get("tool_name", "none")

            # Build cognitive turn analysis for action routing
            turn_data = {
                "state_readout": f"Routing action for user request: {user_message}. Action type: {action_type}",
                "activation_protocol": {
                    "pattern_recognition": "routing",
                    "confidence_score": 9,
                    "planning_requirement": action_type == "need_tool",
                    "quality_speed_tradeoff": "speed",
                    "evidence_threshold": "low",
                    "audience_level": "professional",
                    "meta_cycle_check": "routing",
                },
                "strategic_plan": {"is_required": action_type == "need_tool"},
                "execution_log": [
                    "Action routing initiated",
                    f"Action type determined: {action_type}",
                    (
                        f"Tool selection: {tool_name}"
                        if tool_name != "none"
                        else "No tool required"
                    ),
                ],
                "synthesis": {
                    "key_findings": [
                        f"User request: {user_message}",
                        f"Action routing: {action_type}",
                        (
                            f"Tool assignment: {tool_name}"
                            if tool_name != "none"
                            else "Conversational response"
                        ),
                    ],
                    "counterarguments": [],
                    "final_answer_summary": f"Action routing complete: {action_type}",
                },
                "state_update": {
                    "directive": "memory_stream_add",
                    "memory_stream_add": {
                        "summary": f"Routed action {action_type} for: {user_message}",
                        "timestamp": datetime.now().isoformat(),
                        "type": "action_routing",
                    },
                },
                "confidence_calibration": {
                    "final_confidence": 9,
                    "uncertainty_gaps": "Minimal uncertainty in action routing",
                    "risk_assessment": "low",
                    "verification_methods": ["action_analysis", "tool_matching"],
                },
            }

            # Create cognitive turn record
            turn_record = CognitiveTurnRecord(**turn_data)

            # Generate strategic plan if tool gap detected
            if action_type == "need_tool":
                plan = generate_master_cognitive_plan(
                    f"Create tool for: {user_message}"
                )
                turn_record.strategic_plan = StrategicPlan(
                    is_required=True,
                    steps=plan.execution_steps,
                    estimated_duration=120.0,
                    resource_requirements=["tool_creation", "creator_pipeline"],
                )

            logger.info(
                f"Generated cognitive turn for action routing with confidence: {turn_record.confidence_calibration.final_confidence}"
            )
            return turn_record

        except Exception as e:
            logger.error(f"Error in cognitive turn action processing: {e}")
            return None

    # Note: The complex LLM analysis methods (_analyze_message_for_routing,
    # _build_routing_prompt, _parse_llm_decision) have been removed as part of
    # the migration to the PythonicPreprocessorPlugin architecture. Intent
    # analysis and parsing is now handled upstream by the preprocessor.

    async def _emit_tool_call(
        self, session_id: str, tool_name: str, parameters: dict[str, Any]
    ) -> None:
        """
        Emit a ToolCallEvent for the selected tool.

        This is CRITICAL for the tool execution flow.
        """
        try:
            import uuid

            tool_call_event = ToolCallEvent(
                source_plugin=self.name,
                tool_name=tool_name,
                parameters=parameters,
                conversation_id=session_id,  # Use session_id as conversation_id
                session_id=session_id,
                tool_call_id=f"llm_planner_{uuid.uuid4().hex[:8]}",
            )

            await self.event_bus.publish(tool_call_event)
            logger.info(
                f"Emitted tool call for {tool_name} with parameters: {parameters}"
            )

        except Exception as e:
            logger.error(f"Failed to emit tool call: {e}", exc_info=True)

    async def _reply_to_user(self, session_id: str, message: str) -> None:
        """
        Emit a conversational response when no tool is needed.
        """
        try:
            import uuid

            response_event = AgentResponseEvent(
                source_plugin=self.name,
                session_id=session_id,
                response_text=message,
                response_id=f"llm_planner_{uuid.uuid4().hex[:8]}",
            )

            await self.event_bus.publish(response_event)
            logger.debug(f"Emitted conversational response: {message[:50]}...")

        except Exception as e:
            logger.error(f"Failed to emit conversational response: {e}", exc_info=True)

    def _get_stats(self) -> dict[str, Any]:
        """Get plugin statistics."""
        return {
            **self._stats,
            "plugin": self.name,
            "version": self.version,
            "is_running": self.is_running,
            "tools_available": list(self._tool_definitions.keys()),
            "llm_model": self._config.get("llm_model", "unknown"),
            "llm_available": self._llm_client is not None,
        }

    async def health_check(self) -> dict[str, Any]:
        """Health check for the LLM planner plugin."""
        health = {
            "status": "healthy" if self.is_running else "stopped",
            "version": self.version,
            "issues": [],
            "stats": self._get_stats(),
        }

        try:
            # Check LLM availability
            if not HAS_GEMINI:
                health["status"] = "degraded"
                health["issues"].append("google-generativeai not installed")
            elif not self._llm_client:
                health["status"] = "degraded"
                health["issues"].append("LLM client not configured")
            else:
                health["llm_status"] = "available"

            # Check tool definitions
            if not self._tool_definitions:
                health["status"] = "degraded"
                health["issues"].append("No tool definitions available")
            else:
                health["tools_count"] = len(self._tool_definitions)

            # Test basic functionality if running
            if self.is_running and self._llm_client:
                try:
                    # Quick test of decision parsing
                    test_response = "NONE: Test response"
                    parsed = self._parse_llm_decision(test_response)
                    if parsed["action"] == "NONE":
                        health["parsing"] = "functional"
                    else:
                        health["issues"].append(
                            "Decision parsing not working correctly"
                        )
                except Exception as e:
                    health["issues"].append(f"Parsing test failed: {e}")

        except Exception as e:
            health["status"] = "unhealthy"
            health["issues"].append(f"Health check failed: {e}")

        return health
