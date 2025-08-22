#!/usr/bin/env python3

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not available, environment variables should be set manually

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

# FIXED: Cleaner GenAI import with better error handling
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
    _genai_import_error = None
except ImportError as e:
    genai = None
    GEMINI_AVAILABLE = False
    _genai_import_error = str(e)

# Import structured events for clean publishing
from src.core.events import AgentReplyEvent, CognitiveTurnInitiatedEvent
from src.core.plugin_interface import PluginInterface
from src.orchestration.dispatcher import Dispatcher
from src.orchestration.router import Router

# NEW: Import REUGExecutionFlow for SoT integration
try:
    from src.core.execution_flow import REUGExecutionFlow
except Exception as _e:
    REUGExecutionFlow = None  # type: ignore

# NEW: Tool registry
try:
    from src.mcp.registry import ToolRegistry
except Exception as _e:
    ToolRegistry = None  # type: ignore

"""
Conversation Plugin for Super Alita
Handles natural language conversations using Redis-backed event bus
"""

logger = logging.getLogger(__name__)

# Global duplicate message tracking
_last_hash = {}  # Track last message hash by session
_DUP_WINDOW = 5  # Duplicate detection window in seconds


def _contains_creation_intent(text: str) -> bool:
    """Check if text contains tool creation keywords."""
    t = text.lower()
    return any(
        kw in t
        for kw in (
            "create a tool",
            "build a tool",
            "make a tool",
            "generate a tool",
            "create tool",
            "build tool",
            "make tool",
            "generate tool",
        )
    )


def _minimal_tool_descriptions(reg) -> str:
    """Get minimal tool descriptions from registry."""
    try:
        if hasattr(reg, "list_tools"):
            names = reg.list_tools()
        elif hasattr(reg, "get_all_tools"):
            tools = reg.get_all_tools()
            names = list(tools.keys()) if tools else []
        else:
            names = []

        if not names:
            return "(no tools registered)"
        return "\n".join(f"- {n}: callable tool" for n in names)
    except Exception:
        return "(tool registry unavailable)"


class ConversationPlugin(PluginInterface):
    """
    Plugin that handles natural language conversations with users via Redis event bus.

    Features:
    - Natural conversation processing
    - Context-aware responses
    - Redis-backed event communication
    - Multi-turn dialogue management
    """

    def __init__(self):
        super().__init__()
        self.active_sessions: dict[str, dict] = {}
        self.conversation_memory = []
        # THE CRITICAL FIX: Add a lock to ensure one-at-a-time processing
        self._conversation_lock = asyncio.Lock()
        self._selected_tools = []  # NEW: Store selected tools from router
        self._genai_setup_attempted = False  # Track setup attempts
        self.response_templates = {
            "greeting": [
                "Hello! I'm Super Alita, your neuro-symbolic cognitive agent. How can I help you today?",
                "Hi there! I'm ready to chat, analyze, learn, and help with any questions you have.",
                "Greetings! I'm Super Alita - let's have an engaging conversation!",
            ],
            "farewell": [
                "It was great chatting with you! Feel free to come back anytime.",
                "Goodbye! I enjoyed our conversation and learned from it.",
                "Take care! I'll be here whenever you want to chat again.",
            ],
            "thinking": [
                "Let me think about that...",
                "Interesting question, processing...",
                "Analyzing your request...",
            ],
        }

    @property
    def name(self) -> str:
        return "conversation"

    async def setup(
        self, event_bus_or_workspace, store, config: dict[str, Any]
    ) -> None:
        """Initialize the conversation plugin."""
        # Handle both legacy (event_bus) and unified (workspace) architectures
        if hasattr(event_bus_or_workspace, "subscribe") and hasattr(
            event_bus_or_workspace, "update"
        ):
            # This is a workspace (unified architecture)
            self.workspace = event_bus_or_workspace
            await super().setup(None, store, config)  # No event_bus in unified arch
        else:
            # This is an event_bus (legacy architecture)
            await super().setup(event_bus_or_workspace, store, config)

        # Get configuration
        self.max_context_messages = config.get("max_context_messages", 10)
        self.response_delay = config.get("response_delay_seconds", 1.0)
        self.enable_reasoning = config.get("enable_reasoning", True)

        # Initialize orchestration components FIRST (before Gemini setup that might return early)
        # Router should ALWAYS be available since it's a simple class
        try:
            self.router = Router()
            self.router_ready = True
            logger.info(
                "Router initialized for deterministic planner → router → dispatcher flow"
            )
        except Exception as e:
            logger.error(f"Failed to initialize router: {e}")
            self.router = None
            self.router_ready = False

        # Initialize REUGExecutionFlow for SoT integration
        try:
            if REUGExecutionFlow is not None:
                # REUGExecutionFlow needs event_bus and plugin_registry
                # Use a simplified plugin registry for now
                plugin_registry = getattr(self, "plugin_registry", {})
                if hasattr(self, "event_bus") and self.event_bus:
                    self.reug_flow = REUGExecutionFlow(self.event_bus, plugin_registry)
                    self.reug_ready = True
                    logger.info("REUGExecutionFlow initialized for SoT integration")
                else:
                    self.reug_flow = None
                    self.reug_ready = False
                    logger.warning(
                        "REUGExecutionFlow requires event_bus - SoT disabled"
                    )
            else:
                self.reug_flow = None
                self.reug_ready = False
                logger.warning("REUGExecutionFlow not available - SoT disabled")
        except Exception as e:
            logger.error(f"Failed to initialize REUGExecutionFlow: {e}")
            self.reug_flow = None
            self.reug_ready = False

        # Initialize registry for router/dispatcher functionality
        self._init_router_and_registry()

        # Load system prompts
        self.router_prompt = self._load_prompt("router_system_prompt.txt")
        self.finalizer_prompt = self._load_prompt(
            "conversation_finalizer_system_prompt.txt"
        )

        # Initialize Gemini Pilot client
        try:
            from src.core.gemini_pilot import GeminiPilotClient

            self.pilot_client = GeminiPilotClient()
            logger.info("Gemini Pilot client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini Pilot client: {e}")
            self.pilot_client = None

        # Keep legacy Gemini client for direct chat responses
        try:
            if not GEMINI_AVAILABLE:
                if not self._genai_setup_attempted:
                    logger.info(
                        "Gemini not available - LLM features disabled. Install with: pip install google-generativeai"
                    )
                    if _genai_import_error:
                        logger.debug(
                            f"GenAI import error details: {_genai_import_error}"
                        )
                    self._genai_setup_attempted = True
                self.llm_client = None
                return  # Skip the rest of LLM setup

            gemini_api_key = config.get("gemini_api_key")

            # Handle environment variable substitution
            if (
                gemini_api_key
                and gemini_api_key.startswith("${")
                and gemini_api_key.endswith("}")
            ):
                env_var_name = gemini_api_key[2:-1]  # Remove ${ and }
                gemini_api_key = os.environ.get(env_var_name)

            # Also check directly from environment if not in config
            if not gemini_api_key:
                gemini_api_key = os.environ.get("GEMINI_API_KEY")

            if not gemini_api_key:
                if not self._genai_setup_attempted:
                    logger.info(
                        "GEMINI_API_KEY not found - LLM features disabled. Set API key to enable full capabilities."
                    )
                    self._genai_setup_attempted = True
                self.llm_client = None
                return

            genai.configure(api_key=gemini_api_key)

            # Use model from config, default to gemini-2.5-pro
            model_name = config.get("llm_model", "gemini-2.5-pro")
            self.llm_client = genai.GenerativeModel(model_name)
            logger.info(
                f"Gemini LLM client initialized successfully with model: {model_name}"
            )

        except Exception as e:
            if not self._genai_setup_attempted:
                logger.warning(f"Failed to initialize Gemini client: {e}")
                self._genai_setup_attempted = True
            self.llm_client = None

        logger.info("Conversation plugin setup complete")

    def _init_router_and_registry(self) -> None:
        """
        Initialize router hooks and ToolRegistry if available.
        This is resilient: it won't crash if orchestration or registry imports fail.
        """
        # Registry: prefer an application-provided instance if present
        # (e.g., injected via kwargs or global). Otherwise create a lightweight one.
        try:
            # Prefer injected
            inj = getattr(self, "injected_registry", None)
            if inj is not None:
                self.registry = inj
            elif ToolRegistry is not None:
                # Default on-disk registry path for persistence
                self.registry = ToolRegistry()
            else:
                self.registry = None
        except Exception as e:
            logger.warning("Conversation: failed to init ToolRegistry: %s", e)
            self.registry = None

        # Router availability flag - don't override if already set properly
        if not hasattr(self, "router_ready"):
            self.router_ready = hasattr(self, "router") and self.router is not None

        if self.router_ready:
            logger.info(
                "Conversation: router initialized (planner+dispatcher available)"
            )
        else:
            logger.warning(
                "Conversation: router unavailable (planner/dispatcher import failed)"
            )

    def _load_prompt(self, filename: str) -> str:
        """Load prompt template from file."""
        try:
            prompt_path = os.path.join(
                os.path.dirname(__file__), "..", "prompts", filename
            )
            with open(prompt_path, encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning(f"Prompt file not found: {filename}")
            return ""

    async def start(self) -> None:
        """Start the conversation plugin."""
        await super().start()

        # Subscribe to conversation events
        logger.debug("🎯 CONVERSATION: About to subscribe to conversation_message")
        await self.subscribe("conversation_message", self._handle_conversation)
        logger.debug("🎯 CONVERSATION: About to subscribe to user_message")
        await self.subscribe(
            "user_message", self._handle_user_message
        )  # Fix: subscribe to chat client events
        logger.debug("🎯 CONVERSATION: About to subscribe to system_status_request")
        await self.subscribe("system_status_request", self._handle_status_request)
        logger.debug("🎯 CONVERSATION: All subscriptions completed")
        # REMOVED: goal_received subscription - let planner handle goals exclusively

        logger.info("Conversation plugin started - subscribed to events:")
        logger.info("   - conversation_message")
        logger.info("   - user_message")  # Add this to the log
        logger.info("   - system_status_request")
        logger.info("Conversation plugin started - ready for natural dialogue")

    async def shutdown(self) -> None:
        """Shutdown the conversation plugin."""
        logger.info("Shutting down conversation plugin")

        # Save any important conversation data
        if self.conversation_memory:
            logger.info(f"Processed {len(self.conversation_memory)} conversation turns")

    async def _publish_agent_reply(
        self, text: str, correlation_id: str | None = None, **kwargs
    ):
        """Publish structured agent reply event."""
        evt = AgentReplyEvent(
            source_plugin="conversation",
            text=text,
            message=text,  # For backward compatibility
            correlation_id=correlation_id,
            session_id=kwargs.get("session_id"),
            conversation_id=kwargs.get("conversation_id"),
        )
        await self.event_bus.publish_event("agent_reply", evt)
        logger.info(f"Published agent_reply: {text[:50]}...")

    async def _emit_gap_response(
        self, gap_text: str, session_id: str | None = None, **kwargs
    ):
        """Emit GAP response with proper structure."""
        await self._publish_agent_reply(
            text=gap_text,
            correlation_id=session_id,
            session_id=session_id,
            conversation_id=kwargs.get("conversation_id"),
        )
        logger.info("Published agent_reply GAP (keyword)")

    async def _handle_user_message(self, event) -> None:
        """Handle simple user messages from chat client using deterministic router."""
        logger.debug(
            f"🎯 CONVERSATION: _handle_user_message called with event: {event}"
        )
        logger.debug(f"🎯 CONVERSATION: Event type: {type(event)}")
        try:
            # The event might be a raw Redis message, not a pydantic object
            if hasattr(event, "model_dump"):
                data = event.model_dump()
                logger.debug(f"🎯 CONVERSATION: Using model_dump: {data}")
            elif hasattr(event, "__dict__"):
                data = event.__dict__
                logger.debug(f"🎯 CONVERSATION: Using __dict__: {data}")
            elif isinstance(event, dict):
                data = event
                logger.debug(f"🎯 CONVERSATION: Using dict directly: {data}")
            else:
                # Raw message data from Redis
                import json

                try:
                    data = json.loads(str(event))
                    logger.debug(f"🎯 CONVERSATION: Parsed JSON: {data}")
                except (json.JSONDecodeError, ValueError):
                    logger.error(f"Could not parse user message: {event}")
                    return

            user_text = data.get("text", "")
            session_id = data.get("session_id") or "default_session"

            logger.debug(
                f"🎯 CONVERSATION: Extracted user_text: '{user_text}', session_id: '{session_id}'"
            )

            # Skip non-conversation events that don't have text
            event_type = data.get("type", "")
            if event_type in [
                "communication_test",
                "process_stage",
                "task_request",
                "task_result",
            ]:
                # These are test/system events, not conversation messages
                logger.debug(
                    f"🎯 CONVERSATION: Skipping system event type: {event_type}"
                )
                return

            if not user_text:
                logger.warning(f"Received user_message with no text: {data}")
                return

            # Ignore empty or whitespace-only messages to prevent lock collisions
            if not user_text.strip():
                return  # skip completely

            # Deterministic tool creation path for generic "create a tool" requests
            import re
            import uuid

            from src.core.events import AgentResponseEvent, ComposeRequestEvent

            create_tool_patterns = [
                r"create\s+(?:a\s+)?tool",
                r"make\s+(?:a\s+)?tool",
                r"build\s+(?:a\s+)?tool",
                r"generate\s+(?:a\s+)?tool",
                r"new\s+tool",
            ]

            user_text_lower = user_text.lower()
            for pattern in create_tool_patterns:
                if re.search(pattern, user_text_lower):
                    # Extract tool name and description from the user message
                    tool_name = "generic_tool"
                    description = "Generic tool created from user request"

                    # Try to extract specific tool name or description
                    # Look for patterns like "create a fibonacci tool" or "make a calculator"
                    name_match = re.search(
                        r"(?:create|make|build|generate|new)\s+(?:a\s+)?(\w+)\s+tool",
                        user_text_lower,
                    )
                    if name_match:
                        tool_name = name_match.group(1).strip()
                        description = (
                            f"{tool_name.title()} tool created from user request"
                        )
                    elif "fibonacci" in user_text_lower:
                        tool_name = "fibonacci_tool"
                        description = "Fibonacci sequence calculator tool"
                    elif "calculator" in user_text_lower or "math" in user_text_lower:
                        tool_name = "calculator_tool"
                        description = "Mathematical calculator tool"
                    elif "text" in user_text_lower or "string" in user_text_lower:
                        tool_name = "text_processor"
                        description = "Text processing tool"

                    # Emit compose_request event for deterministic tool creation
                    try:
                        compose_event = ComposeRequestEvent(
                            source_plugin=self.name,
                            goal=f"Create {tool_name}",
                            params={
                                "tool_name": tool_name,
                                "description": description,
                                "user_request": user_text,
                            },
                        )

                        await self.event_bus.publish(compose_event)
                        logger.info(
                            f"💡 CONVERSATION: Emitted compose_request for deterministic tool creation: {tool_name}"
                        )

                        # Send immediate feedback to user
                        response_event = AgentResponseEvent(
                            source_plugin=self.name,
                            session_id=session_id,
                            response_text=f"🔧 Creating {tool_name}... I'll generate this tool for you.",
                            response_id=str(uuid.uuid4()),
                            reasoning=f"Detected request to create a tool: {tool_name}",
                        )
                        await self.event_bus.publish(response_event)

                        return  # Skip normal LLM processing for deterministic tool creation

                    except Exception as e:
                        logger.error(
                            f"❌ CONVERSATION: Failed to emit compose_request: {e}"
                        )
                        # Fall through to normal processing if deterministic path fails

            # Handle special commands - INTERCEPT BEFORE CONVERSATION HANDLER
            if user_text.strip() == "/forget-session":
                # Clear session memory
                session_keys = list(self.active_sessions.keys())
                for key in session_keys:
                    if key in self.active_sessions:
                        del self.active_sessions[key]

                # Clear conversation memory
                self.conversation_memory.clear()

                await self.event_bus._redis.publish(
                    "agent_reply",
                    json.dumps({"text": "Session wiped. Starting fresh conversation."}),
                )
                logger.info("Session memory cleared by user request")
                return

            # CRITICAL FIX: Parse /create-atom and /atom-run commands
            if user_text.strip().startswith("/create-atom"):
                await self._handle_create_atom_command(user_text.strip())
                return

            if user_text.strip().startswith("/atom-run"):
                await self._handle_atom_run_command(user_text.strip())
                return

            logger.info(f"Received user message: '{user_text}'")

            # DETERMINISTIC FLOW: planner → router → dispatcher
            conversation_id = data.get("conversation_id") or session_id
            await self._process_message_deterministically(
                user_text, session_id, conversation_id
            )

        except Exception as e:
            logger.error(f"Error handling user message: {e}")
            # Send error response
            import json

            await self.event_bus._redis.publish(
                "agent_reply",
                json.dumps(
                    {"text": "Sorry, I encountered an error processing your message."}
                ),
            )

    async def _process_message_deterministically(
        self, user_message: str, session_id: str, conversation_id: str
    ) -> None:
        """Process user message using deterministic planner → router → dispatcher flow."""
        logger.info(f"🎯 Processing message deterministically: '{user_message}'")

        # Always publish a minimal echo fallback if something goes wrong.
        async def _fallback(reason: str, err: Exception | None = None):
            if err:
                logger.error("Deterministic processing failed (%s): %s", reason, err)
            else:
                logger.warning("Deterministic processing fallback (%s)", reason)
            text = f"I received your message: '{user_message}'. I'm currently having issues with my processing system, but I'm working on understanding your request."
            await self.event_bus._redis.publish(
                "agent_reply", json.dumps({"text": text, "reason": reason})
            )

        # 0) Zero‑LLM emergency rule (works even if LLM is disabled)
        try:
            if _contains_creation_intent(user_message):
                gap_msg = "Need a tool creation pipeline (Ability Contract → code → tests → register)."
                await self._emit_gap_response(
                    gap_text=f"(GAP) {gap_msg}",
                    session_id=session_id,
                    conversation_id=conversation_id,
                )
                return
        except Exception as e:
            await _fallback("gap_keyword_rule_error", e)
            return

        # 1) If router not available, short friendly NONE response for greetings/short text
        if not self.router_ready or self.registry is None:
            try:
                txt = user_message.strip().lower()
                if txt in ("hi", "hello", "hey", "yo", "sup", "hiya", "hola"):
                    await self.event_bus._redis.publish(
                        "agent_reply",
                        json.dumps(
                            {
                                "text": "Hello! How can I help today?",
                                "reason": "none_greeting",
                            }
                        ),
                    )
                else:
                    await self.event_bus._redis.publish(
                        "agent_reply",
                        json.dumps(
                            {
                                "text": f"I heard you say: '{user_message}'.",
                                "reason": "none_basic",
                            }
                        ),
                    )
                return
            except Exception as e:
                await _fallback("router_unavailable", e)
                return

        # 2) Full path: planner → parse → dispatcher
        try:
            tool_desc = _minimal_tool_descriptions(self.registry)
            planner_output = await self._get_planner_output(user_message, tool_desc)
            route = self.router.route_user_message(user_message, planner_output)
            logger.info(f"🧭 Planner output: {planner_output}, Routed to: {route}")
        except Exception as e:
            await _fallback("planner_error", e)
            return

        try:
            if self.event_bus:
                # Ensure session_id and conversation_id are strings
                session_id_str = str(session_id) if session_id else "default_session"
                conversation_id_str = (
                    str(conversation_id) if conversation_id else session_id_str
                )

                dispatcher = Dispatcher(
                    self.event_bus, session_id_str, conversation_id_str
                )
                await dispatcher.dispatch_action(route)
                logger.info(f"✅ Action dispatched: {route.action_type}")
            else:
                await _fallback("no_event_bus", None)
        except Exception as e:
            await _fallback("dispatch_error", e)
            return

    async def _get_planner_output(
        self, user_message: str, tool_descriptions: str = ""
    ) -> str:
        """Get planner output for the user message. Uses REUGExecutionFlow for complex prompts with SoT."""
        try:
            # **NEW: Check if this is a complex prompt that should use SoT**
            if self.reug_ready and self._is_complex_sot_prompt(user_message):
                logger.info(
                    f"Complex SoT prompt detected, routing to REUGExecutionFlow: {user_message[:50]}..."
                )

                # Use REUGExecutionFlow for Script-of-Thought multi-step reasoning
                try:
                    # Start session if needed
                    if not self.reug_flow.current_session_id:
                        await self.reug_flow.start_session()

                    # Process via REUG with SoT support
                    result = await self.reug_flow.process_user_input(user_message)

                    if result.get("success", False):
                        # Return the REUG response directly - it handles SoT internally
                        response = result.get(
                            "response", "Task completed via SoT execution."
                        )
                        logger.info(f"SoT execution successful: {response[:50]}...")
                        return f"SOT_EXECUTED {response}"
                    else:
                        logger.warning(
                            f"SoT execution failed: {result.get('error', 'Unknown error')}"
                        )
                        # Fall through to standard planner simulation

                except Exception as e:
                    logger.error(f"REUGExecutionFlow failed: {e}")
                    # Fall through to standard planner simulation

            # Standard planner simulation for simple prompts
            if self.llm_client:
                prompt = f"""Classify the user request and provide a single line action:

User: "{user_message}"

Available tools:
{tool_descriptions}

Reply with EXACTLY ONE of these formats:
- GAP <description> - if user wants a new capability/tool created
- NONE <response> - if this is casual conversation
- TOOL <tool_name> <params> - if user wants to use an existing tool
- SOT_EXECUTED <response> - if this was processed via Script-of-Thought

Examples:
"hello" → NONE Hello! How can I help you today?
"create a calculator" → GAP Create a calculation tool for math operations
"search for python tutorials" → TOOL web_agent python tutorials
"what's the weather" → TOOL weather_service current_weather
"remember this information" → TOOL memory_manager store_information

Response:"""

                try:
                    response = await asyncio.wait_for(
                        self._call_gemini_async(prompt), timeout=15.0
                    )
                    return response.strip()
                except Exception as e:
                    logger.warning(f"LLM planner simulation failed: {e}")

            # Fallback pattern matching
            user_lower = user_message.lower()

            # Simple patterns for common cases
            if any(
                word in user_lower
                for word in ["hello", "hi", "hey", "good morning", "good afternoon"]
            ):
                return "NONE Hello! How can I help you today?"

            if any(
                word in user_lower for word in ["create", "build", "make", "generate"]
            ):
                return f"GAP Create a tool for: {user_message}"

            if any(
                word in user_lower for word in ["search", "find", "look up", "research"]
            ):
                return f"TOOL web_agent {user_message}"

            if any(
                word in user_lower for word in ["remember", "store", "save", "recall"]
            ):
                return f"TOOL memory_manager {user_message}"

            # Default to conversation
            return f"NONE I understand you're asking about: {user_message}. Let me help you with that."

        except Exception as e:
            logger.error(f"Error getting planner output: {e}")
            return f"NONE I received your message: {user_message}"

    def _is_complex_sot_prompt(self, user_message: str) -> bool:
        """Detect if user input is a complex prompt that should use Script-of-Thought."""
        # Check for multi-step indicators
        step_indicators = [
            r"(?:step|Step)\s*\d+",  # Step 1, step 2, etc.
            r"^\d+[.)]\s",  # 1. or 1) at start of line
            r"first.*then.*finally",  # Sequential language
            r"after.*do.*then",  # Dependency language
            r"plan.*steps",  # Planning language
            r"break.*down",  # Decomposition language
        ]

        import re

        for pattern in step_indicators:
            if re.search(pattern, user_message, re.IGNORECASE | re.MULTILINE):
                return True

        # Check for complex task keywords that benefit from SoT
        complex_keywords = [
            "analyze and create",
            "research and write",
            "plan and implement",
            "design and build",
            "investigate and report",
            "compare and recommend",
            "multi-step",
            "workflow",
            "procedure",
            "methodology",
        ]

        user_lower = user_message.lower()
        for keyword in complex_keywords:
            if keyword in user_lower:
                return True

        # Check message length - longer messages often benefit from SoT
        if len(user_message.split()) > 15:  # More than 15 words
            return True

        return False

    async def _send_fallback_response(self, user_message: str) -> None:
        """Send a fallback response when deterministic processing fails."""
        try:
            fallback_text = f"I received your message: '{user_message}'. I'm currently having issues with my processing system, but I'm working on understanding your request."

            import json

            if self.event_bus and hasattr(self.event_bus, "_redis"):
                await self.event_bus._redis.publish(
                    "agent_reply", json.dumps({"text": fallback_text})
                )
                logger.info("Sent fallback response via Redis")
            else:
                logger.warning(
                    f"Cannot send fallback response - event bus unavailable. Message was: {fallback_text}"
                )
        except Exception as e:
            logger.error(f"Error sending fallback response: {e}")

    async def _handle_create_atom_command(self, command: str) -> None:
        """Handle /create-atom tool="name" description="desc" code="python_code" commands"""
        try:
            import re

            # Parse command: /create-atom tool="prime_counter" description="Count primes up to N" code="import sympy; print(sympy.primepi(int(input())))"
            tool_match = re.search(r'tool="([^"]+)"', command)
            desc_match = re.search(r'description="([^"]+)"', command)
            code_match = re.search(r'code="([^"]+)"', command)

            if not tool_match or not desc_match or not code_match:
                await self.event_bus._redis.publish(
                    "agent_reply",
                    json.dumps(
                        {
                            "text": '❌ Invalid format. Use: /create-atom tool="name" description="desc" code="python_code"'
                        }
                    ),
                )
                return

            tool_name = tool_match.group(1)
            description = desc_match.group(1)
            code = code_match.group(1)

            logger.info(f"🔧 Creating atom: {tool_name}")

            # Emit atom_create event for storage
            await self.emit_event(
                "atom_create",
                tool_name=tool_name,
                description=description,
                code=code,
                created_by="user_command",
            )

            await self.event_bus._redis.publish(
                "agent_reply",
                json.dumps(
                    {"text": f"🧬 Atom '{tool_name}' created and stored in memory"}
                ),
            )

            logger.info(f"✅ Atom creation event emitted for: {tool_name}")

        except Exception as e:
            logger.error(f"Error handling create-atom command: {e}")
            await self.event_bus._redis.publish(
                "agent_reply", json.dumps({"text": f"❌ Atom creation error: {e}"})
            )

    async def _handle_atom_run_command(self, command: str) -> None:
        """Handle /atom-run tool="name" args="value" commands"""
        try:
            import re

            # Parse command: /atom-run tool="prime_counter" args="100"
            tool_match = re.search(r'tool="([^"]+)"', command)
            args_match = re.search(r'args="([^"]+)"', command)

            if not tool_match:
                await self.event_bus._redis.publish(
                    "agent_reply",
                    json.dumps(
                        {
                            "text": '❌ Invalid format. Use: /atom-run tool="tool_name" args="arguments"'
                        }
                    ),
                )
                return

            tool_name = tool_match.group(1)
            args = args_match.group(1) if args_match else ""

            logger.info(f"🔧 Running atom: {tool_name} with args: {args}")

            # Emit atom_run event for execution
            await self.emit_event(
                "atom_run", tool_name=tool_name, args=args, requested_by="user_command"
            )

            # Don't send immediate reply - let the executor respond
            logger.info(f"✅ Atom execution event emitted for: {tool_name}")

        except Exception as e:
            logger.error(f"Error handling atom-run command: {e}")
            await self.event_bus._redis.publish(
                "agent_reply", json.dumps({"text": f"❌ Atom execution error: {e}"})
            )

    async def _handle_atomize_request(self, data, session_id, message_id) -> None:
        """Handle atomize: commands deterministically via fast-path."""
        try:
            user_message = data.get("user_message", "")

            # Extract text after "atomize:"
            text_to_atomize = user_message[8:].strip()  # Remove "atomize:" prefix

            if not text_to_atomize:
                await self.emit_event(
                    "agent_response",
                    session_id=session_id,
                    message_id=message_id,
                    response="❌ No text provided to atomize. Usage: atomize: your text here",
                    status="error",
                )
                return

            logger.info(f"🔬 Processing atomize request for text: '{text_to_atomize}'")

            # Emit atomize request event
            await self.emit_event(
                "atomize_text_request",
                text=text_to_atomize,
                max_notes=5,
                context={
                    "session_id": session_id,
                    "message_id": message_id,
                    "source": "conversation_plugin",
                },
            )

            # Send acknowledgment
            await self.emit_event(
                "agent_response",
                session_id=session_id,
                message_id=message_id,
                response=f"🔬 Atomizing text: '{text_to_atomize}'\n⏳ Processing atoms and bonds...",
                status="processing",
            )

            logger.info("✅ Atomize request processed successfully")

        except Exception as e:
            logger.error(f"Error handling atomize request: {e}")
            await self.emit_event(
                "agent_response",
                session_id=session_id,
                message_id=message_id,
                response=f"❌ Error processing atomize request: {e}",
                status="error",
            )

    async def _handle_conversation(self, event) -> None:
        """Handle incoming conversation messages with strict source validation."""
        logger.debug(f"_handle_conversation called with event: {event}")

        # --- CRITICAL FIX: Disable old path when deterministic router is available ---
        if self.router_ready:
            logger.debug(
                "Ignoring conversation event - deterministic router is handling all messages"
            )
            return

        # --- CRITICAL FIX: Prevent Cognitive Feedback Loop (Priority 1) ---
        # 1. Respect the 'handled_by' contract as a defensive measure.
        if getattr(event, "handled_by", None):
            logger.debug(
                f"ConversationPlugin ignoring event already handled by '{event.handled_by}'."
            )
            return

        # 2. THE DEFINITIVE FIX: Ignore all messages not originating from the user-facing client.
        #    This prevents the plugin from processing the preprocessor's clarification requests.
        #    'chat_client' is assumed to be the canonical name for the user's interface plugin.
        source_plugin = getattr(event, "source_plugin", None)
        if source_plugin and source_plugin != "chat_client":
            logger.debug(
                f"ConversationPlugin ignoring non-user message from source: '{source_plugin}'"
            )
            return
        # --- End Critical Fix Section ---

        # THE SECONDARY FIX: Check if we're already processing another message
        if self._conversation_lock.locked():
            # Handle both WorkspaceEvent and legacy event formats
            if hasattr(event, "data"):
                data = event.data
            else:
                data = event.model_dump() if hasattr(event, "model_dump") else event
            user_message = (
                data.get("user_message", "unknown")
                if isinstance(data, dict)
                else "unknown"
            )
            logger.warning(
                f"Ignoring message '{user_message}' as another conversation is already being processed"
            )
            return

        async with self._conversation_lock:
            # Now we are guaranteed to only be processing one message at a time
            try:
                # Extract data from event object - handle both WorkspaceEvent and legacy formats
                if hasattr(event, "data"):
                    # New WorkspaceEvent format
                    data = event.data
                    event_type = "conversation_message"
                else:
                    # Legacy format
                    data = event.model_dump() if hasattr(event, "model_dump") else event
                    event_type = getattr(event, "event_type", "conversation_message")

                # Ensure data is a dictionary
                if not isinstance(data, dict):
                    logger.warning(f"Event data is not a dictionary: {type(data)}")
                    return

                # Duplicate message detection
                text = (
                    data.get("user_message", "") or ""
                ).strip()  # Extract and strip text
                if not text:  # Check for empty text
                    logger.debug(
                        "Empty message in %s, skipping",
                        data.get("session_id", "unknown"),
                    )
                    return  # Exit if empty

                h = hashlib.md5(text.encode()).hexdigest()  # Hash the message
                now = time.time()  # Get current time
                session_id = data.get("session_id", "default")
                prev_h, prev_t = _last_hash.get(
                    session_id, ("", 0)
                )  # Get previous hash/time
                if (
                    h == prev_h and now - prev_t < _DUP_WINDOW
                ):  # Check for duplicate within window
                    logger.debug(
                        "Duplicate message in %s, ignoring", session_id
                    )  # Log duplicate
                    return  # Exit if duplicate
                _last_hash[session_id] = (h, now)  # Store current hash/time

                # ATOMIZE FAST-PATH: Handle "atomize:" commands deterministically
                if text.lower().startswith("atomize:"):
                    # FIX: Extract message_id from data properly
                    message_id = data.get(
                        "message_id", f"msg_{session_id}_{int(time.time())}"
                    )
                    await self._handle_atomize_request(data, session_id, message_id)
                    return

                logger.info(
                    f"[LOCKED] Conversation handler called with event type: {event_type}"
                )
                logger.info(f"[LOCKED] Event data: {data}")

                session_id = data.get("session_id")
                user_message = data.get("user_message", "")
                context = data.get("context", {})
                message_id = data.get("message_id")

                logger.info(
                    f"[LOCKED] Processing message: '{user_message}' for session: {session_id}"
                )

                if not session_id or not user_message:
                    logger.warning(
                        f"Missing required fields - session_id: {bool(session_id)}, user_message: {bool(user_message)}"
                    )
                    return

                # CRITICAL FIX: Keep raw user text separate from UI echo formatting
                raw_user_text = user_message.strip()

                # Update session context
                if session_id not in self.active_sessions:
                    self.active_sessions[session_id] = {
                        "start_time": datetime.now().isoformat(),
                        "message_count": 0,
                        "context": context,
                    }

                session = self.active_sessions[session_id]
                session["message_count"] += 1
                session["last_message"] = raw_user_text  # Store raw text, not formatted
                session["last_activity"] = datetime.now().isoformat()

                # Emit thinking indicator with formatted version (UI only - don't use downstream)
                thinking_text = (
                    f"I heard you say: {raw_user_text}. I'm thinking about that..."
                )
                logger.info("Emitting thinking indicator...")
                await self.emit_event(
                    "agent_thinking",
                    session_id=session_id,
                    stage="processing",
                    message_id=message_id,
                )

                # --- Stage 0: Enhanced intent classification for memory queries ---------------------------------
                intent_prompt = (
                    f"Classify the user intent: '{user_message}'\n"
                    "Consider memory-related queries (remember, recall, store, save, memories) as task_request.\n"
                    "Reply with exactly one word: chit_chat or task_request"
                )

                if self.llm_client:
                    try:
                        intent_response_text = await asyncio.wait_for(
                            self._call_gemini_async(intent_prompt), timeout=15.0
                        )
                        intent = intent_response_text.strip().lower()
                        logger.info(f"Intent classified as: {intent}")
                    except Exception as e:
                        logger.warning(
                            f"Intent classification failed, defaulting to chit_chat: {e}"
                        )
                        intent = "chit_chat"
                else:
                    intent = "chit_chat"

                # CRITICAL REFACTOR: Simplify to pure intent classification
                # If task_request detected, initiate cognitive turn via DTA 2.0 pipeline
                if (
                    await self._is_complex_task(user_message)
                    or intent == "task_request"
                ):
                    logger.info(
                        f"🎯 Task request detected, initiating cognitive turn for: '{user_message}'"
                    )

                    # CRITICAL: Instead of letting the event propagate, we now explicitly
                    # publish a new, dedicated event to trigger the DTA 2.0 pipeline.
                    # This makes the event flow unambiguous.
                    await self.event_bus.publish(
                        CognitiveTurnInitiatedEvent(
                            source_plugin=self.name,
                            user_message=raw_user_text,  # PATCH: Use raw text, not formatted echo
                            session_id=session_id,
                            conversation_id=getattr(
                                event, "conversation_id", session_id
                            ),
                            original_event_id=getattr(event, "event_id", ""),
                            intent_confidence=0.9,
                            cognitive_context={
                                "message_id": message_id,
                                "classification_method": (
                                    "llm" if self.llm_client else "pattern_matching"
                                ),
                                "detected_keywords": [
                                    kw
                                    for kw in [
                                        "analyze",
                                        "analysis",
                                        "abilities",
                                        "capabilities",
                                        "introspect",
                                        "assess",
                                        "system",
                                        "status",
                                        "health",
                                        "diagnostic",
                                    ]
                                    if kw in user_message.lower()
                                ],
                            },
                        )
                    )
                    logger.info(
                        "✅ Cognitive turn initiated - delegating to DTA 2.0 pipeline"
                    )

                    # Generate an immediate response while cognitive processing happens
                    immediate_response = "I'm processing your request and will provide a comprehensive response shortly. Let me analyze this for you..."

                    # Emit the immediate response
                    await self.emit_event(
                        "conversation_message",
                        session_id=session_id,
                        user_message=immediate_response,
                        message_id=f"response_{message_id}",
                        context={"type": "immediate_response", "processing": True},
                    )
                    logger.info("📤 Emitted immediate response while processing")
                    return

                # Streamlined memory-grounded response generation with stateful context
                logger.info("🧠 Generating stateful, memory-grounded response...")

                # 1. Build running context from last 3 turns
                context_lines = []
                try:
                    if (
                        hasattr(self, "conversation_memory")
                        and self.conversation_memory
                    ):
                        # Get last 3 turns for this session
                        session_turns = [
                            turn
                            for turn in self.conversation_memory
                            if turn.get("session_id") == session_id
                        ][-3:]

                        for turn in session_turns:
                            context_lines.append(
                                f"User: {turn.get('user_message', '')}"
                            )
                            context_lines.append(
                                f"Assistant: {turn.get('agent_response', '')}"
                            )

                    context = (
                        "\n".join(context_lines)
                        if context_lines
                        else "(No previous conversation)"
                    )
                    logger.info(
                        f"📜 Context built: {len(context_lines)} turn fragments"
                    )

                except Exception as e:
                    logger.warning(f"Failed to build conversation context: {e}")
                    context = "(Context unavailable)"

                # 2. Get memory context via embedding similarity
                memory_context = ""
                try:
                    if hasattr(self.store, "embed_text") and hasattr(
                        self.store, "attention"
                    ):
                        logger.info("📚 Retrieving memory context...")
                        query_vec = await self.store.embed_text([user_message])
                        if query_vec and len(query_vec) > 0:
                            memories = await self.store.attention(query_vec[0], top_k=3)
                            memory_lines = []
                            for key, score in memories:
                                atom = self.store.get(key)
                                if atom and hasattr(atom, "value"):
                                    content = str(atom.value)[:120]
                                    memory_lines.append(f"• {content}")
                            memory_context = "\n".join(memory_lines)
                            logger.info(
                                f"✅ Memory context: {len(memory_lines)} relevant memories"
                            )
                        else:
                            logger.warning("embed_text returned empty or None")
                    else:
                        logger.warning("Memory methods not available on store")
                except Exception as e:
                    logger.error(f"Memory retrieval failed: {e}")
                    memory_context = ""

                # 3. Generate response with full context
                prompt = f"""I am Super Alita, a neuro-symbolic cognitive agent with Tool-First Mentality and Action Over Explanation.

Previous Conversation:
{context}

Relevant Memories:
{memory_context if memory_context else "(No relevant memories found)"}

Current User Message: {user_message}

BEHAVIORAL CONTRACT:
- If the user asks something I can answer from conversation history or memory, answer immediately BEFORE creating tools
- If the user needs a computational task, create/use appropriate tools
- If tools already exist for the task, reuse them instead of recreating
- Maintain conversation continuity and remember previous interactions
- Be helpful, intelligent, and action-oriented

Response:"""

                if self.llm_client:
                    try:
                        response_text = await asyncio.wait_for(
                            self._call_gemini_async(prompt), timeout=30.0
                        )
                        logger.info("✅ Memory-grounded response generated")
                    except Exception as e:
                        logger.error(f"LLM response generation failed: {e}")
                        response_text = f"I understand you're asking about: {user_message}\n\nBased on my memory: {memory_context[:200] if memory_context else 'No relevant context found'}\n\nHowever, I encountered an LLM error: {str(e)[:100]}"
                else:
                    response_text = f"I received your message: {user_message}\n\nMemory context: {memory_context if memory_context else 'Memory system not available'}\n\nLLM client not available for full response generation."

                # Store this interaction in memory for future reference and turn history
                try:
                    interaction_memory = {
                        "session_id": session_id,
                        "user_message": user_message,
                        "agent_response": response_text[:500],  # Store summary
                        "timestamp": datetime.now().isoformat(),
                        "mode": "stateful_memory_grounded",
                    }

                    # Add to conversation memory for context building
                    self.conversation_memory.append(interaction_memory)

                    # Also store in semantic memory if available
                    if hasattr(self.store, "upsert"):
                        memory_id = f"turn_{session_id}_{len(self.conversation_memory)}"
                        await self.store.upsert(
                            memory_id=memory_id,
                            content=interaction_memory,
                            hierarchy_path=["turns", session_id],
                        )
                        logger.info(f"📝 Turn stored in semantic memory: {memory_id}")
                    else:
                        logger.info("📝 Turn stored in session memory only")

                except Exception as e:
                    logger.warning(f"Memory storage failed: {e}")

                # Emit response
                await self.emit_event(
                    "agent_response",
                    session_id=session_id,
                    response_text=response_text,
                    reasoning="Memory-grounded response with context",
                    context={
                        "intent": intent if "intent" in locals() else "unknown",
                        "memory_used": bool(memory_context),
                    },
                    timestamp=datetime.now().isoformat(),
                    response_id=f"response_{message_id}",
                )

                # Also send to chat client
                import json

                await self.event_bus._redis.publish(
                    "agent_reply", json.dumps({"text": response_text})
                )
                logger.info("✅ Memory-grounded response delivered")
                return

                # --- Stage 1: Memory-grounded response for chit-chat -----------------------
                logger.info("Processing as chit-chat with memory grounding")

                # Get semantic memory for context - FIXED TO ACTUALLY WORK
                memory_snippet = ""
                try:
                    # CRITICAL FIX: Test if memory system actually works, then use it
                    if hasattr(self.store, "embed_query") and hasattr(
                        self.store, "attention"
                    ):
                        try:
                            # THIS IS THE KEY FIX - actually call the memory system
                            logger.info("Attempting memory system access...")
                            query_vec = await self.store.embed_query(user_message)

                            if query_vec is not None:
                                memories = await self.store.attention(
                                    query_vec, top_k=3
                                )

                                memory_lines = []
                                for key, score in memories:
                                    atom = self.store.get(key)
                                    if atom and hasattr(atom, "value"):
                                        content = (
                                            str(atom.value)[:120] + "..."
                                            if len(str(atom.value)) > 120
                                            else str(atom.value)
                                        )
                                        memory_lines.append(f"- {score:.2f} {content}")

                                memory_snippet = (
                                    "\n".join(memory_lines)
                                    if memory_lines
                                    else "No relevant memories found."
                                )
                                logger.info(
                                    f"✅ MEMORY ACTIVE: Retrieved {len(memory_lines)} relevant memories"
                                )
                            else:
                                memory_snippet = "Memory embedding returned None - semantic_memory plugin may not be connected."
                                logger.warning(
                                    "❌ MEMORY ISSUE: embed_query returned None"
                                )
                        except Exception as e:
                            logger.warning(
                                f"❌ MEMORY FAILED: Memory retrieval failed: {e}"
                            )
                            memory_snippet = (
                                "Memory system methods exist but failed to execute."
                            )
                    else:
                        # More detailed diagnostic of what's missing
                        store_methods = [
                            method
                            for method in dir(self.store)
                            if not method.startswith("_")
                        ]
                        logger.info(f"Available store methods: {store_methods}")

                        missing_methods = []
                        if not hasattr(self.store, "embed_query"):
                            missing_methods.append("embed_query")
                        if not hasattr(self.store, "attention"):
                            missing_methods.append("attention")

                        memory_snippet = f"Memory system not fully configured. Missing methods: {missing_methods}. Available: {len(store_methods)} methods."
                        logger.warning(
                            f"❌ MEMORY INCOMPLETE: missing: {missing_methods}"
                        )

                except Exception as e:
                    logger.warning(f"Error accessing memory: {e}")
                    memory_snippet = f"Memory access error: {str(e)[:100]}..."

                # Add small delay for natural feeling
                await asyncio.sleep(self.response_delay)

                # Generate memory-grounded response
                logger.info("Generating memory-grounded response...")

                full_prompt = f"""User: {user_message}

Relevant past experiences:
{memory_snippet}

Generate a concise, memory-grounded reply that references relevant past experiences when appropriate.""".strip()

                response_text, reasoning = await self._generate_response_with_prompt(
                    full_prompt, user_message, context, session
                )

                logger.info(f"Response generated: '{response_text[:50]}...'")
                logger.info(f"Reasoning: '{reasoning[:50]}...'")

                # Store in memory
                self.conversation_memory.append(
                    {
                        "session_id": session_id,
                        "user_message": user_message,
                        "agent_response": response_text,
                        "reasoning": reasoning,
                        "timestamp": datetime.now().isoformat(),
                        "context": context,
                    }
                )

                # Emit response
                logger.info("Emitting agent response event...")
                await self.emit_event(
                    "agent_response",
                    session_id=session_id,
                    response_text=response_text,
                    reasoning=reasoning if self.enable_reasoning else "",
                    context={"session_info": session, "message_processed": True},
                    timestamp=datetime.now().isoformat(),
                    response_id=f"resp_{message_id}",
                )
                logger.info("Agent response event emitted successfully")

                # Also send to chat client on agent_reply channel
                import json

                await self.event_bus._redis.publish(
                    "agent_reply", json.dumps({"text": response_text})
                )
                logger.info("Response sent to chat client on agent_reply channel")

            except Exception as e:
                logger.error(f"Error handling conversation: {e}")

                # Extract data from event for error handling
                try:
                    if hasattr(event, "data"):
                        data = event.data
                    else:
                        data = (
                            event.model_dump()
                            if hasattr(event, "model_dump")
                            else event
                        )

                    if not isinstance(data, dict):
                        data = {}
                except Exception:
                    data = {}

                error_message = "I apologize, but I encountered an error processing your message. Could you please try again?"

                # Send error response
                await self.emit_event(
                    "agent_response",
                    session_id=(
                        data.get("session_id") if isinstance(data, dict) else "unknown"
                    ),
                    response_text=error_message,
                    reasoning=f"Error occurred: {e!s}",
                    timestamp=datetime.now().isoformat(),
                    response_id=f"error_{data.get('message_id', 'unknown') if isinstance(data, dict) else 'unknown'}",
                )

                # Also send error to chat client
                import json

                await self.event_bus._redis.publish(
                    "agent_reply", json.dumps({"text": error_message})
                )
                logger.info("Error response sent to chat client")

    async def _is_complex_task(self, user_message: str) -> bool:
        """
        Use deterministic router prompt to classify tasks with hardened keywords.
        NOTE: Removed /search prefix requirement - planner now handles natural language intent detection.
        """
        if not self.llm_client or not self.router_prompt:
            # Fallback to pattern matching with enhanced keywords (Your Priority 2)
            # ENHANCED: Include memory-related and system keywords to delegate to planner
            task_indicators = [
                "analyze",
                "create",
                "build",
                "generate",
                "help me with",
                "can you",
                "remember",
                "memory",
                "recall",
                "store",
                "save",
                "list memories",
                "what did you save",
                "did you store",
                "search",
                "find",
                "look up",
                "research",
                "write code for",
                "diagnose",
                "debug",
                "explain",
                "assess",
                "evaluate",
                "check",
                "status",
                "system",
                "capabilities",
                "run",
                "execute",
                "plan",
            ]
            return any(
                indicator in user_message.lower() for indicator in task_indicators
            )

        try:
            # Use bullet-proof prompt for classification and tool selection
            clean_msg = user_message.strip()
            prompt = f"""
Input: "{clean_msg}"
Return ONLY valid JSON, no markdown, no explanation:
{{"intent":"chat|task","tools_needed":[]}}
Examples:
"hello" → {{"intent":"chat","tools_needed":[]}}
"help me analyze this data" → {{"intent":"task","tools_needed":[]}}
"did you store any memories" → {{"intent":"task","tools_needed":["memory_manager"]}}
"remember this information" → {{"intent":"task","tools_needed":["memory_manager"]}}
"what do you recall about" → {{"intent":"task","tools_needed":["memory_manager"]}}
"""

            response = self.llm_client.generate_content(prompt)
            raw = response.text.strip()

            # Strip markdown fences or handle empty response
            if raw.startswith("```json"):
                raw = raw[7:-4]
            elif raw.startswith("```"):
                raw = raw[3:-3]

            # Parse JSON response with fallback
            import json

            try:
                decision = json.loads(raw or '{"intent":"task","tools_needed":[]}')
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response: {raw}")
                # Safe fallback for general tasks
                decision = {"intent": "task", "tools_needed": []}

            # Store tools for later use (though planner will now handle search routing)
            self._selected_tools = decision.get("tools_needed", [])

            # Return True if "task", False if "chat"
            return decision.get("intent") == "task"

        except Exception as e:
            logger.error(f"Error in router classification: {e}")
            # Fallback to simple pattern matching (excluding /search but including memory keywords)
            task_indicators = [
                "analyze",
                "create",
                "build",
                "generate",
                "help me with",
                "can you",
                "remember",
                "memory",
                "recall",
                "store",
                "save",
                "list memories",
                "what did you save",
                "did you store",
                "assess",
                "system",
                "status",
                "health",
                "diagnostic",
                "capabilities",
                "abilities",
            ]
            self._selected_tools = []
            return any(
                indicator in user_message.lower() for indicator in task_indicators
            )

    async def _generate_response(
        self, user_message: str, context: dict, session: dict
    ) -> tuple[str, str]:
        """Generate an intelligent response to the user message."""

        user_message_lower = user_message.lower().strip()

        # Only use pattern matching for very specific, simple cases
        # Let the LLM handle most conversational nuance

        # Handle clear farewells (but let LLM handle ambiguous cases)
        if user_message_lower in ["goodbye", "bye", "exit", "quit"]:
            return self._get_farewell_response(context, session)

        # Handle direct capability questions only if very explicit
        if user_message_lower in ["help", "what can you do", "capabilities"]:
            return self._get_capabilities_response()

        # Handle self-assessment and limitation queries
        if any(
            keyword in user_message_lower
            for keyword in [
                "assess",
                "limitation",
                "what needs",
                "fix",
                "status",
                "diagnose",
                "what's broken",
                "what works",
                "system status",
                "problems",
            ]
        ):
            return await self._generate_self_assessment_response(
                user_message, context, session
            )

        # For everything else, including greetings, use the intelligent LLM response
        # This allows for much more nuanced and context-aware conversations
        return await self._generate_llm_response(user_message, context, session)

    async def _generate_llm_response(
        self, user_message: str, context: dict, session: dict
    ) -> tuple[str, str]:
        """Generate intelligent response using Gemini LLM with rich conversational context."""

        if not self.llm_client:
            logger.warning("LLM client not available, using fallback response")
            return self._generate_fallback_response(user_message, context, session)

        try:
            # Build rich conversation context
            conversation_history = context.get("conversation_history", [])
            session_info = {
                "start_time": session.get("start_time", "unknown"),
                "message_count": session.get("message_count", 0),
                "user_name": context.get("user_name", "Human"),
            }

            # Create comprehensive system prompt
            system_prompt = f"""You are Super Alita, an advanced neuro-symbolic cognitive agent with the following characteristics:

CORE IDENTITY:
- You are a sophisticated AI agent with multiple cognitive plugins (semantic memory, skill discovery, finite state machines)
- You have access to genealogy tracking, neural atoms, and distributed event processing via Redis
- You can reason deeply, learn from interactions, and maintain context across conversations
- You are helpful, intellectually curious, and engage thoughtfully with complex topics

CURRENT SESSION CONTEXT:
- User: {session_info["user_name"]}
- Session started: {session_info["start_time"]}
- Messages exchanged: {session_info["message_count"]}

CONVERSATION HISTORY:"""

            # Add detailed conversation history
            if conversation_history:
                for i, msg in enumerate(
                    conversation_history[-8:],  # Last 8 messages for context
                ):
                    role = msg.get("role", "unknown").capitalize()
                    content = msg.get("content", "")
                    timestamp = msg.get("timestamp", "")
                    system_prompt += f"\n{i}. {role} ({timestamp}): {content}"
            else:
                system_prompt += "\n(This is the start of our conversation)"

            # Add current user message and instruction
            full_prompt = f"""{system_prompt}

CURRENT USER MESSAGE: "{user_message}"

INSTRUCTIONS:
- Respond naturally and contextually, taking into account the full conversation history
- If this relates to previous messages, acknowledge that context
- Be genuine, thoughtful, and avoid scripted-sounding responses
- Show your cognitive capabilities when relevant
- If asked meta-questions about your responses, be honest and reflective

Your response:"""

            # Generate response with timeout protection
            response = await asyncio.wait_for(
                self._call_gemini_async(full_prompt),
                timeout=30.0,  # 30 second timeout
            )

            reasoning = f"Generated context-aware response using full conversation history ({len(conversation_history)} total messages, session message #{session_info['message_count']})"

            return response.strip(), reasoning

        except TimeoutError:
            logger.error("LLM call timed out after 30 seconds")
            return self._generate_timeout_response(user_message)

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return self._generate_fallback_response(user_message, context, session)

    async def _call_gemini_async(self, prompt: str) -> str:
        """Async wrapper for Gemini API call using the resilient LLM client."""
        try:
            # Use the new resilient LLM client with built-in timeout and retry
            from src.core.llm_client import generate

            return await generate(prompt)
        except Exception as e:
            logger.exception(f"LLM client error: {e}")

            # Fallback to direct Gemini call if new client fails
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.llm_client.generate_content(prompt)
            )
            return response.text

    async def _generate_response_with_prompt(
        self, full_prompt: str, user_message: str, context: dict, session: dict
    ) -> tuple[str, str]:
        """Generate response using a custom prompt (for memory-grounded responses)."""

        if not self.llm_client:
            logger.warning("LLM client not available, using fallback response")
            return self._generate_fallback_response(user_message, context, session)

        try:
            # Use the provided prompt directly
            response_text = await asyncio.wait_for(
                self._call_gemini_async(full_prompt), timeout=30.0
            )

            reasoning = (
                f"Memory-grounded response generated for: {user_message[:50]}..."
            )

            logger.info(f"Generated memory-grounded response: {response_text[:100]}...")
            return response_text.strip(), reasoning

        except TimeoutError:
            logger.error("LLM response timed out")
            return self._generate_fallback_response(user_message, context, session)
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return self._generate_fallback_response(user_message, context, session)

    def _generate_fallback_response(
        self, user_message: str, context: dict, session: dict
    ) -> tuple[str, str]:
        """Generate response using memory context and direct Gemini API when LLM client unavailable."""
        try:
            # Try to get memory context
            memory_context = ""
            if hasattr(self.store, "semantic_search"):
                try:
                    memories = self.store.semantic_search(user_message, limit=3)
                    if memories:
                        memory_lines = [
                            f"• {mem.get('content', 'No content')}"  # Fixed: Added missing closing quote
                            for mem in memories[:3]
                        ]
                        memory_context = (
                            "\n\n**Relevant Memory Context:**\n"
                            + "\n".join(memory_lines)
                        )
                except Exception as e:
                    logger.debug(f"Could not retrieve memory context: {e}")

            # Try to use Gemini directly for intelligent response
            try:
                if GEMINI_AVAILABLE:
                    api_key = os.getenv("GEMINI_API_KEY")
                    if api_key:
                        genai.configure(api_key=api_key)

                        # Use model from config, default to gemini-2.5-pro
                        model_name = "gemini-2.5-pro"
                        model = genai.GenerativeModel(model_name)

                        prompt = f"""You are Super Alita, an advanced cognitive agent. Respond naturally to: "{user_message}"

Context: This is a direct API call since the main LLM client is unavailable.{memory_context}

Be helpful, intelligent, and show your cognitive capabilities."""

                        response = model.generate_content(prompt)

                        if response and response.text:
                            return (
                                response.text.strip(),
                                "Generated using direct Gemini API with memory context",
                            )

            except Exception as e:
                logger.debug(f"Direct Gemini API call failed: {e}")

            # Fallback to diagnostic response if both LLM and direct API fail
            response = f"""I appreciate your message: "{user_message}"{memory_context}

**Current System Status:**
• **LLM Integration**: ❌ Gemini client unavailable (check API key)
• **Memory System**: {"✓" if hasattr(self.store, "attention") else "❌"} {"Working" if hasattr(self.store, "attention") else "Memory grounding disabled"}
• **Event Bus**: {"✓ Connected to Redis" if self.event_bus else "❌ Event bus unavailable"}

**What I can currently do:**
• Process and acknowledge your messages
• Route task requests to planning and execution system
• Provide system diagnostics
• Handle basic conversation flow

**To unlock full capabilities:**
1. **Set API Key**: `$Env:GEMINI_API_KEY="your-key-here"`
2. **Restart Agent**: `python launch_super_alita.py`
3. **Enable Memory**: Ensure semantic_memory plugin is running

Type 'system status' or 'what can you do' for detailed diagnostics."""

            reasoning = "LLM unavailable - providing system status with available memory context"
            return response, reasoning

        except Exception as e:
            logger.error(f"Error in fallback response generation: {e}")
            return (
                f"I received your message but encountered an error processing it: {e}",
                "Error in fallback processing",
            )

    def _generate_timeout_response(self, user_message: str) -> tuple[str, str]:
        """Generate response when LLM call times out."""
        response = f"""I apologize - my deep cognitive processing is taking longer than expected for: "{user_message}"

Let me give you a more immediate response: I'm designed to engage thoughtfully with complex topics, and sometimes that requires more processing time than I have available right now.

Could you rephrase your question or ask something more specific? I'll be able to respond more quickly!"""

        reasoning = "LLM call timed out, providing timeout explanation"
        return response, reasoning

    def _get_greeting_response(self, context: dict, session: dict) -> tuple[str, str]:
        """Generate a greeting response."""
        import random

        if session["message_count"] == 1:
            response = random.choice(self.response_templates["greeting"])
        else:
            response = "Hello again! What would you like to discuss now?"

        reasoning = "Detected greeting pattern, responding with warm welcome"
        return response, reasoning

    def _get_farewell_response(self, context: dict, session: dict) -> tuple[str, str]:
        """Generate a farewell response."""
        import random

        response = random.choice(self.response_templates["farewell"])
        reasoning = f"Detected farewell after {session['message_count']} messages in this session"
        return response, reasoning

    def _get_self_description(self) -> tuple[str, str]:
        """Describe the agent's identity."""
        response = """I'm Super Alita, a neuro-symbolic cognitive agent with several fascinating capabilities:

🧠 **Cognitive Architecture**: I use a plugin-based system with neural atoms for reactive state management
💾 **Semantic Memory**: I store and retrieve information using vector embeddings and ChromaDB
🎯 **Planning & Reasoning**: I employ LADDER-AOG (And-Or Graph) reasoning with MCTS for complex problem-solving
🔄 **State Management**: I use semantic finite state machines for intelligent workflow transitions
📈 **Continuous Learning**: I can discover and evolve new skills through evolutionary algorithms
🌐 **Event-Driven**: All my components communicate through an async event bus system

I'm designed to be a helpful, intelligent conversation partner who can analyze, learn, plan, and reason about complex topics!"""

        reasoning = "Providing comprehensive self-description covering architecture and capabilities"
        return response, reasoning

    def _get_capabilities_response(self) -> tuple[str, str]:
        """Describe agent capabilities with current system status."""
        # Check actual system status
        llm_available = bool(self.llm_client)
        memory_methods_available = hasattr(self.store, "attention") and hasattr(
            self.store, "embed_query"
        )
        eventbus_available = bool(
            self.event_bus and hasattr(self.event_bus, "is_running")
        )

        response = f"""## 🤖 Super Alita Capabilities Assessment

### **Currently Working** ✅
• **Event-Driven Architecture**: {"✓" if eventbus_available else "❌"} Redis-backed event bus for distributed communication
• **Conversation Processing**: ✓ Natural language understanding and response generation
• **Intent Classification**: {"✓" if llm_available else "❌"} Distinguish between chat and task requests
• **Task Delegation**: ✓ Forward goals to planning and execution system
• **Session Management**: ✓ Multi-turn dialogue context tracking

### **LLM Integration** {"✅" if llm_available else "❌"}
• **Gemini API**: {"Connected" if llm_available else "Not available - check GEMINI_API_KEY"}
• **Response Generation**: {"Active" if llm_available else "Using fallback responses"}
• **Reasoning Explanation**: {"Available" if llm_available else "Limited to basic logic"}

### **Memory & Learning** {"✅" if memory_methods_available else "❌"}
• **Semantic Memory**: {"Active" if memory_methods_available else "Not connected - check semantic_memory plugin"}
• **Memory Grounding**: {"Working" if memory_methods_available else "Disabled - responses not memory-informed"}
• **Experience Storage**: {"Available" if memory_methods_available else "Session-only memory"}

### **Planning & Execution** ⚠️
• **LADDER-AOG Reasoning**: Enabled (delegates to planning plugin)
• **Tool Execution**: Configured but implementation status unknown
• **Skill Discovery**: Available but may need testing

### **What You Can Try**
1. **Chat & Discussion**: Always available - I can engage on any topic
2. **Task Requests**: Say things like "help me analyze X" or "can you research Y"
3. **System Diagnostics**: Ask "what's your status" or "assess your limitations"
4. **Planning**: Give me goals and I'll work through them systematically

### **Current Limitations**
{("• API Key missing - set GEMINI_API_KEY environment variable" if not llm_available else "")}
{("• Memory system disconnected - semantic grounding unavailable" if not memory_methods_available else "")}
• Web search and external tool access not confirmed active
• Some advanced cognitive features may be in development mode

**Pro Tip**: For a detailed system health check, ask me to "assess your limitations" or type 'system status'"""

        reasoning = f"Providing realistic capabilities assessment - LLM: {'available' if llm_available else 'unavailable'}, Memory: {'available' if memory_methods_available else 'unavailable'}"
        return response, reasoning

    async def _generate_self_assessment_response(
        self, user_message: str, context: dict, session: dict
    ) -> tuple[str, str]:
        """Generate detailed self-assessment and limitation analysis."""
        # Perform quick system diagnosis
        llm_available = bool(self.llm_client)
        memory_methods_available = hasattr(self.store, "attention") and hasattr(
            self.store, "embed_query"
        )
        eventbus_available = bool(
            self.event_bus and hasattr(self.event_bus, "is_running")
        )

        # Check environment
        api_key_set = bool(os.environ.get("GEMINI_API_KEY"))

        # Define all message templates without escape sequences
        api_setup_msg = "```\n$Env:GEMINI_API_KEY='your-api-key-here'\npython launch_super_alita.py\n```"
        package_fix_msg = "✅ API key configured but Gemini package has compatibility issues - Fix with these commands:\n```\npip install google-generativeai==0.3.2\npip install protobuf==4.24.4\npython -m pip install --upgrade google-api-python-client\n```"
        memory_fix_msg_prefix = "Issue: "
        memory_fix_msg_suffix = "\nSolution:\n• Check if semantic_memory plugin is enabled in agent.yaml\n• Verify ChromaDB installation: pip install chromadb\n• Ensure embedding service is working\n• Check logs for semantic_memory plugin startup errors"
        memory_fix_ok = "✅ Memory system connected"

        # Advanced memory system diagnostic
        memory_diagnostic = "Unknown"
        if self.store:
            store_methods = [
                method for method in dir(self.store) if not method.startswith("_")
            ]
            if memory_methods_available:
                try:
                    # Test if memory system actually works
                    test_vec = (
                        await self.store.embed_query("test")
                        if hasattr(self.store, "embed_query")
                        else None
                    )
                    if test_vec is not None:
                        memory_diagnostic = (
                            "✅ Fully functional - embeddings and attention working"
                        )
                    else:
                        memory_diagnostic = "⚠️ Methods exist but embedding returns None - semantic_memory plugin may not be started"
                except Exception as e:
                    memory_diagnostic = (
                        f"❌ Methods exist but fail on execution: {str(e)[:100]}..."
                    )
            else:
                missing = []
                if not hasattr(self.store, "embed_query"):
                    missing.append("embed_query")
                if not hasattr(self.store, "attention"):
                    missing.append("attention")
                memory_diagnostic = f"❌ Missing critical methods: {missing}. Available methods: {len(store_methods)}"
        else:
            memory_diagnostic = "❌ NeuralStore not available"

        response = f"""## 🔍 Super Alita Self-Assessment & Limitations

**You asked**: "{user_message}"

### **System Health Check**

**🔄 Core Architecture** {"✅" if eventbus_available else "❌"}
• Event Bus: {"Running - Redis distributed communication active" if eventbus_available else "Issues detected - check Redis connection"}
• Plugin System: ✅ Conversation plugin operational
• Neural Store: {"✅ Connected" if self.store else "❌ Not available"}

**🧠 Cognitive Capabilities**
• LLM Integration: {"✅ Gemini client active" if llm_available else "❌ Gemini unavailable (package compatibility issue)"}
• Memory System: {memory_diagnostic}
• Intent Classification: {"✅ Working" if llm_available else "⚠️ Basic pattern matching only"}
• Reasoning: {"✅ Full LLM reasoning" if llm_available else "⚠️ Limited to fallback logic"}

**🎯 Task Execution**
• Goal Reception: ✅ Can receive and delegate task requests
• Planning Integration: ✅ Connected to LADDER-AOG plugin
• Tool Execution: ⚠️ Configured but execution status unverified
• Web Search: ❌ Not confirmed active

### **Identified Limitations**

**Critical Issues** ❌
{"• GEMINI_API_KEY not set - this breaks LLM reasoning and memory embeddings" if not llm_available else "• Gemini API package compatibility issue - circular import detected"}
{"• Memory system issues - " + memory_diagnostic.replace("❌ ", "").replace("⚠️ ", "") if not memory_methods_available or "❌" in memory_diagnostic or "⚠️" in memory_diagnostic else ""}
{"• Event bus issues - distributed communication may be impaired" if not eventbus_available else ""}

**Degraded Functionality** ⚠️
• Can acknowledge task requests but actual execution capabilities unverified
• Limited to session-only memory without semantic memory system
• Fallback responses instead of contextual, intelligent replies

### **What Needs to be Fixed**

**Priority 1: Environment Setup**
{api_setup_msg if not api_key_set else package_fix_msg}

**Priority 2: Memory System**
{memory_fix_msg_prefix + memory_diagnostic + memory_fix_msg_suffix if not memory_methods_available or "❌" in memory_diagnostic or "⚠️" in memory_diagnostic else memory_fix_ok}

**Priority 3: Full System Test**
• Test actual web search capability
• Verify tool execution pipeline
• Validate end-to-end planning and execution

### **What's Actually Working Well**

✅ **Event-driven architecture is solid** - the plugin system and event bus provide a robust foundation
✅ **Conversation flow is functional** - I can process messages, classify intents, and maintain session context
✅ **Task delegation works** - I successfully route complex requests to the planning system
✅ **Self-awareness is accurate** - I can honestly assess my own limitations and provide actionable guidance

### **Honest Assessment**

{"I'm currently running in **degraded mode** due to missing API key and/or memory system issues. While my core architecture is sound, I'm using fallback responses instead of intelligent LLM reasoning, and I can't access past conversation context or provide memory-grounded responses." if not llm_available or not memory_methods_available or "❌" in memory_diagnostic else "I'm running in **full operational mode** with LLM reasoning and memory grounding active. My cognitive capabilities are functioning as designed."}

The good news: Once the environment issues are resolved, I should transition from "compliance mode" to "intelligent agent mode" with full reasoning, memory, and execution capabilities.

**Next Steps**: {"Fix the API key and memory system, then restart. That should unlock my full cognitive capabilities." if not api_key_set or not memory_methods_available else "Test my capabilities with some complex requests to verify full system operation."}

**Debug Info for Developer**:
• Store methods available: {len([method for method in dir(self.store) if not method.startswith("_")]) if self.store else 0}
• Memory methods check: embed_query={hasattr(self.store, "embed_query") if self.store else False}, attention={hasattr(self.store, "attention") if self.store else False}
• Plugins that should extend store: semantic_memory_plugin"""

        critical_issues = sum(
            [
                not llm_available,
                not memory_methods_available,
                not eventbus_available,
                not api_key_set,
                "❌" in memory_diagnostic,
            ]
        )

        reasoning = f"Comprehensive self-assessment performed - identified {critical_issues} critical issues, memory diagnostic: {memory_diagnostic[:50]}..."
        return response, reasoning

    def _get_technical_response(self) -> tuple[str, str]:
        """Provide technical architecture details."""
        response = """My technical architecture is quite sophisticated:

🏗️ **Core Systems**:
• NeuralAtom/NeuralStore: Reactive state management with genealogy tracking
• EventBus: Async publish/subscribe communication between all components
• Plugin System: Hot-swappable modular capabilities

🔌 **Active Plugins**:
• Semantic Memory: ChromaDB + Google text-embedding-004 embeddings
• Semantic FSM: Embedding-based state transitions (IDLE/PLANNING/EXECUTING/REFLECTING)
• Skill Discovery: MCTS-based skill evolution and learning
• LADDER-AOG: And-Or Graph reasoning with MCTS planning
• Conversation: This plugin handling our chat!

⚡ **Performance**: Running on RTX 3060 with GPU acceleration for embeddings
📊 **Monitoring**: Real-time health checks and genealogy tracking for all cognitive operations

I'm essentially a distributed cognitive system where each plugin contributes specialized intelligence!"""

        reasoning = "Providing detailed technical architecture explanation"
        return response, reasoning

    async def _generate_analysis_response(
        self, user_message: str, context: dict
    ) -> tuple[str, str]:
        """Generate analytical response."""
        # Extract the topic to analyze
        topic = (
            user_message.replace("analyze", "")
            .replace("explain", "")
            .replace("what do you think about", "")
            .strip()
        )

        response = f"""Let me analyze "{topic}" for you:

🔍 **Initial Observations**: This is an interesting topic that requires multi-faceted consideration.

🧠 **My Analysis**:
• **Complexity**: This topic involves multiple interconnected factors
• **Context**: Understanding the broader context is crucial for proper analysis
• **Perspectives**: There are likely multiple valid viewpoints to consider
• **Implications**: The outcomes and consequences deserve careful thought

💡 **Insights**:
Based on my reasoning capabilities, I can see this involves both analytical and creative thinking. I'd recommend breaking it down into smaller components for deeper understanding.

What specific aspect would you like me to focus on more deeply?"""

        reasoning = f"Provided structured analysis framework for topic: {topic[:50]}..."
        return response, reasoning

    async def _generate_learning_response(
        self, user_message: str, context: dict
    ) -> tuple[str, str]:
        """Generate learning-focused response."""
        response = """I'd be happy to help you learn! My approach to teaching combines:

📚 **Structured Learning**:
• Breaking complex topics into manageable pieces
• Providing clear explanations with examples
• Building understanding step-by-step

🧠 **Cognitive Support**:
• Using my semantic memory to find relevant connections
• Applying different reasoning approaches
• Adapting explanations to your learning style

🔄 **Interactive Process**:
• Ask me questions - I learn from our conversation too!
• Request clarification when needed
• Let's explore topics from multiple angles

What would you like to learn about? I can explain concepts, walk through processes, or help you understand complex ideas."""

        reasoning = "Offering comprehensive learning support with interactive approach"
        return response, reasoning

    async def _generate_problem_solving_response(
        self, user_message: str, context: dict
    ) -> tuple[str, str]:
        """Generate problem-solving response."""
        response = """I'd love to help solve your problem! My problem-solving approach uses:

🎯 **LADDER-AOG Reasoning**:
• Break down the problem into sub-goals (And-Or Graph structure)
• Explore multiple solution paths
• Use MCTS to find optimal approaches

🧩 **Systematic Process**:
1. **Problem Definition**: Clearly understand what needs to be solved
2. **Analysis**: Identify constraints, resources, and requirements
3. **Solution Generation**: Create multiple potential approaches
4. **Evaluation**: Assess pros/cons of each option
5. **Implementation Planning**: Create actionable steps

💡 **Collaborative Approach**:
• I'll work with you to understand your specific situation
• We can iterate and refine solutions together
• I'll explain my reasoning so you can follow the logic

What problem would you like to work on together?"""

        reasoning = "Offering structured problem-solving methodology with collaborative approach"
        return response, reasoning

    async def _generate_general_response(
        self, user_message: str, context: dict, session: dict
    ) -> tuple[str, str]:
        """Generate a thoughtful general response."""

        # Try to identify key topics in the message
        message_length = len(user_message.split())

        if message_length < 5:
            response = f"""That's an interesting point about "{user_message}".

I'd love to explore this topic further with you. Could you tell me more about what specifically interests you or what you'd like to discuss?

My cognitive systems are ready to engage with whatever direction you'd like to take our conversation!"""
            reasoning = "Short message detected, requesting elaboration to provide better assistance"

        else:
            response = """Thank you for sharing that with me. I find your message quite thought-provoking.

🤔 **My Perspective**: What you've described touches on several interesting dimensions that I can help explore further.

🧠 **How I Can Help**:
• Analyze different aspects of this topic
• Provide additional context or information
• Help you think through implications
• Suggest related areas to explore

💬 **Continuing Our Discussion**:
I'm genuinely interested in understanding your thoughts better. What aspects of this topic are most important to you, or where would you like to focus our conversation?"""
            reasoning = f"Engaging with substantive message ({message_length} words), showing interest and offering multiple ways to continue"

        return response, reasoning

    async def _handle_status_request(self, event) -> None:
        """Handle system status requests from chat interface."""
        try:
            # Extract data from event object - handle WorkspaceEvent correctly
            if hasattr(event, "data") and isinstance(event.data, dict):
                data = event.data
            else:
                data = event.model_dump() if hasattr(event, "model_dump") else event

            # Skip non-status events
            event_type = data.get("type", "")
            if event_type in [
                "communication_test",
                "process_stage",
                "task_request",
                "task_result",
            ]:
                # These are test/system events, not status requests
                return

            session_id = data.get("session_id")

            # Emit detailed status
            status_data = {
                "session_id": session_id,
                "system_status": {
                    "conversation_plugin": "healthy",
                    "active_sessions": len(self.active_sessions),
                    "total_conversations": len(self.conversation_memory),
                    "capabilities": [
                        "Natural Language Processing",
                        "Contextual Understanding",
                        "Multi-turn Dialogue",
                        "Reasoning Explanation",
                        "Memory Integration",
                    ],
                },
                "timestamp": datetime.now().isoformat(),
            }

            await self.emit_event("system_status_response", **status_data)

        except Exception as e:
            logger.error(f"Error handling status request: {e}")
