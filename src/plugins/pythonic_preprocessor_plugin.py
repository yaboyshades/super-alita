"""
Pythonic Preprocessor Plugin - DTA 2.0 Integration

This plugin serves as the intelligent "airlock" between user inputs and the system,
converting natural language into validated Python function calls using DTA 2.0.

Key Features:
- Intent detection and classification
- Multi-intent handling and clarification requests
- Code generation routing
- Comprehensive validation pipeline
- Performance monitoring and caching
- Full Super Alita integration
"""

import contextlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

# Core imports - moved to top level
from src.core.events import (
    BaseEvent,
    CognitiveTurnCompletedEvent,
    CognitiveTurnInitiatedEvent,
    ConversationEvent,
)
from src.core.gemini_utils import get_gemini_client, is_gemini_available
from src.core.plugin_interface import PluginInterface
from src.plugins._emit import aemit_safe
from src.utils.event_builders import build_tool_call_event

# DTA components - conditional imports with graceful degradation
DTA_RUNTIME_AVAILABLE = False
try:
    from src.dta import AsyncDTARuntime, DTAContext, DTARequest, DTAResult, DTAStatus
    from src.dta.cache import DTACache, create_cache
    from src.dta.config import DTAConfig, create_default_config
    from src.dta.monitoring import DTAMonitoring, create_monitoring

    DTA_RUNTIME_AVAILABLE = True
except ImportError:
    # Graceful degradation - create minimal stubs for type hints
    class DTAContext:  # type: ignore
        def __init__(self, **kwargs):
            pass

    class DTARequest:  # type: ignore
        def __init__(self, **kwargs):
            pass

    class DTAResult:  # type: ignore
        def __init__(self, **kwargs):
            self.status = "ERROR"
            self.metadata = {}

    class DTAStatus:  # type: ignore
        SUCCESS = "SUCCESS"
        ERROR = "ERROR"

    class DTAConfig:  # type: ignore
        def __init__(self, **kwargs):
            pass

    class AsyncDTARuntime:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        async def process_request(self, request):
            return DTAResult(
                status=DTAStatus.ERROR, metadata={"error": "DTA runtime not available"}
            )

        async def shutdown(self):
            pass

    class DTACache:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        async def get(self, *args, **kwargs):
            return None

        async def set(self, *args, **kwargs):
            pass

        async def close(self):
            pass

        async def get_stats(self):
            return {}

    class DTAMonitoring:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.logger = logging.getLogger(__name__)

        async def shutdown(self):
            pass

    def create_cache(*args, **kwargs):
        return DTACache()

    def create_monitoring(*args, **kwargs):
        return DTAMonitoring()

    def create_default_config():
        return DTAConfig()


# Configure logging early
logger = logging.getLogger(__name__)

# Constants
CONFIDENCE_THRESHOLD_DEFAULT = 0.3
HIGH_CONFIDENCE_THRESHOLD = 0.6
MIN_INPUT_LENGTH = 3
SELF_REFLECTION_MIN_LENGTH = 10
LLM_CONFIDENCE_THRESHOLD = 0.7
ENHANCED_PROTOCOL_MIN_LENGTH = 50
CACHE_TTL_SECONDS = 300

# Regex patterns for input normalization
_FENCE = re.compile(r"^```(?:\w+)?\s*([\s\S]*?)\s*```$", re.MULTILINE)

# Enhanced Protocol v2.0 Integration (optional)
ENHANCED_PROTOCOL_AVAILABLE = False
try:
    from src.core.enhanced_protocol import (
        CognitiveRequest,
        EnhancedProtocolEngine,
        ExpansionTier,
    )

    ENHANCED_PROTOCOL_AVAILABLE = True
except ImportError:
    with contextlib.suppress(Exception):
        logger.warning(
            "Enhanced Protocol v2.0 not available - using legacy processing only"
        )

# DTA 2.0 Cognitive Turn imports
COGNITIVE_TURN_AVAILABLE = False
try:
    from src.dta.cognitive_plan import generate_master_cognitive_plan
    from src.dta.types import CognitiveTurnRecord, StrategicPlan

    COGNITIVE_TURN_AVAILABLE = True
except ImportError:
    pass

# Event imports (including cognitive turns)
COGNITIVE_EVENT_AVAILABLE = False
try:
    from src.core.events import CognitiveTurnCompletedEvent

    COGNITIVE_EVENT_AVAILABLE = True
except ImportError:
    pass

# LLM imports (conditional)
GEMINI_AVAILABLE = is_gemini_available()
GEMINI_TYPES_AVAILABLE = False

if GEMINI_AVAILABLE:
    try:
        from google.generativeai.types import HarmBlockThreshold, HarmCategory

        GEMINI_TYPES_AVAILABLE = True
    except ImportError:
        pass


@dataclass
class IntentDetectionResult:
    """Result of intent detection analysis."""

    primary_intent: str
    confidence: float
    is_multi_intent: bool = False
    secondary_intents: list[str] = field(default_factory=list)
    requires_clarification: bool = False
    clarification_reason: str = ""
    extracted_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingMetrics:
    """Metrics for processing performance."""

    start_time: float
    end_time: float | None = None
    intent_detection_time: float | None = None
    validation_time: float | None = None
    llm_processing_time: float | None = None
    cache_hit: bool = False
    total_processing_time: float | None = None


class PythonicPreprocessorPlugin(PluginInterface):
    """
    Pythonic Preprocessor Plugin implementing DTA 2.0 architecture.
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "pythonic_preprocessor"

        # Core components
        self.dta_runtime: AsyncDTARuntime | None = None
        self.dta_config: DTAConfig | None = None
        self.monitoring: DTAMonitoring | None = None
        self.cache: DTACache | None = None

        # LLM components
        self.gemini_client = None
        self.gemini_model = None

        # Intent patterns (comprehensive regex-based detection)
        self._setup_intent_patterns()

        # Processing metrics
        self.processing_metrics: dict[str, ProcessingMetrics] = {}

        # Configuration flags
        self.thinking_mode_enabled = True
        self.multi_intent_detection = True
        self.clarification_threshold = CONFIDENCE_THRESHOLD_DEFAULT

        # Enhanced Protocol v2.0 Integration
        self.enhanced_engine: Any | None = None
        self.enhanced_protocol_enabled = False
        if ENHANCED_PROTOCOL_AVAILABLE:
            self._init_enhanced_protocol()

        # Deduplication tracking
        self._processed_messages: set = set()
        self._last_cleanup = time.time()

    def _setup_intent_patterns(self) -> None:
        """Setup intent detection patterns."""
        self.intent_patterns = {
            "atomize": [
                r"\batomize\s*:\s*",
                r"^atomize\b",
                r"\batomize\s+[^\s]",
            ],
            "search": [
                r"\b(?:search|find|look\s+(?:for|up)|query|discover)\b.*?"
                r"\b(?:for|about|on)\b",
                r"\b(?:what|where|how|when|why)\b.*?\?",
                r"\b(?:find|locate|get)\s+(?:me\s+)?"
                r"(?:info|information|data|details)\b",
                r"\b(?:tell\s+me|show\s+me|give\s+me)\b.*?"
                r"\b(?:about|regarding|on)\b",
            ],
            "memory": [
                r"\b(?:remember|recall|save|store|memorize)\b",
                r"\b(?:what\s+(?:do\s+)?(?:you\s+)?(?:remember|know|recall))\b",
                r"\b(?:my\s+)?(?:memory|memories|notes?|reminders?)\b",
                r"\b(?:add\s+to|save\s+to|store\s+in)\s+(?:memory|notes?)\b",
            ],
            "tool_request": [
                r"\b(?:create|make|build|generate|develop)\s+(?:a\s+)?"
                r"(?:tool|function|script|utility)\b",
                r"\b(?:I\s+need|can\s+you\s+(?:create|make|build))\s+(?:a\s+)?"
                r"(?:tool|function)\b",
                r"\b(?:tool|function|script|utility)\s+(?:for|to|that)\b",
                r"\b(?:new\s+)?(?:tool|function|script)\b",
            ],
            "code_generation": [
                r"\b(?:write|generate|create|code|program|script)\s+"
                r"(?:code|a\s+function|a\s+script)\b",
                r"\b(?:help\s+me\s+)?(?:code|program|write|implement)\b",
                r"\b(?:function|class|method|algorithm)\s+(?:to|for|that)\b",
                r"\b(?:python|javascript|java|c\+\+|code)\s+"
                r"(?:function|script|program)\b",
            ],
            "clarification": [
                r"\b(?:what\s+(?:do\s+)?you\s+mean|can\s+you\s+clarify|"
                r"I\s+don\'?t\s+understand)\b",
                r"\b(?:unclear|ambiguous|confusing|vague)\b",
                r"\b(?:explain|elaborate|clarify|specify)\b",
                r"\b(?:more\s+details|be\s+more\s+specific|"
                r"can\s+you\s+be\s+clearer)\b",
            ],
            "greeting": [
                r"\b(?:hi|hello|hey|greetings|good\s+"
                r"(?:morning|afternoon|evening))\b",
                r"\b(?:how\s+are\s+you|what\'?s\s+up|how\'?s\s+it\s+going)\b",
            ],
            "self_reflection": [
                r"\b(?:assess|evaluate|analyze|describe|tell\s+me\s+about)\s+"
                r"(?:your|my)\s+(?:abilities|capabilities|skills|strengths|"
                r"weaknesses)\b",
                r"\b(?:what\s+(?:can|are)\s+you\s+(?:able\s+to\s+)?"
                r"(?:do|capable\s+of))\b",
                r"\b(?:list|show|display)\s+(?:your|my)\s+"
                r"(?:abilities|capabilities|skills|functions)\b",
                r"\b(?:how\s+(?:good|well|capable)\s+are\s+you)\b",
                r"\b(?:what\s+are\s+your\s+(?:main\s+)?"
                r"(?:features|functions|capabilities))\b",
            ],
            "general_question": [
                r"\b(?:what|how|why|when|where|who)\b.*\?",
                r"\b(?:tell\s+me|explain|describe)\b",
                r"\b(?:can\s+you|are\s+you\s+able\s+to|do\s+you\s+know)\b",
            ],
        }

    @property
    def name(self) -> str:
        """Return the unique name identifier for this plugin."""
        return self._name

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        """Initialize the plugin with DTA 2.0 components."""
        await super().setup(event_bus, store, config)
        logger.info("Initializing Pythonic Preprocessor Plugin with DTA 2.0")

        try:
            # Only initialize DTA components if runtime is available
            if DTA_RUNTIME_AVAILABLE:
                # Initialize DTA 2.0 configuration
                if "dta" in config:
                    self.dta_config = DTAConfig(**config["dta"])
                else:
                    self.dta_config = create_default_config()

                # Initialize components
                self.dta_runtime = AsyncDTARuntime(self.dta_config)
                self.monitoring = await self._initialize_monitoring()
                self.cache = await self._initialize_cache()

                logger.info("DTA 2.0 components initialized successfully")
            else:
                # Graceful degradation - use stub components
                self.dta_config = DTAConfig()
                self.dta_runtime = AsyncDTARuntime()
                self.monitoring = DTAMonitoring()
                self.cache = DTACache()

                logger.warning("DTA runtime not available - using fallback components")

            # Initialize LLM (independent of DTA)
            await self._initialize_llm(config)

        except Exception as e:
            logger.exception("Failed to initialize DTA 2.0 components: %s", e)
            # Fallback to stub components if initialization fails
            self.dta_config = DTAConfig()
            self.dta_runtime = AsyncDTARuntime()
            self.monitoring = DTAMonitoring()
            self.cache = DTACache()
            logger.warning("Using fallback components due to initialization error")

    async def start(self) -> None:
        """Start the plugin and subscribe to events."""
        await super().start()

        # Subscribe to events
        await self.subscribe("cognitive_turn_initiated", self._handle_cognitive_turn)
        await self.subscribe("conversation", self._handle_conversation_event)
        await self.subscribe("message", self._handle_message_event)

        logger.info("Pythonic Preprocessor Plugin started and subscribed to events")

    async def shutdown(self) -> None:
        """Gracefully shutdown the plugin."""
        if self.dta_runtime:
            await self.dta_runtime.shutdown()

        if self.monitoring:
            await self.monitoring.shutdown()

        if self.cache:
            await self.cache.close()

        logger.info("Pythonic Preprocessor Plugin shutdown complete")

    async def _initialize_monitoring(self) -> DTAMonitoring:
        """Initialize monitoring with custom metrics."""
        if not DTA_RUNTIME_AVAILABLE or not self.dta_config:
            return DTAMonitoring()

        raw_cfg = getattr(self.dta_config, "monitoring", None)
        cfg_dict = self._extract_config_dict(raw_cfg)
        monitoring = create_monitoring(cfg_dict)
        logger.info("Preprocessor monitoring initialized")
        return monitoring

    async def _initialize_cache(self) -> DTACache:
        """Initialize caching for performance optimization."""
        if not DTA_RUNTIME_AVAILABLE or not self.dta_config:
            return DTACache()

        raw_cfg = getattr(self.dta_config, "cache", None)
        cfg_dict = self._extract_config_dict(raw_cfg)
        return create_cache(cfg_dict)

    def _extract_config_dict(self, raw_cfg: Any) -> dict[str, Any] | None:
        """Extract configuration dictionary from various config types."""
        if raw_cfg is None:
            return None

        if hasattr(raw_cfg, "model_dump"):
            return raw_cfg.model_dump()
        if hasattr(raw_cfg, "__dict__"):
            return dict(raw_cfg.__dict__)
        if isinstance(raw_cfg, dict):
            return raw_cfg
        return None

    async def _initialize_llm(self, config: dict[str, Any]) -> None:
        """Initialize Gemini LLM client if available and configured."""
        if not GEMINI_AVAILABLE:
            logger.warning(
                "Gemini not available - using template-based generation only"
            )
            return

        api_key = self._get_api_key(config)
        if not api_key:
            logger.warning(
                "No Gemini API key provided - using template-based generation only"
            )
            return

        try:
            genai_client = get_gemini_client()
            if not genai_client:
                logger.error("Gemini client not available")
                return

            genai_client.configure(api_key=api_key)

            safety_settings = {}
            if GEMINI_TYPES_AVAILABLE:
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }

            self.gemini_model = genai_client.GenerativeModel(
                model_name="gemini-pro", safety_settings=safety_settings
            )
            self.gemini_client = genai_client
            logger.info("Gemini LLM initialized successfully")

        except Exception as e:
            logger.exception("Failed to initialize Gemini LLM: %s", e)
            self.gemini_client = None
            self.gemini_model = None

    def _get_api_key(self, config: dict[str, Any]) -> str | None:
        """Get API key from configuration with fallbacks."""
        api_key = None

        # Try nested config structure
        for plugin_config in config.values():
            if isinstance(plugin_config, dict):
                api_key = plugin_config.get("gemini_api_key")
                if api_key:
                    break

        # Fallback to direct config access
        if not api_key:
            api_key = config.get("gemini_api_key") or config.get("llm", {}).get(
                "api_key"
            )

        # Final fallback to environment variable
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")

        # Expand environment variable references if needed
        if api_key and api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]  # Remove ${ and }
            api_key = os.getenv(env_var)

        return api_key

    # Helper to emit events with proper channel + payload signature
    async def _emit_event(self, channel: str, event: BaseEvent) -> None:
        """Emit events with proper channel + payload signature."""
        try:
            await aemit_safe(self, channel, event)
        except Exception as e:
            logger.exception("Failed emitting on %s: %s", channel, e)

    async def _handle_conversation_event(self, event: ConversationEvent) -> None:
        """Enhanced conversation handler with cognitive turn processing."""
        try:
            # Extract message content
            user_message = getattr(event, "message", str(event))
            session_id = str(getattr(event, "conversation_id", "default"))
            conversation_id = str(getattr(event, "conversation_id", session_id))
            context = getattr(event, "context", {})

            # Process through cognitive turn if available
            if COGNITIVE_TURN_AVAILABLE:
                cognitive_turn = await self._process_cognitive_turn(
                    user_message, context, session_id
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
                        session_id=session_id,
                        conversation_id=conversation_id,
                    )
                    await self._emit_event("cognitive_turn_completed", turn_event)
                    logger.info("Published cognitive turn for session %s", session_id)

            # Process the message through legacy DTA pipeline
            result = await self.process_user_input(user_message, session_id)

            # Route based on processing result
            await self._route_processing_result(result, event)

        except Exception as e:
            logger.exception("Error handling conversation event: %s", e)
            await self._emit_error_response(event, str(e))

    async def _handle_cognitive_turn(self, event: CognitiveTurnInitiatedEvent) -> None:
        """Handle cognitive turn initiated events from the ConversationPlugin."""
        try:
            # Deduplication: Create a unique key for this message
            message_key = f"{event.session_id}:{event.user_message}:{int(time.time() // 10)}"  # 10-second window

            # Clean up old processed messages periodically
            current_time = time.time()
            if current_time - self._last_cleanup > 300:  # 5 minutes
                old_keys = [
                    k
                    for k in self._processed_messages
                    if current_time - float(k.split(":")[-1]) * 10 > 300
                ]
                for old_key in old_keys:
                    self._processed_messages.discard(old_key)
                self._last_cleanup = current_time

            # Skip if we've already processed this message recently
            if message_key in self._processed_messages:
                logger.debug(
                    "Skipping duplicate cognitive turn for: '%s'", event.user_message
                )
                return

            # Mark as processed
            self._processed_messages.add(message_key)

            logger.info(
                "🧠 Cognitive turn initiated for: '%s' (session: %s)",
                event.user_message,
                event.session_id,
            )

            # Process through the DTA 2.0 cognitive pipeline
            result = await self.process_user_input(event.user_message, event.session_id)

            # Route the processing result to the appropriate handler
            await self._route_processing_result(result, event)

            logger.info("✅ Cognitive turn completed for session %s", event.session_id)

        except Exception as e:
            logger.exception("Error in cognitive turn processing: %s", e)
            await self._emit_error_response(event, str(e))

    async def _handle_message_event(self, event: BaseEvent) -> None:
        """Handle generic message events."""
        # Convert BaseEvent to ConversationEvent for processing
        if isinstance(event, ConversationEvent):
            await self._handle_conversation_event(event)
        else:
            # Create a minimal ConversationEvent from BaseEvent
            conversation_event = ConversationEvent(
                source_plugin=getattr(event, "source_plugin", "unknown"),
                message=str(event),
                session_id=getattr(event, "session_id", "default"),
                conversation_id=getattr(event, "conversation_id", "default"),
                message_id=getattr(event, "message_id", f"msg_{id(event)}"),
            )
            await self._handle_conversation_event(conversation_event)

    async def _process_cognitive_turn(
        self, user_message: str, context: dict[str, Any], session_id: str
    ) -> CognitiveTurnRecord | None:
        """Process user input through cognitive turn methodology."""

        if not COGNITIVE_TURN_AVAILABLE:
            return None

        try:
            # Build cognitive turn analysis
            turn_data = {
                "state_readout": f"Processing user request: {user_message}",
                "activation_protocol": {
                    "pattern_recognition": self._detect_cognitive_pattern(user_message),
                    "confidence_score": 8,
                    "planning_requirement": self._requires_planning(user_message),
                    "quality_speed_tradeoff": "balance",
                    "evidence_threshold": "medium",
                    "audience_level": "professional",
                    "meta_cycle_check": "analysis",
                },
                "strategic_plan": {
                    "is_required": self._requires_planning(user_message)
                },
                "execution_log": [
                    "Cognitive turn initiated",
                    "User input analyzed",
                    "Processing strategy determined",
                ],
                "synthesis": {
                    "key_findings": [
                        f"User request: {user_message}",
                        "Cognitive processing applied",
                        "Structured analysis complete",
                    ],
                    "counterarguments": [],
                    "final_answer_summary": f"Cognitive analysis complete for: {user_message}",
                },
                "state_update": {
                    "directive": "memory_stream_add",
                    "memory_stream_add": {
                        "summary": f"Processed cognitive turn for: {user_message}",
                        "timestamp": datetime.now().isoformat(),
                        "type": "cognitive_processing",
                    },
                },
                "confidence_calibration": {
                    "final_confidence": 8,
                    "uncertainty_gaps": "Minimal uncertainty",
                    "risk_assessment": "low",
                    "verification_methods": ["cognitive_analysis"],
                },
            }

            # Create cognitive turn record
            turn_record = cast("Any", CognitiveTurnRecord)(**turn_data)  # type: ignore[call-arg]

            # Generate strategic plan if required
            if getattr(turn_record, "activation_protocol", None) and getattr(
                turn_record.activation_protocol, "planning_requirement", False
            ):
                try:
                    plan = generate_master_cognitive_plan(user_message)  # type: ignore
                    turn_record.strategic_plan = StrategicPlan(  # type: ignore[attr-defined]
                        is_required=True,
                        steps=getattr(plan, "execution_steps", []),
                        estimated_duration=60.0,
                        resource_requirements=["cognitive_processing"],
                    )
                except Exception:
                    pass

            logger.info("Generated cognitive turn")
            return turn_record

        except Exception as e:
            logger.exception("Error in cognitive turn processing: %s", e)
            return None

    def _detect_cognitive_pattern(self, user_message: str) -> str:
        """Detect the cognitive pattern type from user message."""
        message_lower = user_message.lower()

        if any(word in message_lower for word in ["analyze", "explain", "why", "how"]):
            return "analytical"
        if any(
            word in message_lower for word in ["create", "build", "generate", "make"]
        ):
            return "creative"
        if any(word in message_lower for word in ["fix", "debug", "error", "problem"]):
            return "diagnostic"
        if any(word in message_lower for word in ["plan", "strategy", "approach"]):
            return "strategic"
        return "exploratory"

    def _requires_planning(self, user_message: str) -> bool:
        """Determine if the request requires strategic planning."""
        planning_keywords = [
            "plan",
            "strategy",
            "roadmap",
            "architecture",
            "design",
            "structure",
        ]
        return any(keyword in user_message.lower() for keyword in planning_keywords)

    async def process_user_input(
        self, user_input: str, session_id: str = "default"
    ) -> DTAResult:
        """
        Main processing pipeline for user input.

        Args:
            user_input: Raw user input text
            session_id: Session identifier for context tracking

        Returns:
            DTAResult with processing outcome and routing information
        """
        start_time = time.time()
        metrics = ProcessingMetrics(start_time=start_time)

        try:
            # Create DTA request context
            context = DTAContext(
                session_id=session_id,
                timestamp=datetime.now(),
                user_id="user",
                metadata={},
            )

            # Enhanced Protocol v2.0 Integration - optional fast path
            # Feature flag integration (currently disabled unless flag enabled)
            # Enhanced Protocol fast path intentionally disabled for now. The
            # prior implementation is retained here (commented) for future
            # reactivation once flag gating + tests are ready.
            # from src.core.config.flags import enhanced_protocol_enabled
            # if ENHANCED_PROTOCOL_AVAILABLE and enhanced_protocol_enabled():
            #     if await self._should_use_enhanced_protocol(user_input):
            #         enhanced_result = await self._process_with_enhanced_protocol(user_input, session_id)
            #         if enhanced_result:
            #             logger.info("Using Enhanced Protocol v2.0 fast path")
            #             metrics.end_time = time.time()
            #             metrics.total_processing_time = metrics.end_time - metrics.start_time
            #             self.processing_metrics[session_id] = metrics
            #             return enhanced_result

            request = DTARequest(
                user_message=user_input,
                context=context,
                metadata={
                    "thinking_mode": self.thinking_mode_enabled,
                    "multi_intent_detection": self.multi_intent_detection,
                    "clarification_threshold": self.clarification_threshold,
                },
            )

            # Check cache first
            if self.cache:
                cached_result = await self.cache.get(
                    "preprocessor", {"input": user_input, "session": session_id}
                )
                if cached_result:
                    metrics.cache_hit = True
                    metrics.end_time = time.time()
                    metrics.total_processing_time = (
                        metrics.end_time - metrics.start_time
                    )
                    self.processing_metrics[session_id] = metrics
                    return cached_result

            # Detect intent
            intent_start = time.time()
            intent_result = await self._detect_intent(user_input)
            metrics.intent_detection_time = time.time() - intent_start

            # Ensure dta_runtime is initialized before processing request
            if not self.dta_runtime:
                raise RuntimeError("DTA runtime not initialized")

            # Process through DTA pipeline
            dta_result = await self.dta_runtime.process_request(request)

            # Enhance result with intent information
            dta_result.metadata.update(
                {"intent_detection": intent_result, "processing_metrics": metrics}
            )

            # Cache successful results
            if self.cache and dta_result.status == DTAStatus.SUCCESS:
                await self.cache.set(
                    "preprocessor",
                    {"input": user_input, "session": session_id},
                    dta_result,
                    ttl=300,
                )  # 5 minute TTL

            # Update metrics
            metrics.end_time = time.time()
            metrics.total_processing_time = metrics.end_time - metrics.start_time
            self.processing_metrics[session_id] = metrics

            # Record monitoring metrics
            if self.monitoring:
                # Use available monitoring methods
                self.monitoring.logger.info(
                    "Preprocessor request processed",
                    intent_detection_time=metrics.intent_detection_time or 0,
                )

            return dta_result

        except Exception as e:
            logger.exception("Error processing user input: %s", e)
            metrics.end_time = time.time()
            metrics.total_processing_time = metrics.end_time - metrics.start_time
            self.processing_metrics[session_id] = metrics

            # Return error result
            return DTAResult(
                status=DTAStatus.ERROR,
                python_code="",
                confidence_score=0.0,
                metadata={"processing_metrics": metrics, "error_message": str(e)},
            )

    async def _detect_intent(self, user_input: str) -> IntentDetectionResult:
        """
        Comprehensive intent detection using regex patterns and optional LLM enhancement.

        Args:
            user_input: User input text to analyze

        Returns:
            IntentDetectionResult with detected intent and metadata
        """
        user_input_lower = user_input.lower().strip()

        # PATCH: Early detection for atomize directive (highest priority)
        if re.search(r"\batomize\s*[:]\s*", user_input_lower, re.IGNORECASE):
            # Extract and clean content after atomize:
            content = user_input.split(":", 1)[-1].strip()

            # Handle fenced code blocks (removes ``` markers)
            m = _FENCE.match(content)
            if m:
                content = m.group(1).strip()

            return IntentDetectionResult(
                primary_intent="atomize",
                confidence=0.95,
                extracted_parameters={"content": content},
            )

        # Initialize result
        result = IntentDetectionResult(
            primary_intent="general_question", confidence=0.0
        )

        # Pattern matching with confidence scoring
        intent_scores = {}

        for intent, patterns in self.intent_patterns.items():
            max_score = 0.0
            for pattern in patterns:
                if re.search(pattern, user_input_lower, re.IGNORECASE):
                    # Base score for pattern match
                    score = 0.8

                    # Boost score for exact keyword matches
                    if (
                        intent == "search"
                        and any(
                            kw in user_input_lower
                            for kw in ["search", "find", "look for"]
                        )
                    ) or (
                        intent == "memory"
                        and any(
                            kw in user_input_lower
                            for kw in ["remember", "save", "store"]
                        )
                    ):
                        score += 0.1
                    elif (
                        intent == "tool_request"
                        and any(
                            kw in user_input_lower
                            for kw in ["create tool", "make tool", "build tool"]
                        )
                    ) or (
                        intent == "code_generation"
                        and any(
                            kw in user_input_lower
                            for kw in ["write code", "generate code", "create function"]
                        )
                    ):
                        score += 0.15
                    elif intent == "self_reflection" and any(
                        kw in user_input_lower
                        for kw in [
                            "assess",
                            "abilities",
                            "capabilities",
                            "what can you do",
                        ]
                    ):
                        score += 0.2  # High confidence boost for self-reflection

                    max_score = max(max_score, score)

            if max_score > 0:
                intent_scores[intent] = max_score

        # Determine primary intent
        if intent_scores:
            result.primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            result.confidence = intent_scores[result.primary_intent]

            # Check for multi-intent scenarios
            if self.multi_intent_detection:
                high_confidence_intents = [
                    intent
                    for intent, score in intent_scores.items()
                    if score > 0.6 and intent != result.primary_intent
                ]

                if high_confidence_intents:
                    result.is_multi_intent = True
                    result.secondary_intents = high_confidence_intents

        # Check if clarification is needed - bypass for self-reflection with reasonable input
        if (
            result.confidence < self.clarification_threshold
            or len(user_input.strip()) < 3
        ) and not (
            result.primary_intent == "self_reflection" and len(user_input.strip()) >= 10
        ):
            result.requires_clarification = True
            result.clarification_reason = "Intent unclear or input too brief"

        # Extract parameters based on intent
        result.extracted_parameters = await self._extract_parameters(
            user_input, result.primary_intent
        )

        # Enhance with LLM if available and confidence is low
        if self.gemini_model and result.confidence < 0.7:
            try:
                llm_result = await self._enhance_intent_with_llm(user_input, result)
                if llm_result and llm_result.confidence > result.confidence:
                    result = llm_result
            except Exception as e:
                logger.warning(f"LLM intent enhancement failed: {e}")

        return result

    async def _extract_parameters(self, user_input: str, intent: str) -> dict[str, Any]:
        """Extract relevant parameters based on detected intent."""
        parameters = {}

        if intent == "search":
            # Extract search query
            search_patterns = [
                r"(?:search|find|look)\s+(?:for|up)?\s*(.+?)(?:\?|$)",
                r"(?:what|where|how)\s+(.+?)\?",
                r"(?:tell|show|give)\s+me\s+(?:about\s+)?(.+?)(?:\?|$)",
            ]

            for pattern in search_patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    parameters["query"] = match.group(1).strip()
                    break

        elif intent == "memory":
            # Extract memory content
            if "remember" in user_input.lower():
                memory_match = re.search(
                    r"remember\s+(.+?)(?:\?|$)", user_input, re.IGNORECASE
                )
                if memory_match:
                    parameters["content"] = memory_match.group(1).strip()

        elif intent == "tool_request":
            # Extract tool description
            tool_patterns = [
                r"(?:create|make|build)\s+(?:a\s+)?tool\s+(?:for|to|that)\s+(.+?)(?:\?|$)",
                r"tool\s+(?:for|to|that)\s+(.+?)(?:\?|$)",
            ]

            for pattern in tool_patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    parameters["description"] = match.group(1).strip()
                    break

        elif intent == "code_generation":
            # Extract code requirements
            code_patterns = [
                r"(?:write|generate|create)\s+(?:code|function|script)\s+(?:for|to|that)\s+(.+?)(?:\?|$)",
                r"function\s+(?:to|for|that)\s+(.+?)(?:\?|$)",
            ]

            for pattern in code_patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    parameters["requirements"] = match.group(1).strip()
                    break

        return parameters

    async def _enhance_intent_with_llm(
        self, user_input: str, current_result: IntentDetectionResult
    ) -> IntentDetectionResult | None:
        """Use LLM to enhance intent detection for ambiguous cases."""
        if not self.gemini_model:
            return None

        try:
            prompt = f"""
            Analyze the following user input and determine the most likely intent:

            User Input: "{user_input}"

            Current Analysis:
            - Detected Intent: {current_result.primary_intent}
            - Confidence: {current_result.confidence}

            Available Intents:
            - search: User wants to find information
            - memory: User wants to store or retrieve memories
            - tool_request: User wants to create a new tool
            - code_generation: User wants code written
            - clarification: User needs clarification
            - greeting: User is greeting
            - general_question: General question or conversation

            Respond with just the intent name and confidence (0.0-1.0), separated by a comma.
            Example: "search,0.85"
            """

            response = await self.gemini_model.generate_content_async(prompt)
            response_text = response.text.strip()

            if "," in response_text:
                intent, confidence_str = response_text.split(",", 1)
                intent = intent.strip()
                confidence = float(confidence_str.strip())

                if intent in self.intent_patterns and 0.0 <= confidence <= 1.0:
                    enhanced_result = IntentDetectionResult(
                        primary_intent=intent,
                        confidence=confidence,
                        extracted_parameters=current_result.extracted_parameters,
                    )
                    return enhanced_result

        except Exception as e:
            logger.warning(f"LLM intent enhancement failed: {e}")

        return None

    async def _route_processing_result(
        self, result: DTAResult, original_event: BaseEvent
    ):
        """Route the processing result to appropriate handlers."""
        try:
            if not getattr(result, "success", False):
                await self._emit_error_response(
                    original_event,
                    getattr(result, "error_message", "Processing failed")
                    or "Processing failed",
                )
                return
            intent_info = result.metadata.get("intent_detection", {})
            primary_intent = str(
                getattr(
                    intent_info, "primary_intent", getattr(result, "intent", "") or ""
                )
            )
            if getattr(intent_info, "requires_clarification", False):
                await self._emit_clarification_request(original_event, intent_info)  # type: ignore[arg-type]
                return
            if getattr(intent_info, "is_multi_intent", False):
                await self._handle_multi_intent(result, original_event, intent_info)  # type: ignore[arg-type]
                return
            if primary_intent in ["search", "memory", "tool_request"]:
                await self._emit_tool_call(result, original_event)
            elif primary_intent == "code_generation":
                await self._handle_code_generation(result, original_event)
            elif primary_intent in ["greeting", "general_question"]:
                await self._emit_direct_response(result, original_event)
            else:
                await self._emit_preprocessed_action(result, original_event)
        except Exception as e:
            logger.exception("Error routing processing result: %s", e)
            await self._emit_error_response(original_event, f"Routing failed: {e}")

    async def _emit_tool_call(self, result: DTAResult, original_event: BaseEvent):
        """Emit a tool call event for tool-based intents."""
        try:
            intent_info = result.metadata.get("intent_detection", {})
            intent = str(
                getattr(
                    intent_info, "primary_intent", getattr(result, "intent", "") or ""
                )
            )
            intent_to_tool = {
                "search": "web_agent",
                "memory": "memory_manager",
                "tool_request": "creator",
                "self_reflection": "self_reflection",
            }
            tool_name = intent_to_tool.get(intent, "default_tool")
            parameters = getattr(intent_info, "extracted_parameters", {})
            if not parameters and getattr(result, "function_call", None):
                parameters = {"query": result.function_call}
            tool_event = build_tool_call_event(
                source_plugin=self._name,
                tool_name=tool_name,
                parameters=parameters,
                conversation_id=getattr(original_event, "conversation_id", "default"),
                session_id=getattr(original_event, "session_id", "default"),
                message_id=getattr(original_event, "message_id", None),
            )
            await self._emit_event("tool_call", tool_event)
            logger.info("Emitted tool call for %s (%s)", tool_name, intent)
        except Exception as e:
            logger.exception("Error emitting tool call: %s", e)
            await self._emit_error_response(original_event, f"Tool call failed: {e}")

    async def _emit_preprocessed_action(
        self, result: DTAResult, original_event: BaseEvent
    ):
        try:
            # Fallback: emit a conversation event describing action
            msg = f"Action prepared (intent={getattr(result, 'intent', 'unknown')}): {getattr(result, 'function_call', '')}"
            action_event = ConversationEvent(  # type: ignore[call-arg]
                source_plugin=self._name,
                session_id=getattr(original_event, "session_id", "default"),
                user_message=msg,
                message_id=getattr(
                    original_event, "message_id", f"action_{id(original_event)}"
                ),
                metadata={"type": "preprocessed_action"},
            )
            await self._emit_event("conversation", action_event)
        except Exception as e:
            logger.exception("Error emitting preprocessed action: %s", e)
            await self._emit_error_response(
                original_event, f"Action emission failed: {e}"
            )

    async def _emit_clarification_request(
        self, original_event: BaseEvent, intent_info: IntentDetectionResult
    ):
        try:
            clarification_text = f"I need clarification: {intent_info.clarification_reason}. Could you please be more specific?"
            clarification_event = ConversationEvent(  # type: ignore[call-arg]
                source_plugin=self.name,
                session_id=getattr(original_event, "session_id", "default"),
                user_message=clarification_text,
                message_id=getattr(
                    original_event, "message_id", f"clarif_{id(original_event)}"
                ),
                metadata={
                    "requires_clarification": True,
                    "original_intent": intent_info.primary_intent,
                },
            )
            clarification_event.handled_by = self.name  # type: ignore[attr-defined]
            await self._emit_event("conversation", clarification_event)
        except Exception as e:
            logger.exception("Error emitting clarification request: %s", e)

    async def _handle_multi_intent(
        self,
        result: DTAResult,
        original_event: BaseEvent,
        intent_info: IntentDetectionResult,
    ):
        try:
            note = f"Multiple intents detected. Primary: {intent_info.primary_intent}"
            if intent_info.secondary_intents:
                note += f" | Secondary: {', '.join(intent_info.secondary_intents)}"
            multi_event = ConversationEvent(  # type: ignore[call-arg]
                source_plugin=self.name,
                session_id=getattr(original_event, "session_id", "default"),
                user_message=note,
                message_id=getattr(
                    original_event, "message_id", f"multi_{id(original_event)}"
                ),
                metadata={"multi_intent": True},
            )
            await self._emit_event("conversation", multi_event)
            # Continue with primary intent
            await self._emit_tool_call(result, original_event)
        except Exception as e:
            logger.exception("Error handling multi-intent: %s", e)
            await self._emit_error_response(
                original_event, f"Multi-intent processing failed: {e}"
            )

    async def _handle_code_generation(
        self, result: DTAResult, original_event: BaseEvent
    ):
        try:
            parameters = {
                "type": "code_generation",
                "requirements": getattr(result, "function_call", None)
                or "Generate code as requested",
            }
            tool_event = build_tool_call_event(
                source_plugin=self._name,
                tool_name="creator",
                parameters=parameters,
                conversation_id=getattr(original_event, "conversation_id", "default"),
                session_id=getattr(original_event, "session_id", "default"),
                message_id=getattr(original_event, "message_id", None),
            )
            await self._emit_event("tool_call", tool_event)
        except Exception as e:
            logger.exception("Error handling code generation: %s", e)
            await self._emit_error_response(
                original_event, f"Code generation failed: {e}"
            )

    async def _emit_direct_response(self, result: DTAResult, original_event: BaseEvent):
        try:
            intent_info = result.metadata.get("intent_detection", {})
            intent = str(
                getattr(
                    intent_info, "primary_intent", getattr(result, "intent", "") or ""
                )
            )
            if intent == "greeting":
                response_message = (
                    "Hello! I'm your AI assistant. How can I help you today?"
                )
            else:
                response_message = (
                    getattr(result, "function_call", None)
                    or "I'm here to help. What would you like me to do?"
                )
            response_event = ConversationEvent(  # type: ignore[call-arg]
                source_plugin=self.name,
                session_id=getattr(original_event, "session_id", "default"),
                user_message=response_message,
                message_id=getattr(
                    original_event, "message_id", f"response_{id(original_event)}"
                ),
                metadata={
                    "intent": intent,
                    "confidence": getattr(result, "confidence_score", 0.0),
                },
            )
            response_event.handled_by = self.name  # type: ignore[attr-defined]
            await self._emit_event("conversation", response_event)
        except Exception as e:
            logger.exception("Error emitting direct response: %s", e)

    async def _emit_error_response(self, original_event: BaseEvent, error_message: str):
        try:
            error_event = ConversationEvent(  # type: ignore[call-arg]
                source_plugin=self.name,
                session_id=getattr(original_event, "session_id", "default"),
                user_message=f"Sorry, I encountered an error: {error_message}",
                message_id=getattr(
                    original_event, "message_id", f"error_{id(original_event)}"
                ),
                metadata={"error": True, "error_message": error_message},
            )
            error_event.handled_by = self.name  # type: ignore[attr-defined]
            await self._emit_event("conversation", error_event)
        except Exception as e:
            logger.exception("Error emitting error response: %s", e)

    async def get_processing_metrics(
        self, session_id: str | None = None
    ) -> dict[str, Any]:
        if session_id:
            data = self.processing_metrics.get(session_id)
            if data is None:
                return {}
            if isinstance(data, ProcessingMetrics):
                return data.__dict__
            return dict(data)  # type: ignore[return-value]
        return {
            "total_sessions": len(self.processing_metrics),
            "sessions": {
                sid: (pm.__dict__ if isinstance(pm, ProcessingMetrics) else pm)
                for sid, pm in self.processing_metrics.items()
            },
        }

    async def get_health_status(self) -> dict[str, Any]:
        """Get health status of the plugin and its components."""
        status = {
            "plugin_status": "healthy",
            "dta_runtime": "unknown",
            "monitoring": "unknown",
            "cache": "unknown",
            "llm": "unknown",
        }

        try:
            if self.dta_runtime:
                status["dta_runtime"] = "healthy"

            if self.monitoring:
                status["monitoring"] = "healthy"

            if self.cache:
                cache_stats = await self.cache.get_stats()
                status["cache"] = "healthy"
                status["cache_stats"] = cache_stats

            if self.gemini_model:
                status["llm"] = "healthy"
            elif GEMINI_AVAILABLE:
                status["llm"] = "available_not_configured"
            else:
                status["llm"] = "not_available"

        except Exception as e:
            status["plugin_status"] = "unhealthy"
            status["error"] = str(e)

        return status

    def _init_enhanced_protocol(self) -> None:
        """Initialize enhanced protocol engine."""
        try:
            self.enhanced_engine = EnhancedProtocolEngine()
            self.enhanced_protocol_enabled = True
            logger.info("Enhanced Protocol v2.0 initialized successfully")
        except Exception as e:
            logger.exception("Enhanced Protocol initialization failed: %s", e)
            self.enhanced_engine = None
            self.enhanced_protocol_enabled = False

    async def _should_use_enhanced_protocol(self, user_input: str) -> bool:
        if not getattr(self, "enhanced_protocol_enabled", False):
            return False
        triggers = [
            "assess",
            "analyze",
            "evaluate",
            "capabilities",
            "abilities",
            "comprehensive",
            "detailed",
            "security",
            "compliance",
            "complex",
            "thorough",
            "in-depth",
            "what can you do",
        ]
        lower = user_input.lower()
        if any(t in lower for t in triggers):
            return True
        return len(user_input.strip()) > 50

    async def _process_with_enhanced_protocol(self, user_input: str, session_id: str):
        try:
            request = CognitiveRequest(
                user_input=user_input,
                session_id=session_id,
                expansion_tier=ExpansionTier.STANDARD,
                confidence_threshold=self.clarification_threshold,
                compliance_required=True,
            )
            protocol_response = await self.enhanced_engine.process_request(request)  # type: ignore[attr-defined]
            return IntentDetectionResult(
                primary_intent="enhanced_protocol_response",
                confidence=protocol_response.confidence_score,
                requires_clarification=False,
                extracted_parameters={"protocol_response": protocol_response},
            )
        except Exception as e:  # pragma: no cover
            logger.exception("Enhanced protocol processing failed: %s", e)
            return None


# Plugin registration and metadata
__plugin_class__ = PythonicPreprocessorPlugin
__plugin_name__ = "pythonic_preprocessor"
__plugin_version__ = "2.0.0"
__plugin_description__ = (
    "DTA 2.0 Pythonic Preprocessor for intelligent intent detection and routing"
)
