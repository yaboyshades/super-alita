"""
Co-Architect Mode for Super Alita - Enhanced Copilot Agent Integration
=====================================================================

AGENT DEV MODE (Copilot read this):
- Event-driven agent mode with Redis pub/sub integration
- Neural Atoms for agent reasoning and memory
- Plugin architecture for modular agent capabilities
- Conversation history summarization and context awareness
- Use logging.getLogger(__name__), never print. Clamp sizes/ranges
- Write tests: fixed inputs ⇒ fixed outputs; handler validation
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from src.core.events import BaseEvent
from src.core.neural_atom import NeuralAtom, NeuralAtomMetadata
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)

# Constants
MAX_CONVERSATION_HISTORY = 100
MAX_MESSAGE_LENGTH = 1000
SUMMARY_CHUNK_SIZE = 50
ANALYSIS_CONFIDENCE_THRESHOLD = 0.7
RETRY_COUNT = 3


@dataclass
class ConversationSummary:
    """Structured conversation summary for context awareness."""

    timestamp: float
    session_id: str
    user_intent: str
    technical_context: str
    progress_status: str
    key_decisions: list[str]
    next_actions: list[str]


@dataclass
class AgentModeEvent(BaseEvent):
    """Event for agent mode operations."""

    event_type: str = "agent_mode"
    operation: str = ""
    parameters: dict[str, Any] | None = None
    conversation_summary: ConversationSummary | None = None

    def __post_init__(self) -> None:
        if self.parameters is None:
            self.parameters = {}


class CopilotAgentAtom(NeuralAtom):
    """Neural Atom for Copilot agent reasoning and conversation analysis."""

    def __init__(self, metadata: NeuralAtomMetadata) -> None:
        super().__init__(metadata)
        self.conversation_history: list[dict[str, Any]] = []
        self.session_context: dict[str, Any] = {}

    async def execute(self, input_data: Any = None) -> Any:
        """Execute agent reasoning with conversation analysis."""
        if input_data is None:
            input_data = {}

        operation = input_data.get("operation", "analyze")

        if operation == "analyze":
            return await self._analyze_conversation(input_data)
        if operation == "summarize":
            return await self._summarize_conversation(input_data)
        if operation == "extract_context":
            return await self._extract_technical_context(input_data)
        return {"error": f"Unknown operation: {operation}"}

    def get_embedding(self) -> list[float]:
        """Get semantic embedding for agent reasoning."""
        # Simplified embedding based on conversation patterns
        return [0.1] * 384

    def can_handle(self, task_description: str) -> float:
        """Determine if this atom can handle the task."""
        agent_keywords = ["analyze", "summarize", "conversation", "context", "copilot"]

        task_lower = task_description.lower()
        matches = sum(1 for keyword in agent_keywords if keyword in task_lower)

        return min(matches / len(agent_keywords), 1.0)

    async def _analyze_conversation(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze conversation for patterns and insights."""
        messages = data.get("messages", [])

        if not messages:
            return {"analysis": "No messages to analyze", "confidence": 0.0}

        # Extract patterns
        user_intents = []
        technical_topics = []

        for msg in messages[-10:]:  # Analyze last 10 messages
            content = msg.get("content", "")[:MAX_MESSAGE_LENGTH]

            # Simple intent extraction
            if any(
                word in content.lower() for word in ["create", "implement", "build"]
            ):
                user_intents.append("creation")
            elif any(word in content.lower() for word in ["fix", "debug", "error"]):
                user_intents.append("debugging")
            elif any(
                word in content.lower() for word in ["optimize", "improve", "enhance"]
            ):
                user_intents.append("optimization")

            # Technical topic extraction
            if "docker" in content.lower():
                technical_topics.append("containerization")
            elif "redis" in content.lower():
                technical_topics.append("data_storage")
            elif "python" in content.lower():
                technical_topics.append("python_development")

        return {
            "analysis": {
                "dominant_intent": max(set(user_intents), key=user_intents.count)
                if user_intents
                else "unknown",
                "technical_focus": list(set(technical_topics)),
                "message_count": len(messages),
                "conversation_depth": "deep"
                if len(messages) > 20
                else "moderate"
                if len(messages) > 5
                else "shallow",
            },
            "confidence": min(len(messages) / 10.0, 1.0),
        }

    async def _summarize_conversation(self, data: dict[str, Any]) -> dict[str, Any]:
        """Summarize conversation history for context continuity."""
        messages = data.get("messages", [])
        session_id = data.get("session_id", "unknown")

        if not messages:
            return {"summary": None, "error": "No messages to summarize"}

        # Process messages in chunks
        recent_messages = messages[-SUMMARY_CHUNK_SIZE:]

        # Extract key information
        user_requests = []
        system_responses = []
        technical_context = []

        for msg in recent_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")[:MAX_MESSAGE_LENGTH]

            if role == "user":
                user_requests.append(content)
            elif role == "assistant":
                system_responses.append(content)

            # Extract technical context
            for tech_term in [
                "docker",
                "redis",
                "python",
                "copilot",
                "agent",
                "neural",
            ]:
                if tech_term in content.lower():
                    technical_context.append(tech_term)

        # Create structured summary
        summary = ConversationSummary(
            timestamp=time.time(),
            session_id=session_id,
            user_intent=self._extract_primary_intent(user_requests),
            technical_context=", ".join(set(technical_context)),
            progress_status=self._assess_progress_status(messages),
            key_decisions=self._extract_key_decisions(system_responses),
            next_actions=self._extract_next_actions(messages),
        )

        return {"summary": summary, "confidence": ANALYSIS_CONFIDENCE_THRESHOLD}

    async def _extract_technical_context(self, data: dict[str, Any]) -> dict[str, Any]:
        """Extract technical context from conversation."""
        messages = data.get("messages", [])

        context = {
            "technologies": set(),
            "file_patterns": set(),
            "commands_mentioned": set(),
            "error_patterns": set(),
        }

        for msg in messages[-20:]:  # Look at recent messages
            content = msg.get("content", "").lower()

            # Technology detection
            tech_patterns = [
                "docker",
                "redis",
                "python",
                "nodejs",
                "react",
                "vue",
                "angular",
            ]
            for tech in tech_patterns:
                if tech in content:
                    context["technologies"].add(tech)

            # File pattern detection
            if ".py" in content:
                context["file_patterns"].add("python")
            if ".js" in content or ".ts" in content:
                context["file_patterns"].add("javascript")
            if ".yml" in content or ".yaml" in content:
                context["file_patterns"].add("yaml")

            # Command detection
            if "docker-compose" in content:
                context["commands_mentioned"].add("docker-compose")
            if "pytest" in content:
                context["commands_mentioned"].add("pytest")
            if "npm" in content:
                context["commands_mentioned"].add("npm")

        # Convert sets to lists for JSON serialization
        return {
            "context": {
                "technologies": list(context["technologies"]),
                "file_patterns": list(context["file_patterns"]),
                "commands_mentioned": list(context["commands_mentioned"]),
                "error_patterns": list(context["error_patterns"]),
            },
            "confidence": 0.8,
        }

    def _extract_primary_intent(self, user_requests: list[str]) -> str:
        """Extract primary user intent from requests."""
        if not user_requests:
            return "unknown"

        combined_text = " ".join(user_requests).lower()

        intents = {
            "setup": ["setup", "install", "configure", "initialize"],
            "development": ["create", "implement", "build", "develop"],
            "debugging": ["fix", "debug", "error", "issue", "problem"],
            "optimization": ["optimize", "improve", "enhance", "performance"],
            "testing": ["test", "validate", "check", "verify"],
        }

        scores = {}
        for intent, keywords in intents.items():
            scores[intent] = sum(1 for keyword in keywords if keyword in combined_text)

        return (
            max(scores.items(), key=lambda x: x[1])[0]
            if any(scores.values())
            else "general"
        )

    def _assess_progress_status(self, messages: list[dict[str, Any]]) -> str:
        """Assess overall progress status from conversation."""
        if len(messages) < 5:
            return "starting"

        recent_content = " ".join(
            [msg.get("content", "")[:MAX_MESSAGE_LENGTH] for msg in messages[-5:]]
        ).lower()

        if any(
            word in recent_content
            for word in ["completed", "finished", "done", "success"]
        ):
            return "completed"
        if any(
            word in recent_content for word in ["working", "progress", "implementing"]
        ):
            return "in_progress"
        if any(
            word in recent_content for word in ["error", "failed", "issue", "problem"]
        ):
            return "blocked"
        return "ongoing"

    def _extract_key_decisions(self, system_responses: list[str]) -> list[str]:
        """Extract key decisions from system responses."""
        decisions = []

        for response in system_responses[-RETRY_COUNT:]:  # Last 3 responses
            content = response[:MAX_MESSAGE_LENGTH].lower()

            # Look for decision patterns
            decision_indicators = [
                "i will",
                "let's use",
                "we should",
                "the approach is",
                "i'll implement",
                "the solution is",
                "we'll proceed with",
            ]

            for indicator in decision_indicators:
                if indicator in content:
                    # Extract the sentence containing the decision
                    sentences = content.split(".")
                    for sentence in sentences:
                        if indicator in sentence:
                            decisions.append(sentence.strip()[:100])  # Limit length
                            break

        return decisions[:RETRY_COUNT]  # Limit to 3 key decisions

    def _extract_next_actions(self, messages: list[dict[str, Any]]) -> list[str]:
        """Extract next actions from conversation."""
        actions = []

        # Look at last few messages for action items
        for msg in messages[-RETRY_COUNT:]:
            content = msg.get("content", "")[:MAX_MESSAGE_LENGTH].lower()

            action_patterns = [
                "next step",
                "we need to",
                "let's",
                "should",
                "will",
                "todo",
                "action item",
                "follow up",
            ]

            for pattern in action_patterns:
                if pattern in content:
                    # Extract action context
                    sentences = content.split(".")
                    for sentence in sentences:
                        if pattern in sentence and len(sentence.strip()) > 10:
                            actions.append(sentence.strip()[:100])
                            break

        return actions[:RETRY_COUNT]  # Limit to 3 next actions


class CopilotAgentPlugin(PluginInterface):
    """Plugin for Copilot agent mode with conversation summarization."""

    def __init__(self) -> None:
        super().__init__()
        self.agent_atom: CopilotAgentAtom | None = None
        self.conversation_cache: dict[str, list[dict[str, Any]]] = {}

    @property
    def name(self) -> str:
        return "copilot_agent"

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        """Setup the Copilot agent plugin."""
        await super().setup(event_bus, store, config)

        # Create agent atom
        metadata = NeuralAtomMetadata(
            name="copilot_agent_atom",
            description="Neural Atom for Copilot agent reasoning and conversation analysis",
            capabilities=[
                "conversation_analysis",
                "context_extraction",
                "summarization",
            ],
        )

        self.agent_atom = CopilotAgentAtom(metadata)

        # Register with store if available
        if hasattr(store, "register"):
            await store.register(self.agent_atom)

        logger.info("Copilot agent plugin setup completed")

    async def start(self) -> None:
        """Start the Copilot agent plugin."""
        await super().start()

        # Subscribe to agent mode events
        if self.event_bus:
            await self.subscribe("agent_mode", self._handle_agent_mode_event)
            await self.subscribe(
                "conversation_update", self._handle_conversation_update
            )

        logger.info("Copilot agent plugin started")

    async def shutdown(self) -> None:
        """Shutdown the Copilot agent plugin."""
        logger.info("Copilot agent plugin shutting down")
        await super().shutdown()

    async def _handle_agent_mode_event(self, event: AgentModeEvent) -> None:
        """Handle agent mode events."""
        try:
            operation = event.operation
            parameters = event.parameters or {}

            if not self.agent_atom:
                logger.error("Agent atom not initialized")
                return

            # Execute operation
            result = await self.agent_atom.execute(
                {"operation": operation, **parameters}
            )

            # Emit result event
            if self.event_bus:
                await self.emit_event(
                    "agent_mode_result",
                    {
                        "operation": operation,
                        "result": result,
                        "session_id": getattr(event, "session_id", "unknown"),
                    },
                )

            logger.info(f"Agent mode operation '{operation}' completed successfully")

        except Exception as e:
            logger.error(f"Error handling agent mode event: {e}")

            if self.event_bus:
                await self.emit_event(
                    "agent_mode_error",
                    {
                        "operation": event.operation,
                        "error": str(e),
                        "session_id": getattr(event, "session_id", "unknown"),
                    },
                )

    async def _handle_conversation_update(self, event: BaseEvent) -> None:
        """Handle conversation updates for context awareness."""
        try:
            session_id = getattr(event, "session_id", "default")
            message = getattr(event, "message", {})

            # Update conversation cache
            if session_id not in self.conversation_cache:
                self.conversation_cache[session_id] = []

            self.conversation_cache[session_id].append(message)

            # Maintain cache size
            if len(self.conversation_cache[session_id]) > MAX_CONVERSATION_HISTORY:
                self.conversation_cache[session_id] = self.conversation_cache[
                    session_id
                ][-MAX_CONVERSATION_HISTORY:]

            # Generate summary if needed
            if len(self.conversation_cache[session_id]) % 10 == 0:  # Every 10 messages
                await self._generate_conversation_summary(session_id)

        except Exception as e:
            logger.error(f"Error handling conversation update: {e}")

    async def _generate_conversation_summary(self, session_id: str) -> None:
        """Generate conversation summary for context continuity."""
        try:
            if not self.agent_atom or session_id not in self.conversation_cache:
                return

            messages = self.conversation_cache[session_id]

            result = await self.agent_atom.execute(
                {
                    "operation": "summarize",
                    "messages": messages,
                    "session_id": session_id,
                }
            )

            summary = result.get("summary")
            if summary and self.event_bus:
                await self.emit_event(
                    "conversation_summary",
                    {
                        "session_id": session_id,
                        "summary": summary,
                        "timestamp": time.time(),
                    },
                )

            logger.info(f"Generated conversation summary for session {session_id}")

        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")


# Agent mode utility functions
async def enable_agent_mode(event_bus: Any, session_id: str = "default") -> bool:
    """Enable agent mode for enhanced Copilot interaction."""
    try:
        event = AgentModeEvent(
            source_plugin="copilot_agent",
            operation="enable",
            parameters={"session_id": session_id},
        )

        if event_bus:
            await event_bus.publish(event)

        logger.info(f"Agent mode enabled for session {session_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to enable agent mode: {e}")
        return False


async def summarize_conversation_history(
    event_bus: Any, messages: list[dict[str, Any]], session_id: str = "default"
) -> ConversationSummary | None:
    """Summarize conversation history for context continuity."""
    try:
        event = AgentModeEvent(
            source_plugin="copilot_agent",
            operation="summarize",
            parameters={"messages": messages, "session_id": session_id},
        )

        if event_bus:
            await event_bus.publish(event)

        # Note: In a real implementation, you'd wait for the result event
        # For now, return a placeholder
        return ConversationSummary(
            timestamp=time.time(),
            session_id=session_id,
            user_intent="context_continuity",
            technical_context="conversation_summarization",
            progress_status="in_progress",
            key_decisions=["Implement conversation summarization"],
            next_actions=["Wait for summary result"],
        )

    except Exception as e:
        logger.error(f"Failed to summarize conversation: {e}")
        return None


# Export key components
__all__ = [
    "AgentModeEvent",
    "ConversationSummary",
    "CopilotAgentAtom",
    "CopilotAgentPlugin",
    "enable_agent_mode",
    "summarize_conversation_history",
]
