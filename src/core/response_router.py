#!/usr/bin/env python3
"""
Dynamic Response Routing System
Implements context-aware response selection based on conversation state, user intent, and available capabilities.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from core.plugin_communication import (
    MessagePriority,
    PluginCommunicationHub,
    PluginMessage,
)


class IntentType(Enum):
    """Types of user intents"""

    QUESTION = "question"
    COMMAND = "command"
    CREATION = "creation"
    ANALYSIS = "analysis"
    COLLABORATION = "collaboration"
    LEARNING = "learning"
    DEBUGGING = "debugging"
    EXPLORATION = "exploration"
    PLANNING = "planning"
    EXECUTION = "execution"


class ResponseStrategy(Enum):
    """Response routing strategies"""

    SINGLE_BEST = "single_best"
    MULTI_AGENT = "multi_agent"
    COLLABORATIVE = "collaborative"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"


class ConfidenceLevel(Enum):
    """Confidence levels for routing decisions"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ConversationContext:
    """Context information for a conversation"""

    conversation_id: str
    user_id: str = ""
    session_start: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_interaction: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Conversation state
    current_intent: IntentType = IntentType.QUESTION
    previous_intents: list[IntentType] = field(default_factory=list)
    context_tags: set[str] = field(default_factory=set)
    active_topics: list[str] = field(default_factory=list)

    # User preferences and history
    preferred_style: str = "balanced"
    expertise_level: str = "intermediate"
    language: str = "en"
    previous_queries: list[str] = field(default_factory=list)

    # Technical context
    available_tools: set[str] = field(default_factory=set)
    active_plugins: set[str] = field(default_factory=set)
    workspace_context: dict[str, Any] = field(default_factory=dict)

    # Response tracking
    successful_strategies: list[ResponseStrategy] = field(default_factory=list)
    failed_strategies: list[ResponseStrategy] = field(default_factory=list)
    response_times: list[float] = field(default_factory=list)
    satisfaction_scores: list[float] = field(default_factory=list)

    def update_interaction(self) -> None:
        """Update the last interaction timestamp"""
        self.last_interaction = datetime.now(UTC)

    def add_intent(self, intent: IntentType) -> None:
        """Add a new intent to the conversation"""
        if self.current_intent != intent:
            self.previous_intents.append(self.current_intent)
            self.current_intent = intent
        self.update_interaction()

    def add_topic(self, topic: str) -> None:
        """Add a topic to active topics"""
        if topic not in self.active_topics:
            self.active_topics.append(topic)
        if len(self.active_topics) > 5:  # Keep only recent topics
            self.active_topics.pop(0)

    def record_strategy_result(
        self,
        strategy: ResponseStrategy,
        success: bool,
        response_time: float,
        satisfaction: float = 0.5,
    ) -> None:
        """Record the result of a response strategy"""
        if success:
            self.successful_strategies.append(strategy)
        else:
            self.failed_strategies.append(strategy)

        self.response_times.append(response_time)
        self.satisfaction_scores.append(satisfaction)

        # Keep only recent results (last 20)
        if len(self.response_times) > 20:
            self.response_times.pop(0)
            self.satisfaction_scores.pop(0)

        if len(self.successful_strategies) > 10:
            self.successful_strategies.pop(0)

        if len(self.failed_strategies) > 10:
            self.failed_strategies.pop(0)


@dataclass
class RoutingDecision:
    """A routing decision with confidence and reasoning"""

    selected_plugins: list[str]
    strategy: ResponseStrategy
    confidence: ConfidenceLevel
    reasoning: str
    estimated_time: float
    fallback_options: list[tuple[list[str], ResponseStrategy]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityMatch:
    """Match between user intent and plugin capabilities"""

    plugin_name: str
    capability_name: str
    match_score: float
    confidence: ConfidenceLevel
    reasoning: str
    required_inputs: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)


class IntentClassifier:
    """Classifies user intents from natural language input"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Intent patterns (simplified for demo)
        self.intent_patterns = {
            IntentType.QUESTION: [
                r"\b(what|how|why|when|where|who)\b",
                r"\b(explain|describe|tell me)\b",
                r"\?$",
            ],
            IntentType.COMMAND: [
                r"\b(run|execute|start|stop|do)\b",
                r"\b(please|can you)\s+(run|do|execute)\b",
            ],
            IntentType.CREATION: [
                r"\b(create|make|build|generate|write)\b",
                r"\b(new|develop|design)\b",
            ],
            IntentType.ANALYSIS: [
                r"\b(analyze|examine|review|check|inspect)\b",
                r"\b(find|search|look for)\b",
                r"\b(performance|metrics)\b",
            ],
            IntentType.COLLABORATION: [
                r"\b(together|collaborate|work with)\b",
                r"\b(team|group|shared)\b",
                r"\blet's\b",
            ],
            IntentType.DEBUGGING: [
                r"\b(fix|debug|error|problem|issue|bug)\b",
                r"\b(broken|not working|fails|leak)\b",
                r"\bthere's\s+(a|an)\s+(bug|error|problem|issue)\b",
            ],
            IntentType.PLANNING: [
                r"\b(plan|strategy|roadmap|schedule)\b",
                r"\b(organize|structure|outline)\b",
            ],
        }

    def classify_intent(
        self, text: str, context: ConversationContext
    ) -> tuple[IntentType, float]:
        """Classify the intent of user input"""
        text_lower = text.lower()
        intent_scores = {}

        # Score each intent type based on patterns
        for intent_type, patterns in self.intent_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches * 0.2

            # Boost score based on previous intents
            if intent_type in context.previous_intents[-3:]:
                score += 0.1

            intent_scores[intent_type] = min(score, 1.0)

        # Find best match
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            best_score = intent_scores[best_intent]
        else:
            best_intent = IntentType.QUESTION
            best_score = 0.5

        self.logger.debug(
            f"Classified intent: {best_intent.value} (score: {best_score:.2f})"
        )
        return best_intent, best_score


class CapabilityMatcher:
    """Matches user intents to available plugin capabilities"""

    def __init__(self, comm_hub: PluginCommunicationHub):
        self.comm_hub = comm_hub
        self.logger = logging.getLogger(__name__)

        # Capability keywords for matching
        self.capability_keywords = {
            "code_generation": [
                "code",
                "programming",
                "script",
                "function",
                "class",
                "create",
                "generate",
            ],
            "analysis": [
                "analyze",
                "examine",
                "review",
                "inspect",
                "study",
                "check",
                "performance",
            ],
            "creation": ["create", "make", "build", "generate", "produce", "develop"],
            "debugging": ["debug", "fix", "error", "problem", "issue", "bug", "broken"],
            "planning": ["plan", "strategy", "organize", "structure", "roadmap"],
            "memory": ["remember", "recall", "store", "history", "past"],
            "search": ["find", "search", "look", "discover", "locate"],
            "collaboration": [
                "team",
                "group",
                "together",
                "shared",
                "collaborate",
                "work",
            ],
            "refactoring": ["refactor", "improve", "optimize", "clean", "restructure"],
            "metrics": ["metrics", "measurement", "statistics", "data", "performance"],
            "repair": ["repair", "fix", "correct", "resolve", "solve"],
            "organization": ["organize", "schedule", "plan", "structure", "manage"],
        }

    def find_capability_matches(
        self, text: str, intent: IntentType, context: ConversationContext
    ) -> list[CapabilityMatch]:
        """Find plugins and capabilities that match the user intent"""
        text_lower = text.lower()
        matches = []

        # Get available plugins and capabilities
        plugin_stats = self.comm_hub.get_plugin_stats()

        for plugin_name, stats in plugin_stats.items():
            plugin_capabilities = stats.get("capabilities", [])

            for capability in plugin_capabilities:
                match_score = self._calculate_match_score(
                    text_lower, intent, capability, context
                )

                if match_score > 0.2:  # Lower threshold for relevance
                    confidence = self._determine_confidence(match_score)
                    reasoning = self._generate_reasoning(
                        capability, intent, match_score
                    )

                    match = CapabilityMatch(
                        plugin_name=plugin_name,
                        capability_name=capability,
                        match_score=match_score,
                        confidence=confidence,
                        reasoning=reasoning,
                    )
                    matches.append(match)

        # Sort by match score
        matches.sort(key=lambda x: x.match_score, reverse=True)

        self.logger.debug(f"Found {len(matches)} capability matches")
        return matches

    def _calculate_match_score(
        self,
        text: str,
        intent: IntentType,
        capability: str,
        context: ConversationContext,
    ) -> float:
        """Calculate how well a capability matches the user input"""
        score = 0.0

        # Direct keyword matching
        capability_lower = capability.lower()
        if capability_lower in self.capability_keywords:
            keywords = self.capability_keywords[capability_lower]
            for keyword in keywords:
                if keyword in text:
                    score += 0.3  # More generous scoring

        # Partial capability name matching
        for word in capability_lower.split("_"):
            if word in text:
                score += 0.2

        # Intent-based matching
        intent_boost = self._get_intent_capability_boost(intent, capability)
        score += intent_boost

        # Context-based matching
        if capability in context.context_tags:
            score += 0.2

        # Historical success
        if context.available_tools and capability in context.available_tools:
            score += 0.1

        # General fallback - if no specific matches, give small base score
        if score == 0.0:
            score = 0.1  # Small base score to ensure some matches

        return min(score, 1.0)

    def _get_intent_capability_boost(
        self, intent: IntentType, capability: str
    ) -> float:
        """Get boost score based on intent-capability alignment"""
        capability_lower = capability.lower()

        intent_capability_map = {
            IntentType.CREATION: ["creation", "generation", "build", "make"],
            IntentType.ANALYSIS: ["analysis", "review", "examine", "inspect"],
            IntentType.DEBUGGING: ["debug", "fix", "error", "problem"],
            IntentType.PLANNING: ["plan", "strategy", "organize"],
            IntentType.COLLABORATION: ["team", "collaborate", "shared"],
            IntentType.LEARNING: ["learn", "study", "understand", "education"],
        }

        if intent in intent_capability_map:
            for keyword in intent_capability_map[intent]:
                if keyword in capability_lower:
                    return 0.3

        return 0.0

    def _determine_confidence(self, match_score: float) -> ConfidenceLevel:
        """Determine confidence level based on match score"""
        if match_score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif match_score >= 0.6:
            return ConfidenceLevel.HIGH
        elif match_score >= 0.4:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _generate_reasoning(
        self, capability: str, intent: IntentType, score: float
    ) -> str:
        """Generate human-readable reasoning for the match"""
        return f"Capability '{capability}' matches intent '{intent.value}' with score {score:.2f}"


class ResponseRouter:
    """Main response routing system"""

    def __init__(self, comm_hub: PluginCommunicationHub):
        self.comm_hub = comm_hub
        self.logger = logging.getLogger(__name__)

        # Components
        self.intent_classifier = IntentClassifier()
        self.capability_matcher = CapabilityMatcher(comm_hub)

        # Context management
        self.active_contexts: dict[str, ConversationContext] = {}

        # Routing strategies
        self.strategy_handlers = {
            ResponseStrategy.SINGLE_BEST: self._handle_single_best,
            ResponseStrategy.MULTI_AGENT: self._handle_multi_agent,
            ResponseStrategy.COLLABORATIVE: self._handle_collaborative,
            ResponseStrategy.SEQUENTIAL: self._handle_sequential,
            ResponseStrategy.PARALLEL: self._handle_parallel,
        }

    def get_or_create_context(
        self, conversation_id: str, user_id: str = ""
    ) -> ConversationContext:
        """Get existing context or create new one"""
        if conversation_id not in self.active_contexts:
            self.active_contexts[conversation_id] = ConversationContext(
                conversation_id=conversation_id, user_id=user_id
            )

        context = self.active_contexts[conversation_id]
        context.update_interaction()
        return context

    async def route_request(
        self, user_input: str, conversation_id: str, user_id: str = ""
    ) -> RoutingDecision:
        """Route a user request to appropriate plugins"""
        start_time = datetime.now(UTC)

        # Get conversation context
        context = self.get_or_create_context(conversation_id, user_id)

        # Classify intent
        intent, intent_confidence = self.intent_classifier.classify_intent(
            user_input, context
        )
        context.add_intent(intent)

        # Find capability matches
        capability_matches = self.capability_matcher.find_capability_matches(
            user_input, intent, context
        )

        # Determine optimal strategy
        strategy = self._select_strategy(intent, capability_matches, context)

        # Create routing decision
        decision = self._create_routing_decision(
            user_input, intent, capability_matches, strategy, context
        )

        # Log decision
        processing_time = (datetime.now(UTC) - start_time).total_seconds()
        self.logger.info(
            f"Routing decision for {conversation_id}: {strategy.value} -> {decision.selected_plugins} (confidence: {decision.confidence.value}, time: {processing_time:.3f}s)"
        )

        return decision

    def _select_strategy(
        self,
        intent: IntentType,
        matches: list[CapabilityMatch],
        context: ConversationContext,
    ) -> ResponseStrategy:
        """Select the optimal response strategy"""
        # Simple strategy selection logic
        if not matches:
            return ResponseStrategy.SINGLE_BEST

        # High-confidence single match
        if len(matches) == 1 and matches[0].confidence in [
            ConfidenceLevel.HIGH,
            ConfidenceLevel.VERY_HIGH,
        ]:
            return ResponseStrategy.SINGLE_BEST

        # Multiple good matches - consider collaboration
        high_confidence_matches = [
            m
            for m in matches
            if m.confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]
        ]

        if len(high_confidence_matches) > 1:
            # Check if intent suggests collaboration
            if intent in [
                IntentType.COLLABORATION,
                IntentType.ANALYSIS,
                IntentType.PLANNING,
            ]:
                return ResponseStrategy.COLLABORATIVE
            else:
                return ResponseStrategy.PARALLEL

        # Multiple medium matches - try sequential
        if len(matches) > 2:
            return ResponseStrategy.SEQUENTIAL

        # Default to multi-agent
        return ResponseStrategy.MULTI_AGENT

    def _create_routing_decision(
        self,
        user_input: str,
        intent: IntentType,
        matches: list[CapabilityMatch],
        strategy: ResponseStrategy,
        context: ConversationContext,
    ) -> RoutingDecision:
        """Create a routing decision based on analysis"""

        # Select plugins based on strategy
        if strategy == ResponseStrategy.SINGLE_BEST:
            selected_plugins = [matches[0].plugin_name] if matches else []
            confidence = matches[0].confidence if matches else ConfidenceLevel.LOW
        elif strategy in [
            ResponseStrategy.MULTI_AGENT,
            ResponseStrategy.COLLABORATIVE,
            ResponseStrategy.PARALLEL,
        ]:
            # Take top 3 matches
            selected_plugins = [m.plugin_name for m in matches[:3]]
            confidence = (
                ConfidenceLevel.HIGH if len(matches) >= 2 else ConfidenceLevel.MEDIUM
            )
        elif strategy == ResponseStrategy.SEQUENTIAL:
            # Take top 2-4 matches for sequential processing
            selected_plugins = [m.plugin_name for m in matches[:4]]
            confidence = ConfidenceLevel.MEDIUM
        else:
            selected_plugins = [m.plugin_name for m in matches[:2]]
            confidence = ConfidenceLevel.MEDIUM

        # Remove duplicates while preserving order
        seen = set()
        unique_plugins = []
        for plugin in selected_plugins:
            if plugin not in seen:
                seen.add(plugin)
                unique_plugins.append(plugin)
        selected_plugins = unique_plugins

        # Generate reasoning
        reasoning = f"Selected {strategy.value} strategy for {intent.value} intent with {len(matches)} capability matches"

        # Estimate processing time
        estimated_time = self._estimate_processing_time(strategy, len(selected_plugins))

        # Create fallback options
        fallback_options = self._generate_fallback_options(matches, strategy)

        return RoutingDecision(
            selected_plugins=selected_plugins,
            strategy=strategy,
            confidence=confidence,
            reasoning=reasoning,
            estimated_time=estimated_time,
            fallback_options=fallback_options,
            metadata={
                "intent": intent.value,
                "num_matches": len(matches),
                "user_input_length": len(user_input),
            },
        )

    def _estimate_processing_time(
        self, strategy: ResponseStrategy, num_plugins: int
    ) -> float:
        """Estimate processing time based on strategy and plugin count"""
        base_time = 1.0  # Base processing time in seconds

        strategy_multipliers = {
            ResponseStrategy.SINGLE_BEST: 1.0,
            ResponseStrategy.MULTI_AGENT: 1.5,
            ResponseStrategy.COLLABORATIVE: 2.0,
            ResponseStrategy.SEQUENTIAL: num_plugins * 0.8,
            ResponseStrategy.PARALLEL: 1.2,
            ResponseStrategy.HIERARCHICAL: 2.5,
        }

        multiplier = strategy_multipliers.get(strategy, 1.0)
        return base_time * multiplier

    def _generate_fallback_options(
        self, matches: list[CapabilityMatch], primary_strategy: ResponseStrategy
    ) -> list[tuple[list[str], ResponseStrategy]]:
        """Generate fallback routing options"""
        fallbacks = []

        if len(matches) > 1:
            # Fallback to single best
            if primary_strategy != ResponseStrategy.SINGLE_BEST:
                fallbacks.append(
                    ([matches[0].plugin_name], ResponseStrategy.SINGLE_BEST)
                )

            # Fallback to multi-agent if not already selected
            if primary_strategy != ResponseStrategy.MULTI_AGENT and len(matches) >= 2:
                fallbacks.append(
                    ([m.plugin_name for m in matches[:2]], ResponseStrategy.MULTI_AGENT)
                )

        return fallbacks[:2]  # Limit to 2 fallback options

    async def execute_routing_decision(
        self, decision: RoutingDecision, user_input: str, conversation_id: str
    ) -> dict[str, Any]:
        """Execute the routing decision"""
        strategy_handler = self.strategy_handlers.get(decision.strategy)

        if not strategy_handler:
            self.logger.error(f"No handler for strategy: {decision.strategy}")
            return {"error": f"Unsupported strategy: {decision.strategy}"}

        try:
            result = await strategy_handler(decision, user_input, conversation_id)

            # Record success
            context = self.active_contexts.get(conversation_id)
            if context:
                context.record_strategy_result(
                    decision.strategy, True, decision.estimated_time
                )

            return result

        except Exception as e:
            self.logger.error(f"Error executing strategy {decision.strategy}: {e}")

            # Record failure
            context = self.active_contexts.get(conversation_id)
            if context:
                context.record_strategy_result(
                    decision.strategy, False, decision.estimated_time
                )

            # Try fallback if available
            if decision.fallback_options:
                self.logger.info(f"Trying fallback strategy for {conversation_id}")

                try:
                    fallback_plugins, fallback_strategy = decision.fallback_options[0]

                    fallback_decision = RoutingDecision(
                        selected_plugins=fallback_plugins,
                        strategy=fallback_strategy,
                        confidence=ConfidenceLevel.MEDIUM,
                        reasoning="Fallback after primary strategy failed",
                        estimated_time=decision.estimated_time * 0.8,
                        fallback_options=[],
                    )

                    fallback_handler = self.strategy_handlers.get(fallback_strategy)
                    if fallback_handler:
                        fallback_result = await fallback_handler(
                            fallback_decision, user_input, conversation_id
                        )

                        # Record fallback success
                        if context:
                            context.record_strategy_result(
                                fallback_strategy,
                                True,
                                fallback_decision.estimated_time,
                            )

                        return fallback_result
                    else:
                        return {
                            "error": f"Fallback strategy handler not found: {fallback_strategy}"
                        }

                except Exception as fallback_error:
                    self.logger.error(
                        f"Fallback strategy also failed: {fallback_error}"
                    )
                    return {
                        "error": f"Both primary and fallback strategies failed: {e}, {fallback_error}"
                    }

            return {"error": f"Strategy execution failed: {e}"}

    async def _handle_single_best(
        self, decision: RoutingDecision, user_input: str, conversation_id: str
    ) -> dict[str, Any]:
        """Handle single best plugin strategy"""
        if not decision.selected_plugins:
            return {"error": "No plugins selected"}

        plugin_name = decision.selected_plugins[0]

        message = PluginMessage(
            sender="response_router",
            recipients=[plugin_name],
            message_type="user_request",
            payload={
                "user_input": user_input,
                "conversation_id": conversation_id,
                "strategy": "single_best",
            },
            priority=MessagePriority.HIGH,
        )

        success = await self.comm_hub.send_message(message)

        return {
            "strategy": "single_best",
            "selected_plugin": plugin_name,
            "message_sent": success,
            "estimated_time": decision.estimated_time,
        }

    async def _handle_multi_agent(
        self, decision: RoutingDecision, user_input: str, conversation_id: str
    ) -> dict[str, Any]:
        """Handle multi-agent strategy"""
        if not decision.selected_plugins:
            return {"error": "No plugins selected"}

        results = []

        for plugin_name in decision.selected_plugins:
            message = PluginMessage(
                sender="response_router",
                recipients=[plugin_name],
                message_type="user_request",
                payload={
                    "user_input": user_input,
                    "conversation_id": conversation_id,
                    "strategy": "multi_agent",
                    "peer_plugins": [
                        p for p in decision.selected_plugins if p != plugin_name
                    ],
                },
                priority=MessagePriority.HIGH,
            )

            success = await self.comm_hub.send_message(message)
            results.append({"plugin": plugin_name, "message_sent": success})

        return {
            "strategy": "multi_agent",
            "selected_plugins": decision.selected_plugins,
            "results": results,
            "estimated_time": decision.estimated_time,
        }

    async def _handle_collaborative(
        self, decision: RoutingDecision, user_input: str, conversation_id: str
    ) -> dict[str, Any]:
        """Handle collaborative strategy"""
        if len(decision.selected_plugins) < 2:
            return {"error": "Collaborative strategy requires at least 2 plugins"}

        # Send coordination message to all plugins
        coordination_message = PluginMessage(
            sender="response_router",
            recipients=decision.selected_plugins,
            message_type="collaboration_request",
            payload={
                "user_input": user_input,
                "conversation_id": conversation_id,
                "strategy": "collaborative",
                "collaborators": decision.selected_plugins,
                "coordination_needed": True,
            },
            priority=MessagePriority.HIGH,
        )

        success = await self.comm_hub.send_message(coordination_message)

        return {
            "strategy": "collaborative",
            "selected_plugins": decision.selected_plugins,
            "coordination_sent": success,
            "estimated_time": decision.estimated_time,
        }

    async def _handle_sequential(
        self, decision: RoutingDecision, user_input: str, conversation_id: str
    ) -> dict[str, Any]:
        """Handle sequential strategy"""
        if not decision.selected_plugins:
            return {"error": "No plugins selected"}

        # Send to first plugin with sequence information
        first_plugin = decision.selected_plugins[0]

        message = PluginMessage(
            sender="response_router",
            recipients=[first_plugin],
            message_type="sequential_request",
            payload={
                "user_input": user_input,
                "conversation_id": conversation_id,
                "strategy": "sequential",
                "sequence": decision.selected_plugins,
                "current_position": 0,
                "next_plugins": decision.selected_plugins[1:],
            },
            priority=MessagePriority.HIGH,
        )

        success = await self.comm_hub.send_message(message)

        return {
            "strategy": "sequential",
            "sequence": decision.selected_plugins,
            "started_with": first_plugin,
            "message_sent": success,
            "estimated_time": decision.estimated_time,
        }

    async def _handle_parallel(
        self, decision: RoutingDecision, user_input: str, conversation_id: str
    ) -> dict[str, Any]:
        """Handle parallel strategy"""
        if not decision.selected_plugins:
            return {"error": "No plugins selected"}

        # Send parallel execution requests
        correlation_id = f"parallel_{conversation_id}_{datetime.now(UTC).timestamp()}"
        results = []

        for plugin_name in decision.selected_plugins:
            message = PluginMessage(
                sender="response_router",
                recipients=[plugin_name],
                message_type="parallel_request",
                payload={
                    "user_input": user_input,
                    "conversation_id": conversation_id,
                    "strategy": "parallel",
                    "parallel_group": decision.selected_plugins,
                    "correlation_id": correlation_id,
                },
                correlation_id=correlation_id,
                priority=MessagePriority.HIGH,
            )

            success = await self.comm_hub.send_message(message)
            results.append({"plugin": plugin_name, "message_sent": success})

        return {
            "strategy": "parallel",
            "selected_plugins": decision.selected_plugins,
            "correlation_id": correlation_id,
            "results": results,
            "estimated_time": decision.estimated_time,
        }

    def get_routing_stats(self) -> dict[str, Any]:
        """Get routing system statistics"""
        total_contexts = len(self.active_contexts)

        # Aggregate statistics from all contexts
        total_interactions = 0
        strategy_counts = {}
        avg_satisfaction = 0.0
        avg_response_time = 0.0

        for context in self.active_contexts.values():
            total_interactions += len(context.previous_intents) + 1

            for strategy in context.successful_strategies:
                strategy_counts[strategy.value] = (
                    strategy_counts.get(strategy.value, 0) + 1
                )

            if context.satisfaction_scores:
                avg_satisfaction += sum(context.satisfaction_scores) / len(
                    context.satisfaction_scores
                )

            if context.response_times:
                avg_response_time += sum(context.response_times) / len(
                    context.response_times
                )

        if total_contexts > 0:
            avg_satisfaction /= total_contexts
            avg_response_time /= total_contexts

        return {
            "active_conversations": total_contexts,
            "total_interactions": total_interactions,
            "strategy_usage": strategy_counts,
            "average_satisfaction": round(avg_satisfaction, 2),
            "average_response_time": round(avg_response_time, 3),
            "active_plugins": len(self.comm_hub.registered_plugins),
        }
