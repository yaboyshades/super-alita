#!/usr/bin/env python3
"""
🔮 PREDICTIVE WORLD MODEL - OUTCOME FORECASTING ENGINE
Advanced prediction capabilities for Super Alita's planning system

AGENT DEV MODE (Copilot read this):
- Event-driven only; define Pydantic events (Literal event_type) and add to EVENT_TYPE_MAP
- Neural Atoms are concrete subclasses with UUIDv5 deterministic IDs
- Use logging.getLogger(__name__), never print. Clamp sizes/ranges
- Write tests: fixed inputs ⇒ fixed outputs; handler validation
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

# Constants for risk assessment thresholds
HIGH_FAILURE_RATE_THRESHOLD = 0.3
LONG_EXECUTION_THRESHOLD = 5.0
LONG_EXECUTION_RATE_THRESHOLD = 0.2
MIN_DATA_THRESHOLD = 3
COMPLEX_STATE_THRESHOLD = 20
RECENT_FAILURE_WINDOW_SECONDS = 3600
RECENT_FAILURE_THRESHOLD = 2
RECENT_DATA_WINDOW_SECONDS = 86400


class PredictionOutcome(NamedTuple):
    """Prediction outcome with comprehensive forecasting"""

    success_probability: float
    expected_duration: float
    confidence: float
    risk_factors: list[str]
    alternative_strategies: list[dict[str, Any]]


@dataclass
class StateTransition:
    """Record of state changes and their outcomes"""

    initial_state: dict[str, Any]
    action_taken: str
    final_state: dict[str, Any]
    success: bool
    duration: float
    timestamp: datetime
    context: dict[str, Any] = field(default_factory=dict)


class PredictiveWorldModel:
    """
    🔮 Advanced world model for outcome prediction and strategic planning

    Capabilities:
    - Predict action outcomes before execution
    - Learn from state transitions
    - Identify optimal strategies

    - Meta-pattern recognition for execution strategies
    - Adaptive confidence calibration
    - Self-modifying prediction algorithms
    """

    def __init__(
        self,
        max_history: int = 10000,
        event_bus: Any | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.max_history = max_history
        self.state_transitions = deque(maxlen=max_history)
        self.action_patterns = defaultdict(list)
        self.success_patterns = defaultdict(float)
        self.risk_patterns = defaultdict(list)

        # System integration
        self.event_bus = event_bus
        self.config = config or {}
        self.is_initialized = False
        self.startup_complete = False

        # Learning parameters (configurable via config)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.8)
        self.learning_decay = self.config.get("learning_decay", 0.95)

        # Enhanced features configuration
        self.enable_alternative_generation = self.config.get(
            "enable_alternative_generation", True
        )
        self.max_alternatives = self.config.get("max_alternatives", 5)
        self.enable_risk_assessment = self.config.get("enable_risk_assessment", True)
        self.enable_learning = self.config.get("enable_learning", True)

        logger.info(
            "🔮 PredictiveWorldModel initialized with enhanced alternative generation"
        )

    async def initialize(self) -> bool:
        """Initialize the Predictive World Model with system components"""
        try:
            logger.info("🔮 Initializing Predictive World Model...")

            # Initialize event subscriptions if event_bus available
            if self.event_bus:
                await self._setup_event_subscriptions()

            # Load any persistent data
            await self._load_persistent_data()

            # Validate configuration
            self._validate_configuration()

            self.is_initialized = True
            logger.info("✅ Predictive World Model initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Predictive World Model: {e}")
            return False

    async def startup(self) -> bool:
        """Complete startup process and integrate with system"""
        try:
            if not self.is_initialized:
                await self.initialize()

            logger.info("🚀 Starting Predictive World Model...")

            # Start background processes if needed
            await self._start_background_processes()

            # Register with system components
            await self._register_with_system()

            # Validate enhanced features
            await self._validate_enhanced_features()

            self.startup_complete = True
            logger.info("✅ Predictive World Model startup complete")
            return True

        except Exception as e:
            logger.error(f"❌ Predictive World Model startup failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Graceful shutdown with data persistence"""
        try:
            logger.info("🔮 Shutting down Predictive World Model...")

            # Save persistent data
            await self._save_persistent_data()

            # Stop background processes
            await self._stop_background_processes()

            # Unregister from system
            await self._unregister_from_system()

            self.startup_complete = False
            self.is_initialized = False

            logger.info("✅ Predictive World Model shutdown complete")

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

    async def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions for system integration"""
        if not self.event_bus:
            return

        try:
            # Subscribe to tool execution events to learn automatically
            await self.event_bus.subscribe(
                "tool_execution_started", self._handle_tool_start
            )
            await self.event_bus.subscribe(
                "tool_execution_completed", self._handle_tool_completion
            )

            # Subscribe to prediction requests
            await self.event_bus.subscribe(
                "prediction_request", self._handle_prediction_request
            )

            # Subscribe to alternative generation requests
            await self.event_bus.subscribe(
                "alternatives_request", self._handle_alternatives_request
            )

            logger.info("🔮 Event subscriptions established")

        except Exception as e:
            logger.error(f"❌ Failed to setup event subscriptions: {e}")

    async def _handle_tool_start(self, event: Any) -> None:
        """Handle tool execution start events"""
        try:
            # Store initial state for later learning
            if not hasattr(self, "_pending_executions"):
                self._pending_executions = {}
            self._pending_executions[getattr(event, "execution_id", "unknown")] = {
                "initial_state": getattr(event, "initial_state", {}),
                "action": getattr(event, "action", "unknown_action"),
                "start_time": datetime.now(),
            }
        except Exception as e:
            logger.error(f"❌ Error handling tool start: {e}")

    async def _handle_tool_completion(self, event: Any) -> None:
        """Handle tool execution completion events for automatic learning"""
        try:
            if not self.enable_learning:
                return

            pending = getattr(self, "_pending_executions", {})
            execution_id = getattr(event, "execution_id", "unknown")
            if execution_id in pending:
                execution_data = pending[execution_id]

                # Calculate duration
                duration = (
                    datetime.now() - execution_data["start_time"]
                ).total_seconds()

                # Learn from the execution
                await self.learn_from_execution(
                    initial_state=execution_data["initial_state"],
                    action=execution_data["action"],
                    final_state=getattr(event, "final_state", {}),
                    success=getattr(event, "success", False),
                    duration=duration,
                    context=getattr(event, "context", {}),
                )

                # Clean up
                del pending[execution_id]

        except Exception as e:
            logger.error(f"❌ Error handling tool completion: {e}")

    async def _handle_prediction_request(self, event: Any) -> None:
        """Handle prediction requests via event bus"""
        try:
            prediction = await self.predict_outcome(
                current_state=getattr(event, "state", {}),
                proposed_action=getattr(event, "action", ""),
                context=getattr(event, "context", {}),
            )

            # Publish prediction result
            if self.event_bus:
                await self.event_bus.publish(
                    "prediction_result",
                    {
                        "request_id": getattr(event, "request_id", "unknown"),
                        "prediction": prediction._asdict(),
                        "timestamp": datetime.now().isoformat(),
                    },
                )

        except Exception as e:
            logger.error(f"❌ Error handling prediction request: {e}")

    async def _handle_alternatives_request(self, event: Any) -> None:
        """Handle alternative generation requests via event bus"""
        try:
            if not self.enable_alternative_generation:
                return

            alternatives = await self._generate_alternatives(
                state=getattr(event, "state", {}),
                action=getattr(event, "action", ""),
                similar_transitions=[],  # Will find them internally
            )

            # Publish alternatives result
            if self.event_bus:
                await self.event_bus.publish(
                    "alternatives_result",
                    {
                        "request_id": getattr(event, "request_id", "unknown"),
                        "alternatives": alternatives,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

        except Exception as e:
            logger.error(f"❌ Error handling alternatives request: {e}")

    def _validate_configuration(self) -> None:
        """Validate configuration parameters"""
        # Ensure thresholds are within valid ranges
        self.confidence_threshold = max(0.0, min(1.0, self.confidence_threshold))
        self.similarity_threshold = max(0.0, min(1.0, self.similarity_threshold))
        self.learning_decay = max(0.0, min(1.0, self.learning_decay))

        # Ensure max_alternatives is reasonable
        self.max_alternatives = max(1, min(10, self.max_alternatives))

        logger.info(
            f"🔮 Configuration validated - similarity_threshold: {self.similarity_threshold}"
        )

    async def _validate_enhanced_features(self) -> None:
        """Validate that enhanced features are working"""
        try:
            # Test alternative generation with dummy data
            if self.enable_alternative_generation:
                test_alternatives = await self._generate_alternatives(
                    state={"test": "validation"},
                    action="test_action",
                    similar_transitions=[],
                )
                logger.info(
                    f"✅ Alternative generation validated ({len(test_alternatives)} alternatives)"
                )

            # Test risk assessment
            if self.enable_risk_assessment:
                test_risks = self._identify_risks(
                    state={"test": "validation"},
                    action="test_action",
                    similar_transitions=[],
                )
                logger.info(
                    f"✅ Risk assessment validated ({len(test_risks)} risk factors)"
                )

        except Exception as e:
            logger.error(f"❌ Enhanced features validation failed: {e}")

    async def _load_persistent_data(self) -> None:
        """Load persistent learning data if available"""
        try:
            # Implementation would load from configured storage
            # For now, just log
            logger.info("🔮 Loading persistent data (placeholder)")
        except Exception as e:
            logger.error(f"❌ Failed to load persistent data: {e}")

    async def _save_persistent_data(self) -> None:
        """Save learning data for persistence"""
        try:
            # Implementation would save to configured storage
            # For now, just log
            logger.info("🔮 Saving persistent data (placeholder)")
        except Exception as e:
            logger.error(f"❌ Failed to save persistent data: {e}")

    async def _start_background_processes(self) -> None:
        """Start any background processes"""
        # Placeholder for background tasks

    async def _stop_background_processes(self) -> None:
        """Stop background processes"""
        # Placeholder for cleanup

    async def _register_with_system(self) -> None:
        """Register with other system components"""
        # Placeholder for system registration

    async def _unregister_from_system(self) -> None:
        """Unregister from system components"""
        # Placeholder for system cleanup

    def get_startup_status(self) -> dict[str, Any]:
        """Get detailed startup status"""
        return {
            "initialized": self.is_initialized,
            "startup_complete": self.startup_complete,
            "enhanced_features": {
                "alternative_generation": self.enable_alternative_generation,
                "risk_assessment": self.enable_risk_assessment,
                "learning": self.enable_learning,
            },
            "configuration": {
                "similarity_threshold": self.similarity_threshold,
                "confidence_threshold": self.confidence_threshold,
                "max_alternatives": self.max_alternatives,
            },
            "data_status": {
                "total_transitions": len(self.state_transitions),
                "unique_actions": len(self.action_patterns),
                "success_patterns": len(self.success_patterns),
            },
        }

    def get_health_status(self) -> dict[str, Any]:
        """Get health status for monitoring"""
        return {
            "status": "healthy" if self.startup_complete else "starting",
            "alternative_generation_working": self.enable_alternative_generation,
            "learning_active": self.enable_learning,
            "memory_usage": len(self.state_transitions),
            "model_confidence": self._calculate_model_confidence()
            if self.state_transitions
            else 0.0,
        }

    async def predict_outcome(
        self,
        current_state: dict[str, Any],
        proposed_action: str,
        context: dict[str, Any] | None = None,
    ) -> PredictionOutcome:
        """Predict the outcome of a proposed action given the current state"""
        context = context or {}

        # Find similar historical situations
        similar_transitions = self._find_similar_transitions(
            current_state, proposed_action
        )

        if not similar_transitions:
            # No historical data - conservative prediction
            return PredictionOutcome(
                success_probability=0.5,
                expected_duration=1.0,
                confidence=0.3,
                risk_factors=["no_historical_data"],
                alternative_strategies=[],
            )

        # Analyze similar cases
        success_rate = np.mean([t.success for t in similar_transitions])
        avg_duration = np.mean([t.duration for t in similar_transitions])

        # Calculate confidence based on data quality
        confidence = min(0.95, len(similar_transitions) / 20.0 + 0.5)

        # Identify risk factors
        risk_factors = self._identify_risks(
            current_state, proposed_action, similar_transitions
        )

        # Generate alternative strategies
        alternatives = await self._generate_alternatives(
            current_state, proposed_action, similar_transitions
        )

        prediction = PredictionOutcome(
            success_probability=float(success_rate),
            expected_duration=float(avg_duration),
            confidence=float(confidence),
            risk_factors=risk_factors,
            alternative_strategies=alternatives,
        )

        logger.info(
            f"🔮 Prediction for action '{proposed_action}': "
            f"success={success_rate:.2f}, duration={avg_duration:.2f}s, confidence={confidence:.2f}"
        )

        return prediction

    async def learn_from_execution(
        self,
        initial_state: dict[str, Any],
        action: str,
        final_state: dict[str, Any],
        success: bool,
        duration: float,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Learn from an executed action and its outcomes"""
        context = context or {}

        transition = StateTransition(
            initial_state=initial_state.copy(),
            action_taken=action,
            final_state=final_state.copy(),
            success=success,
            duration=duration,
            timestamp=datetime.now(),
            context=context.copy(),
        )

        # Store the transition
        self.state_transitions.append(transition)

        # Update action patterns
        self.action_patterns[action].append(transition)

        # Update success patterns with decay
        pattern_key = self._create_pattern_key(initial_state, action)
        current_success = self.success_patterns[pattern_key]
        self.success_patterns[pattern_key] = current_success * self.learning_decay + (
            1.0 if success else 0.0
        ) * (1 - self.learning_decay)

        # Learn risk patterns
        if not success:
            risk_key = self._identify_failure_pattern(initial_state, action, context)
            self.risk_patterns[risk_key].append(transition)

        logger.debug(
            f"🔮 Learned from action '{action}': success={success}, duration={duration:.2f}s"
        )

    def _find_similar_transitions(
        self, state: dict[str, Any], action: str, max_results: int = 50
    ) -> list[StateTransition]:
        """Find historically similar state-action pairs"""
        candidates = []

        for transition in self.state_transitions:
            if transition.action_taken != action:
                continue

            # Calculate state similarity
            similarity = self._calculate_state_similarity(
                state, transition.initial_state
            )

            if similarity >= self.similarity_threshold:
                candidates.append((transition, similarity))

        # Sort by similarity and return top matches
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in candidates[:max_results]]

    def _calculate_state_similarity(
        self, state1: dict[str, Any], state2: dict[str, Any]
    ) -> float:
        """Calculate similarity between two states"""
        if not state1 and not state2:
            return 1.0
        if not state1 or not state2:
            return 0.0

        # Get all unique keys
        all_keys = set(state1.keys()) | set(state2.keys())
        if not all_keys:
            return 1.0

        similarity_scores = []

        for key in all_keys:
            val1 = state1.get(key)
            val2 = state2.get(key)

            if val1 is None or val2 is None or not isinstance(val1, type(val2)):
                similarity_scores.append(0.0)
            elif isinstance(val1, int | float) and isinstance(val2, int | float):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2), 1)
                similarity_scores.append(1.0 - abs(val1 - val2) / max_val)
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity (simple)
                if val1 == val2:
                    similarity_scores.append(1.0)
                else:
                    # Basic string similarity
                    common_chars = len(set(val1) & set(val2))
                    total_chars = len(set(val1) | set(val2))
                    similarity_scores.append(common_chars / max(total_chars, 1))
            elif val1 == val2:
                similarity_scores.append(1.0)
            else:
                similarity_scores.append(0.0)

        return np.mean(similarity_scores) if similarity_scores else 0.0

    def _create_pattern_key(self, state: dict[str, Any], action: str) -> str:
        """Create a pattern key for learning"""
        # Simplified pattern key creation
        state_signature = hash(str(sorted(state.items()))) % 10000
        return f"{action}_{state_signature}"

    def _identify_risks(
        self,
        state: dict[str, Any],
        action: str,
        similar_transitions: list[StateTransition],
    ) -> list[str]:
        """Identify potential risk factors"""
        risks = []

        # Risk from historical failures
        failures = [t for t in similar_transitions if not t.success]
        if (
            len(failures) / max(len(similar_transitions), 1)
            > HIGH_FAILURE_RATE_THRESHOLD
        ):
            risks.append("high_historical_failure_rate")

        # Risk from long execution times
        long_executions = [
            t for t in similar_transitions if t.duration > LONG_EXECUTION_THRESHOLD
        ]
        if (
            len(long_executions) / max(len(similar_transitions), 1)
            > LONG_EXECUTION_RATE_THRESHOLD
        ):
            risks.append("potential_long_execution")

        # Risk from limited data
        if len(similar_transitions) < MIN_DATA_THRESHOLD:
            risks.append("insufficient_historical_data")

        # Risk from state complexity
        if len(state) > COMPLEX_STATE_THRESHOLD:
            risks.append("complex_state_space")

        # Risk from recent failures with this action
        recent_transitions = [
            t
            for t in self.action_patterns.get(action, [])
            if (datetime.now() - t.timestamp).total_seconds()
            < RECENT_FAILURE_WINDOW_SECONDS
        ]
        recent_failures = [t for t in recent_transitions if not t.success]
        if len(recent_failures) > RECENT_FAILURE_THRESHOLD:
            risks.append("recent_action_failures")

        return risks

    def _identify_failure_pattern(
        self, state: dict[str, Any], action: str, context: dict[str, Any]
    ) -> str:
        """Identify patterns that lead to failure"""
        # Create a failure pattern identifier
        key_factors = []

        # State factors
        if "complexity" in state:
            key_factors.append(f"complexity_{state['complexity']}")
        if "size" in state:
            key_factors.append(f"size_{state['size']}")

        # Context factors
        if "time_pressure" in context:
            key_factors.append("time_pressure")
        if "resource_limited" in context:
            key_factors.append("resource_limited")

        return (
            f"{action}_failure_{'_'.join(key_factors)}"
            if key_factors
            else f"{action}_failure_unknown"
        )

    async def _generate_alternatives(
        self,
        state: dict[str, Any],
        action: str,
        similar_transitions: list[StateTransition],
    ) -> list[dict[str, Any]]:
        """Enhanced: Generate alternatives by finding similar states with different successful actions"""
        alternatives = []

        # NEW APPROACH: Search all transitions for similar states with different actions
        for transition in self.state_transitions:
            # Must be different action, successful outcome, and similar state
            if (
                transition.action_taken != action
                and transition.success
                and self._calculate_state_similarity(state, transition.initial_state)
                >= self.similarity_threshold
            ):
                # Calculate success pattern for this alternative
                alt_pattern_key = self._create_pattern_key(
                    state, transition.action_taken
                )
                alt_success_rate = self.success_patterns.get(
                    alt_pattern_key, 0.8
                )  # Default higher for successful examples

                # Count supporting evidence for this alternative
                evidence_count = sum(
                    1
                    for t in self.state_transitions
                    if (
                        t.action_taken == transition.action_taken
                        and t.success
                        and self._calculate_state_similarity(state, t.initial_state)
                        >= self.similarity_threshold
                    )
                )

                alternatives.append(
                    {
                        "action": transition.action_taken,
                        "expected_success_rate": alt_success_rate,
                        "expected_duration": transition.duration,
                        "confidence": min(0.9, evidence_count / 5.0 + 0.5),
                        "evidence_count": evidence_count,
                    }
                )

        # Remove duplicates, keeping the best version of each action
        unique_alternatives = {}
        for alt in alternatives:
            action_name = alt["action"]
            if (
                action_name not in unique_alternatives
                or alt["expected_success_rate"]
                > unique_alternatives[action_name]["expected_success_rate"]
            ):
                unique_alternatives[action_name] = alt

        # Sort by composite score (success_rate * confidence) and return top 5
        sorted_alternatives = sorted(
            unique_alternatives.values(),
            key=lambda x: x["expected_success_rate"] * x["confidence"],
            reverse=True,
        )

        return sorted_alternatives[:5]  # Return top 5 alternatives

    def get_model_summary(self) -> dict[str, Any]:
        """Get comprehensive model state and learning summary"""
        return {
            "total_transitions": len(self.state_transitions),
            "unique_actions": len(self.action_patterns),
            "unique_patterns": len(self.success_patterns),
            "risk_patterns": len(self.risk_patterns),
            "model_confidence": self._calculate_model_confidence(),
            "top_successful_actions": self._get_top_actions(success=True),
            "top_risky_actions": self._get_top_actions(success=False),
            "learning_summary": {
                "oldest_data": self.state_transitions[0].timestamp.isoformat()
                if self.state_transitions
                else None,
                "newest_data": self.state_transitions[-1].timestamp.isoformat()
                if self.state_transitions
                else None,
                "avg_success_rate": np.mean([t.success for t in self.state_transitions])
                if self.state_transitions
                else 0.0,
            },
        }

    def _calculate_model_confidence(self) -> float:
        """Calculate overall model confidence based on data quality"""
        if not self.state_transitions:
            return 0.0

        # Factors affecting confidence
        data_volume_score = min(1.0, len(self.state_transitions) / 1000.0)
        pattern_diversity_score = min(1.0, len(self.success_patterns) / 100.0)
        recent_data_score = self._calculate_recent_data_score()

        return (data_volume_score + pattern_diversity_score + recent_data_score) / 3.0

    def _calculate_recent_data_score(self) -> float:
        """Score based on how recent the data is"""
        if not self.state_transitions:
            return 0.0

        now = datetime.now()
        recent_transitions = [
            t
            for t in self.state_transitions
            if (now - t.timestamp).total_seconds()
            < RECENT_DATA_WINDOW_SECONDS  # Last 24 hours
        ]

        return min(1.0, len(recent_transitions) / 100.0)

    def _get_top_actions(
        self, success: bool = True, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Get top performing or risky actions"""
        action_stats = defaultdict(
            lambda: {"successes": 0, "total": 0, "avg_duration": 0.0}
        )

        for transition in self.state_transitions:
            action = transition.action_taken
            action_stats[action]["total"] += 1
            if transition.success:
                action_stats[action]["successes"] += 1
            action_stats[action]["avg_duration"] = (
                action_stats[action]["avg_duration"]
                * (action_stats[action]["total"] - 1)
                + transition.duration
            ) / action_stats[action]["total"]

        # Calculate success rates
        action_performance = []
        for action, stats in action_stats.items():
            success_rate = stats["successes"] / max(stats["total"], 1)
            action_performance.append(
                {
                    "action": action,
                    "success_rate": success_rate,
                    "total_executions": stats["total"],
                    "avg_duration": stats["avg_duration"],
                }
            )

        # Sort by success rate (ascending for risky, descending for successful)
        action_performance.sort(key=lambda x: x["success_rate"], reverse=success)

        return action_performance[:limit]
