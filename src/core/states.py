"""
REUG v9.0 State Machine Definitions for Super Alita

This module defines the state machine that governs the cognitive execution flow,
replacing the brittle linear tool chain with a robust, state-driven architecture.

REUG v9.0 States:
- READY: Initial state, waiting for user input
- ENGAGE: Processing user request, determining intent
- UNDERSTAND: Building context, loading memory, routing to tools
- GENERATE: Executing tools, creating responses
- ERROR_RECOVERY: Handling failures gracefully
- COMPLETE: Final state, ready for next turn

State transitions are event-driven and include comprehensive error handling.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any

from src.core.session import Session

# Import state types and session for concurrency controls
from src.core.state_types import StateType
from src.core.tool_types import ToolSpec

logger = logging.getLogger(__name__)

# Compatibility aliases for legacy imports
State = StateType

__all__ = [
    "StateMachine",
    "State",
    "TransitionTrigger",
    "Context",
    "ToolSpec",
]

# Circuit breaker and mailbox constants
MAILBOX_MAX_SIZE = 100
MAILBOX_WARNING_SIZE = 80
TRANSITION_RATE_LIMIT = 50
CIRCUIT_BREAKER_TIMEOUT = 30.0  # seconds


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for the state machine."""

    is_open: bool = False
    transition_count: int = 0
    last_transition_time: float = 0.0
    open_time: float = 0.0
    failure_count: int = 0
    last_failure_time: float = 0.0
    transition_window_start: float = 0.0


class TransitionTrigger(Enum):
    """Event types that trigger state transitions"""

    USER_INPUT = auto()  # User provides input
    INTENT_DETECTED = auto()  # Intent successfully parsed
    CONTEXT_LOADED = auto()  # Memory and context retrieved
    TOOLS_ROUTED = auto()  # Tools selected and prepared
    TOOL_SUCCESS = auto()  # Tool execution completed successfully
    TOOL_FAILURE = auto()  # Tool execution failed
    RESPONSE_READY = auto()  # Response generated and ready
    DYNAMIC_TOOL_REQUEST = auto()  # Request to create dynamic tool
    DYNAMIC_TOOL_CREATED = auto()  # Dynamic tool successfully created
    ERROR_OCCURRED = auto()  # Recoverable error detected
    FATAL_ERROR = auto()  # Unrecoverable error occurred
    RECOVERY_SUCCESS = auto()  # Error recovery completed
    TURN_COMPLETE = auto()  # Cognitive turn finished
    SHUTDOWN_REQUESTED = auto()  # System shutdown initiated


@dataclass
class StateContext:
    """Context data passed between states"""

    user_input: str | None = None
    detected_intent: str | None = None
    tools_selected: list[ToolSpec] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    response: str | None = None
    error_count: int = 0
    memory_context: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    turn_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    # NEW: Script of Thought integration fields
    sot_parse_result: Any = None  # ScriptOfThought object
    sot_execution_state: dict[str, Any] = field(default_factory=dict)

    # NEW: Computational environment fields
    comp_env_session: str | None = None
    comp_env_results: dict[str, Any] = field(default_factory=dict)

    # NEW: Dynamic tool creation fields
    dynamic_tool_name: str | None = None
    dynamic_tool_schema: Any = None  # ToolSchema object
    error_message: str | None = None
    response_content: dict[str, Any] = field(default_factory=dict)

# Backwards-compatible alias
Context = StateContext

@dataclass
class StateTransition:
    """Defines a state transition"""

    from_state: StateType
    trigger: TransitionTrigger
    to_state: StateType
    condition: Callable[[StateContext], bool] | None = None
    action: Callable[[StateContext], Any] | None = None
    description: str = ""


class StateMachine:
    """
    REUG v9.0 State Machine with Concurrency Safety

    Manages cognitive execution flow with resilient state transitions,
    comprehensive error handling, event-driven architecture, and
    concurrency-safe input handling.
    """

    def __init__(self, event_bus=None, session: Session | None = None):
        self.current_state = StateType.READY
        self.context = StateContext()
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)

        # Session with concurrency controls
        self.session = session or Session()

        # Transition registry
        self.transitions: dict[
            tuple[StateType, TransitionTrigger], StateTransition
        ] = {}
        self._setup_transitions()

        # Define ignored triggers (no warnings)
        self.ignored_triggers = {
            (StateType.GENERATE, TransitionTrigger.USER_INPUT),  # Queue instead
            (StateType.COMPLETE, TransitionTrigger.RESPONSE_READY),  # Stale completion
            (StateType.READY, TransitionTrigger.RESPONSE_READY),  # Out-of-order
            (StateType.UNDERSTAND, TransitionTrigger.USER_INPUT),  # Queue instead
        }

        # State handlers
        self.state_handlers: dict[StateType, Callable] = {
            StateType.READY: self._handle_ready_state,
            StateType.ENGAGE: self._handle_engage_state,
            StateType.UNDERSTAND: self._handle_understand_state,
            StateType.GENERATE: self._handle_generate_state,
            StateType.ERROR_RECOVERY: self._handle_error_recovery_state,
            StateType.COMPLETE: self._handle_complete_state,
            StateType.SHUTDOWN: self._handle_shutdown_state,
        }

        # Metrics
        self.state_transitions_count = 0
        self.error_recoveries_count = 0  # Circuit breaker state
        self.circuit_breaker = CircuitBreakerState()

        self.successful_turns_count = 0

        # Mailbox for queued input
        self.mailbox: list[str] = []

        # Placeholder metrics registry
        self.metrics_registry = None

    def _setup_transitions(self):
        """Configure all valid state transitions"""
        transitions = [
            # READY state transitions
            StateTransition(
                StateType.READY,
                TransitionTrigger.USER_INPUT,
                StateType.ENGAGE,
                description="User provides input, begin processing",
            ),
            StateTransition(
                StateType.READY,
                TransitionTrigger.SHUTDOWN_REQUESTED,
                StateType.SHUTDOWN,
                description="System shutdown requested",
            ),
            # ENGAGE state transitions
            StateTransition(
                StateType.ENGAGE,
                TransitionTrigger.INTENT_DETECTED,
                StateType.UNDERSTAND,
                description="Intent successfully detected, move to understanding",
            ),
            StateTransition(
                StateType.ENGAGE,
                TransitionTrigger.ERROR_OCCURRED,
                StateType.ERROR_RECOVERY,
                description="Error during intent detection",
            ),
            StateTransition(
                StateType.ENGAGE,
                TransitionTrigger.FATAL_ERROR,
                StateType.COMPLETE,
                description="Fatal error, complete turn with failure",
            ),
            # UNDERSTAND state transitions
            StateTransition(
                StateType.UNDERSTAND,
                TransitionTrigger.TOOLS_ROUTED,
                StateType.GENERATE,
                description="Tools selected and routed, begin generation",
            ),
            StateTransition(
                StateType.UNDERSTAND,
                TransitionTrigger.DYNAMIC_TOOL_REQUEST,
                StateType.CREATE_DYNAMIC_TOOL,
                description="Dynamic tool creation requested",
            ),
            StateTransition(
                StateType.UNDERSTAND,
                TransitionTrigger.ERROR_OCCURRED,
                StateType.ERROR_RECOVERY,
                description="Error during context loading or tool routing",
            ),
            StateTransition(
                StateType.UNDERSTAND,
                TransitionTrigger.FATAL_ERROR,
                StateType.COMPLETE,
                description="Fatal error in understanding phase",
            ),
            # GENERATE state transitions
            StateTransition(
                StateType.GENERATE,
                TransitionTrigger.TOOL_SUCCESS,
                StateType.COMPLETE,
                description="Tool execution successful, complete turn",
            ),
            StateTransition(
                StateType.GENERATE,
                TransitionTrigger.TOOL_FAILURE,
                StateType.ERROR_RECOVERY,
                description="Tool execution failed, attempt recovery",
            ),
            StateTransition(
                StateType.GENERATE,
                TransitionTrigger.RESPONSE_READY,
                StateType.COMPLETE,
                description="Response generated successfully",
            ),
            StateTransition(
                StateType.GENERATE,
                TransitionTrigger.FATAL_ERROR,
                StateType.COMPLETE,
                description="Fatal error during generation",
            ),
            # CREATE_DYNAMIC_TOOL state transitions
            StateTransition(
                StateType.CREATE_DYNAMIC_TOOL,
                TransitionTrigger.DYNAMIC_TOOL_CREATED,
                StateType.COMPLETE,
                description="Dynamic tool created successfully",
            ),
            StateTransition(
                StateType.CREATE_DYNAMIC_TOOL,
                TransitionTrigger.ERROR_OCCURRED,
                StateType.ERROR_RECOVERY,
                description="Error during dynamic tool creation",
            ),
            StateTransition(
                StateType.CREATE_DYNAMIC_TOOL,
                TransitionTrigger.FATAL_ERROR,
                StateType.COMPLETE,
                description="Fatal error in dynamic tool creation",
            ),
            # ERROR_RECOVERY state transitions
            StateTransition(
                StateType.ERROR_RECOVERY,
                TransitionTrigger.RECOVERY_SUCCESS,
                StateType.UNDERSTAND,
                condition=lambda ctx: ctx.error_count < 3,
                description="Recovery successful, retry understanding phase",
            ),
            StateTransition(
                StateType.ERROR_RECOVERY,
                TransitionTrigger.RECOVERY_SUCCESS,
                StateType.COMPLETE,
                condition=lambda ctx: ctx.error_count >= 3,
                description="Recovery successful but max retries reached",
            ),
            StateTransition(
                StateType.ERROR_RECOVERY,
                TransitionTrigger.FATAL_ERROR,
                StateType.COMPLETE,
                description="Recovery failed, complete turn with error",
            ),
            # COMPLETE state transitions
            StateTransition(
                StateType.COMPLETE,
                TransitionTrigger.TURN_COMPLETE,
                StateType.READY,
                description="Turn completed, ready for next input",
            ),
            StateTransition(
                StateType.COMPLETE,
                TransitionTrigger.USER_INPUT,
                StateType.ENGAGE,
                description="New user input received, begin new processing cycle",
            ),
            StateTransition(
                StateType.COMPLETE,
                TransitionTrigger.SHUTDOWN_REQUESTED,
                StateType.SHUTDOWN,
                description="Shutdown requested after turn completion",
            ),
            # SHUTDOWN transitions (terminal state)
            # No outgoing transitions from SHUTDOWN
        ]

        # Build transition lookup table
        for transition in transitions:
            key = (transition.from_state, transition.trigger)
            self.transitions[key] = transition

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should prevent transitions."""
        current_time = time.time()

        # Check mailbox size
        if len(self.mailbox) > MAILBOX_MAX_SIZE:
            logger.warning(
                f"Circuit breaker: Mailbox size {len(self.mailbox)} exceeds limit {MAILBOX_MAX_SIZE}"
            )
            self._trip_circuit_breaker("mailbox_overflow")
            return False

        # Check transition rate
        if current_time - self.circuit_breaker.transition_window_start >= 1.0:
            # Reset transition count every second
            self.circuit_breaker.transition_count = 0
            self.circuit_breaker.transition_window_start = current_time

        if self.circuit_breaker.transition_count >= TRANSITION_RATE_LIMIT:
            logger.warning(
                f"Circuit breaker: Transition rate {self.circuit_breaker.transition_count}/s exceeds limit"
            )
            self._trip_circuit_breaker("rate_limit")
            return False

        # Check if circuit is open and timeout has passed
        if self.circuit_breaker.is_open:
            if (
                current_time - self.circuit_breaker.last_failure_time
                > CIRCUIT_BREAKER_TIMEOUT
            ):
                logger.info("Circuit breaker timeout expired, attempting reset")
                self._reset_circuit_breaker()
            else:
                logger.warning("Circuit breaker is open, blocking transition")
                return False

        return True

    def _trip_circuit_breaker(self, reason: str) -> None:
        """Trip the circuit breaker."""
        self.circuit_breaker.is_open = True
        self.circuit_breaker.failure_count += 1
        self.circuit_breaker.last_failure_time = time.time()

        # Emit metrics
        if self.metrics_registry:
            self.metrics_registry.increment_counter(
                "sa_fsm_circuit_breaker_trips_total", {"reason": reason}
            )
            self.metrics_registry.set_gauge("sa_fsm_circuit_breaker_open", 1.0)

        logger.error(
            f"Circuit breaker tripped: {reason} (failure count: {self.circuit_breaker.failure_count})"
        )

    def _reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker."""
        self.circuit_breaker.is_open = False
        self.circuit_breaker.failure_count = 0

        # Emit metrics
        if self.metrics_registry:
            self.metrics_registry.set_gauge("sa_fsm_circuit_breaker_open", 0.0)

        logger.info("Circuit breaker reset")

    def _update_transition_metrics(self) -> None:
        """Update transition rate metrics."""
        self.circuit_breaker.transition_count += 1
        self.circuit_breaker.last_transition_time = time.time()

        # Update mailbox pressure metric
        mailbox_pressure = len(self.mailbox) / MAILBOX_MAX_SIZE
        if self.metrics_registry:
            self.metrics_registry.set_gauge("sa_fsm_mailbox_pressure", mailbox_pressure)

        # Warning if approaching limits
        if len(self.mailbox) > MAILBOX_WARNING_SIZE:
            logger.warning(
                f"Mailbox size {len(self.mailbox)} approaching limit {MAILBOX_MAX_SIZE}"
            )

    async def transition(
        self, trigger: TransitionTrigger, context_updates: dict[str, Any] | None = None
    ) -> bool:
        """
        Execute a state transition with concurrency safety

        Args:
            trigger: The event triggering the transition
            context_updates: Optional updates to apply to context

        Returns:
            bool: True if transition was successful
        """
        # Serialize transitions to prevent races
        async with self.session.transition_lock:
            # Apply context updates
            if context_updates:
                for key, value in context_updates.items():
                    setattr(self.context, key, value)

            # Look up transition
            transition_key = (self.current_state, trigger)

            # Check if this trigger should be ignored
            if transition_key in self.ignored_triggers:
                self.session.increment_ignored_triggers()
                self.logger.debug(
                    f"Ignoring trigger {trigger} in state {self.current_state}"
                )
                return False

            transition = self.transitions.get(transition_key)

            if not transition:
                self.logger.warning(
                    f"No transition defined from {self.current_state} with trigger {trigger}"
                )
                return False

            # Check transition condition
            if transition.condition and not transition.condition(self.context):
                self.logger.debug(
                    f"Transition condition failed for {self.current_state} -> {transition.to_state}"
                )
                return False

            # Execute transition action
            if transition.action:
                try:
                    result = transition.action(self.context)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    self.logger.error(f"Transition action failed: {e}")
                    return False

            # Update state
            old_state = self.current_state
            self.current_state = transition.to_state
            self.state_transitions_count += 1

            self.logger.info(
                f"State transition: {old_state} -> {self.current_state} ({trigger.name})"
            )

            # Emit state change event
            if self.event_bus:
                await self.event_bus.emit(
                    "state_changed",
                    source_plugin="state_machine",
                    old_state=old_state.name,
                    new_state=self.current_state.name,
                    trigger=trigger.name,
                    context_summary=self._get_context_summary(),
                )

            return True

    async def handle_current_state(self) -> TransitionTrigger | None:
        """
        Execute the handler for the current state

        Returns:
            Optional[TransitionTrigger]: Next trigger to process, if any
        """
        handler = self.state_handlers.get(self.current_state)
        if not handler:
            self.logger.error(f"No handler defined for state {self.current_state}")
            return TransitionTrigger.FATAL_ERROR

        try:
            result = handler()
            if asyncio.iscoroutine(result):
                return await result
            return result
        except Exception as e:
            self.logger.error(f"State handler error in {self.current_state}: {e}")
            self.context.error_count += 1

            if self.context.error_count >= 3:
                return TransitionTrigger.FATAL_ERROR
            else:
                return TransitionTrigger.ERROR_OCCURRED

    def handle_user_input(self, text: str):
        """
        Handle user input with re-entrant protection.

        If the FSM is in a state that accepts input, trigger transition.
        Otherwise, queue the input for later processing.
        """
        if self.session.can_accept_input(self.current_state):
            # Create async task for transition
            asyncio.create_task(
                self.transition(TransitionTrigger.USER_INPUT, {"user_input": text})
            )
        else:
            # Queue for later processing
            self.session.queue_input(text)
            self.logger.debug(
                f"Queued input in state {self.current_state}: {text[:50]}..."
            )

    async def handle_response_ready(self, op_id: str, payload: dict):
        """
        Handle response ready with idempotency protection.

        Only accept RESPONSE_READY if:
        1. op_id matches current inflight operation
        2. FSM is in GENERATE state
        """
        if not self.session.is_operation_current(op_id):
            self.logger.debug(
                f"Ignoring stale RESPONSE_READY (op_id={op_id}, current={self.session.inflight_op_id})"
            )
            return

        if self.current_state != StateType.GENERATE:
            self.logger.debug(f"Ignoring RESPONSE_READY in state {self.current_state}")
            return

        # Mark operation as complete
        self.session.complete_operation(op_id)

        # Proceed with transition
        await self.transition(TransitionTrigger.RESPONSE_READY, payload)

    async def _drain_one_mailbox_item(self):
        """
        Drain exactly one input from mailbox after completing a turn.
        """
        queued_input = await self.session.drain_one_input()
        if queued_input:
            self.logger.debug(f"Processing queued input: {queued_input[:50]}...")
            await self.transition(
                TransitionTrigger.USER_INPUT, {"user_input": queued_input}
            )
        return None

    def _get_context_summary(
        self,
    ) -> dict[str, Any]:  # pragma: no cover - simple accessor
        """Get a summary of current context for logging/events"""
        return {
            "user_input_length": len(self.context.user_input or ""),
            "intent": self.context.detected_intent,
            "tools_count": len(self.context.tools_selected),
            "error_count": self.context.error_count,
            "turn_id": self.context.turn_id,
        }

    # State Handlers (to be implemented in execution_flow.py)
    async def _handle_ready_state(self) -> TransitionTrigger | None:
        """Handle READY state - waiting for user input"""
        self.logger.debug("In READY state, waiting for input")
        return None  # Wait for external trigger

    async def _handle_engage_state(self) -> TransitionTrigger | None:
        """Handle ENGAGE state - process user input and detect intent"""
        self.logger.debug("In ENGAGE state, processing user input")
        # Implementation will be in execution_flow.py
        return TransitionTrigger.INTENT_DETECTED

    async def _handle_understand_state(self) -> TransitionTrigger | None:
        """Handle UNDERSTAND state - load context and route tools"""
        self.logger.debug("In UNDERSTAND state, building context")
        # Implementation will be in execution_flow.py
        return TransitionTrigger.TOOLS_ROUTED

    async def _handle_generate_state(self) -> TransitionTrigger | None:
        """Handle GENERATE state - execute tools and create response"""
        self.logger.debug("In GENERATE state, executing tools")
        # Implementation will be in execution_flow.py
        return TransitionTrigger.RESPONSE_READY

    async def _handle_error_recovery_state(self) -> TransitionTrigger | None:
        """Handle ERROR_RECOVERY state - attempt to recover from errors"""
        self.logger.debug("In ERROR_RECOVERY state, attempting recovery")
        # Implementation will be in execution_flow.py
        return TransitionTrigger.RECOVERY_SUCCESS

    async def _handle_complete_state(self) -> TransitionTrigger | None:
        """Handle COMPLETE state - finalize turn and prepare for next"""
        self.logger.debug("In COMPLETE state, finalizing turn")
        self.successful_turns_count += 1

        # Schedule draining one queued input after completion
        asyncio.create_task(self._drain_one_mailbox_item())

        return TransitionTrigger.TURN_COMPLETE

    async def _handle_shutdown_state(self) -> TransitionTrigger | None:
        """Handle SHUTDOWN state - graceful system shutdown"""
        self.logger.info("In SHUTDOWN state, performing graceful shutdown")
        return None  # Terminal state

    def reset_context(self, session_id: str | None = None):
        """Reset context for new cognitive turn"""
        self.context = StateContext(
            session_id=session_id,
            turn_id=f"turn_{self.successful_turns_count + 1}",
            timestamp=datetime.now(UTC),
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get state machine metrics"""
        return {
            "current_state": self.current_state.name,
            "transitions_count": self.state_transitions_count,
            "error_recoveries": self.error_recoveries_count,
            "successful_turns": self.successful_turns_count,
            "current_error_count": self.context.error_count,
        }
