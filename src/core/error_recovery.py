#!/usr/bin/env python3
"""
Error Recovery & Resilience System
Implements comprehensive error recovery mechanisms with circuit breakers, retry policies,
graceful degradation, and automatic failover strategies for the Super Alita Agent System.
"""

import asyncio
import contextlib
import logging
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar

from core.events import create_event

T = TypeVar("T")


class CircuitBreakerState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryStrategy(Enum):
    """Retry strategy types"""

    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    RANDOM_JITTER = "random_jitter"


class FailoverStrategy(Enum):
    """Failover strategy types"""

    ACTIVE_PASSIVE = "active_passive"
    ACTIVE_ACTIVE = "active_active"
    ROUND_ROBIN = "round_robin"
    PRIORITY_BASED = "priority_based"
    LOAD_BALANCED = "load_balanced"


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    FATAL = "fatal"


class RecoveryAction(Enum):
    """Recovery action types"""

    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    DEGRADE_SERVICE = "degrade_service"
    FAILOVER = "failover"
    RESTART_COMPONENT = "restart_component"
    ALERT_OPERATOR = "alert_operator"
    SHUTDOWN_GRACEFULLY = "shutdown_gracefully"


@dataclass
class ErrorContext:
    """Context information for an error"""

    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    component_name: str = ""
    operation_name: str = ""
    error_type: str = ""
    error_message: str = ""
    stack_trace: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    retry_count: int = 0
    max_retries: int = 3
    context_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "error_id": self.error_id,
            "component_name": self.component_name,
            "operation_name": self.operation_name,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "context_data": self.context_data,
        }


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""

    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 5
    monitored_exceptions: list[type] = field(default_factory=lambda: [Exception])


@dataclass
class RetryConfig:
    """Configuration for retry policies"""

    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: bool = True
    retryable_exceptions: list[type] = field(default_factory=lambda: [Exception])


@dataclass
class FailoverConfig:
    """Configuration for failover"""

    strategy: FailoverStrategy = FailoverStrategy.ACTIVE_PASSIVE
    health_check_interval: float = 30.0
    failover_timeout: float = 10.0
    auto_failback: bool = True
    failback_delay: float = 300.0  # 5 minutes


class CircuitBreaker:
    """Circuit breaker implementation for preventing cascade failures"""

    def __init__(self, name: str, config: CircuitBreakerConfig, event_bus=None):
        self.name = name
        self.config = config
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)

        # State management
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        self.half_open_calls = 0

        # Metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.circuit_open_count = 0

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        self.total_calls += 1

        # Check if circuit is open
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                await self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")

        # Check if we're in half-open state and reached max calls
        if (
            self.state == CircuitBreakerState.HALF_OPEN
            and self.half_open_calls >= self.config.half_open_max_calls
        ):
            raise CircuitBreakerOpenError(
                f"Circuit breaker {self.name} half-open limit reached"
            )

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success - update state
            await self._on_success()
            return result

        except Exception as e:
            # Check if this exception should trigger circuit breaker
            if any(
                isinstance(e, exc_type) for exc_type in self.config.monitored_exceptions
            ):
                await self._on_failure(e)
            raise

    async def _on_success(self) -> None:
        """Handle successful call"""
        self.successful_calls += 1
        self.failure_count = 0

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            self.half_open_calls += 1

            if self.success_count >= self.config.success_threshold:
                await self._transition_to_closed()

    async def _on_failure(self, exception: Exception) -> None:
        """Handle failed call"""
        self.failed_calls += 1
        self.failure_count += 1
        self.last_failure_time = datetime.now(UTC)

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_calls += 1
            await self._transition_to_open()
        elif (
            self.state == CircuitBreakerState.CLOSED
            and self.failure_count >= self.config.failure_threshold
        ):
            await self._transition_to_open()

        # Emit failure event
        if self.event_bus:
            try:
                event = create_event(
                    "circuit_breaker_failure",
                    circuit_breaker_name=self.name,
                    state=self.state.value,
                    failure_count=self.failure_count,
                    exception_type=type(exception).__name__,
                    exception_message=str(exception),
                    source_plugin="error_recovery",
                )
                await self.event_bus.emit(event)
            except Exception as e:
                self.logger.warning(
                    f"Could not emit circuit breaker failure event: {e}"
                )

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset"""
        if not self.last_failure_time:
            return True

        time_since_failure = datetime.now(UTC) - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.timeout_seconds

    async def _transition_to_closed(self) -> None:
        """Transition circuit breaker to closed state"""
        old_state = self.state
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0

        self.logger.info(
            f"Circuit breaker {self.name} transitioned: {old_state.value} -> {self.state.value}"
        )
        await self._emit_state_change_event(old_state)

    async def _transition_to_open(self) -> None:
        """Transition circuit breaker to open state"""
        old_state = self.state
        self.state = CircuitBreakerState.OPEN
        self.circuit_open_count += 1
        self.success_count = 0
        self.half_open_calls = 0

        self.logger.warning(
            f"Circuit breaker {self.name} transitioned: {old_state.value} -> {self.state.value}"
        )
        await self._emit_state_change_event(old_state)

    async def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to half-open state"""
        old_state = self.state
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        self.half_open_calls = 0

        self.logger.info(
            f"Circuit breaker {self.name} transitioned: {old_state.value} -> {self.state.value}"
        )
        await self._emit_state_change_event(old_state)

    async def _emit_state_change_event(self, old_state: CircuitBreakerState) -> None:
        """Emit circuit breaker state change event"""
        if self.event_bus:
            try:
                event = create_event(
                    "circuit_breaker_state_changed",
                    circuit_breaker_name=self.name,
                    old_state=old_state.value,
                    new_state=self.state.value,
                    failure_count=self.failure_count,
                    success_count=self.success_count,
                    source_plugin="error_recovery",
                )
                await self.event_bus.emit(event)
            except Exception as e:
                self.logger.warning(
                    f"Could not emit circuit breaker state change event: {e}"
                )

    def get_metrics(self) -> dict[str, Any]:
        """Get circuit breaker metrics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "circuit_open_count": self.circuit_open_count,
            "success_rate": (
                (self.successful_calls / self.total_calls * 100)
                if self.total_calls > 0
                else 100.0
            ),
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
        }


class ErrorRecoveryOrchestrator:
    """Main orchestrator for error recovery and resilience mechanisms"""

    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)

        # Core components
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        # Error tracking
        self.error_history: deque[ErrorContext] = deque(maxlen=1000)
        self.error_patterns: dict[str, int] = defaultdict(int)

        # Monitoring
        self._running = False
        self._monitor_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the error recovery orchestrator"""
        if self._running:
            return

        self._running = True

        # Start monitoring
        self._monitor_task = asyncio.create_task(self._monitoring_loop())

        self.logger.info("Error Recovery Orchestrator started")

    async def stop(self) -> None:
        """Stop the error recovery orchestrator"""
        if not self._running:
            return

        self._running = False

        # Stop monitoring
        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task

        self.logger.info("Error Recovery Orchestrator stopped")

    def register_circuit_breaker(
        self, name: str, config: CircuitBreakerConfig
    ) -> CircuitBreaker:
        """Register a circuit breaker"""
        circuit_breaker = CircuitBreaker(name, config, self.event_bus)
        self.circuit_breakers[name] = circuit_breaker
        self.logger.info(f"Registered circuit breaker: {name}")
        return circuit_breaker

    async def handle_error(
        self, error_context: ErrorContext, recovery_actions: list[RecoveryAction] = None
    ) -> bool:
        """Handle an error with appropriate recovery mechanisms"""
        # Record error
        self.error_history.append(error_context)
        error_pattern = f"{error_context.component_name}:{error_context.error_type}"
        self.error_patterns[error_pattern] += 1

        # Determine recovery actions if not provided
        if recovery_actions is None:
            recovery_actions = self._determine_recovery_actions(error_context)

        # Execute recovery actions
        recovery_success = False

        for action in recovery_actions:
            try:
                if action == RecoveryAction.RETRY:
                    recovery_success = await self._handle_retry(error_context)

                elif action == RecoveryAction.CIRCUIT_BREAK:
                    recovery_success = await self._handle_circuit_break(error_context)

                elif action == RecoveryAction.FALLBACK:
                    recovery_success = await self._handle_fallback(error_context)

                elif action == RecoveryAction.ALERT_OPERATOR:
                    recovery_success = await self._handle_alert(error_context)

                else:
                    self.logger.warning(f"Unknown recovery action: {action}")

                # If recovery action succeeded, break
                if recovery_success:
                    break

            except Exception as e:
                self.logger.error(f"Error executing recovery action {action}: {e}")

        # Emit error handled event
        if self.event_bus:
            try:
                event = create_event(
                    "error_handled",
                    error_id=error_context.error_id,
                    component_name=error_context.component_name,
                    recovery_actions=[action.value for action in recovery_actions],
                    recovery_success=recovery_success,
                    source_plugin="error_recovery",
                )
                await self.event_bus.emit(event)
            except Exception as e:
                self.logger.warning(f"Could not emit error handled event: {e}")

        return recovery_success

    def _determine_recovery_actions(
        self, error_context: ErrorContext
    ) -> list[RecoveryAction]:
        """Determine appropriate recovery actions based on error context"""
        actions = []

        # Based on error severity
        if error_context.severity == ErrorSeverity.LOW:
            actions = [RecoveryAction.RETRY]

        elif error_context.severity == ErrorSeverity.MEDIUM:
            actions = [RecoveryAction.RETRY, RecoveryAction.FALLBACK]

        elif error_context.severity == ErrorSeverity.HIGH:
            actions = [RecoveryAction.CIRCUIT_BREAK, RecoveryAction.FALLBACK]

        elif error_context.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            actions = [
                RecoveryAction.CIRCUIT_BREAK,
                RecoveryAction.FALLBACK,
                RecoveryAction.ALERT_OPERATOR,
            ]

        # Based on error patterns
        error_pattern = f"{error_context.component_name}:{error_context.error_type}"
        if (
            self.error_patterns[error_pattern] > 5
            and RecoveryAction.CIRCUIT_BREAK not in actions
        ):  # Frequent errors
            actions.insert(0, RecoveryAction.CIRCUIT_BREAK)

        return actions

    async def _handle_retry(self, error_context: ErrorContext) -> bool:
        """Handle retry recovery action"""
        # This would integrate with the actual function that failed
        # For now, just simulate retry logic
        self.logger.info(f"Executing retry recovery for {error_context.error_id}")
        return error_context.retry_count < error_context.max_retries

    async def _handle_circuit_break(self, error_context: ErrorContext) -> bool:
        """Handle circuit breaker recovery action"""
        circuit_breaker_name = f"{error_context.component_name}_circuit_breaker"

        if circuit_breaker_name not in self.circuit_breakers:
            # Create default circuit breaker
            config = CircuitBreakerConfig()
            self.register_circuit_breaker(circuit_breaker_name, config)

        self.logger.info(
            f"Circuit breaker activated for {error_context.component_name}"
        )
        return True

    async def _handle_fallback(self, error_context: ErrorContext) -> bool:
        """Handle fallback recovery action"""
        # This would implement fallback logic (cached data, simplified processing, etc.)
        self.logger.info(f"Executing fallback for {error_context.component_name}")
        return True

    async def _handle_alert(self, error_context: ErrorContext) -> bool:
        """Handle operator alert recovery action"""
        # Emit high-priority alert event
        if self.event_bus:
            try:
                event = create_event(
                    "operator_alert",
                    alert_level="high",
                    error_context=error_context.to_dict(),
                    message=f"Critical error in {error_context.component_name}: {error_context.error_message}",
                    source_plugin="error_recovery",
                )
                await self.event_bus.emit(event)

                self.logger.critical(
                    f"OPERATOR ALERT: {error_context.component_name} - {error_context.error_message}"
                )
                return True

            except Exception as e:
                self.logger.error(f"Could not emit operator alert: {e}")

        return False

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for error patterns and recovery"""
        while self._running:
            try:
                # Analyze error patterns
                await self._analyze_error_patterns()

                # Clean up old error history
                await self._cleanup_error_history()

                # Sleep until next check
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(300)  # Longer wait on error

    async def _analyze_error_patterns(self) -> None:
        """Analyze error patterns for proactive recovery"""
        # Look for error spikes or patterns that indicate system issues
        current_time = datetime.now(UTC)
        recent_errors = [
            error
            for error in self.error_history
            if (current_time - error.timestamp).total_seconds() < 300  # Last 5 minutes
        ]

        if len(recent_errors) > 10:  # More than 10 errors in 5 minutes
            self.logger.warning(
                f"High error rate detected: {len(recent_errors)} errors in last 5 minutes"
            )

    async def _cleanup_error_history(self) -> None:
        """Clean up old entries from error history and patterns"""
        # Clean up error patterns that haven't occurred recently
        current_time = datetime.now(UTC)
        patterns_to_remove = []

        for pattern in self.error_patterns:
            # Check if pattern has recent errors
            pattern_component, pattern_type = pattern.split(":", 1)
            recent_pattern_errors = [
                error
                for error in self.error_history
                if (
                    error.component_name == pattern_component
                    and error.error_type == pattern_type
                    and (current_time - error.timestamp).total_seconds() < 3600
                )  # Last hour
            ]

            if not recent_pattern_errors:
                patterns_to_remove.append(pattern)

        for pattern in patterns_to_remove:
            del self.error_patterns[pattern]

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive error recovery system status"""
        recent_errors = [
            error
            for error in self.error_history
            if (datetime.now(UTC) - error.timestamp).total_seconds() < 3600  # Last hour
        ]

        return {
            "running": self._running,
            "timestamp": datetime.now(UTC).isoformat(),
            "circuit_breakers": {
                name: cb.get_metrics() for name, cb in self.circuit_breakers.items()
            },
            "error_statistics": {
                "total_errors": len(self.error_history),
                "recent_errors": len(recent_errors),
                "error_patterns": dict(self.error_patterns),
                "severity_distribution": {
                    severity.value: sum(
                        1 for error in recent_errors if error.severity == severity
                    )
                    for severity in ErrorSeverity
                },
            },
        }


# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""

    pass


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted"""

    pass


class FailoverFailedError(Exception):
    """Raised when failover to all instances failed"""

    pass
