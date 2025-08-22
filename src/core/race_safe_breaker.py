#!/usr/bin/env python3
"""
Race-Safe Circuit Breaker Implementation
Provides async-safe circuit breaker with monotonic clock, proper state transitions,
decorrelated jitter backoff, and probe gate protection.
"""

import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from core.clock import ClockInterface, MonotonicClock
from core.events import create_event

T = TypeVar("T")


class BreakerState(Enum):
    """Circuit breaker states with proper state machine"""

    CLOSED = "closed"  # Normal operation, allowing requests
    OPEN = "open"  # Failing fast, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery with limited probes


@dataclass
class BreakerConfig:
    """Configuration for circuit breaker behavior"""

    failure_threshold: int = 5
    window_seconds: float = 10.0
    open_min_seconds: float = 2.0
    open_cap_seconds: float = 60.0
    base_seconds: float = 1.0
    half_open_max_probes: int = 1
    monitored_exceptions: list[type] = field(default_factory=lambda: [Exception])


class ProbeGate:
    """Single-flight gate for half-open probes to prevent double execution"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._in_probe = False

    async def try_probe(self, probe_func: Callable[[], T]) -> T:
        """Execute probe function with single-flight protection"""
        async with self._lock:
            if self._in_probe:
                raise RuntimeError("Probe already in progress")
            self._in_probe = True

        try:
            return await probe_func()
        finally:
            async with self._lock:
                self._in_probe = False


def decorrelated_jitter_backoff(previous: float, base: float, cap: float) -> float:
    """
    AWS-style decorrelated jitter backoff algorithm
    next ∈ [base, min(cap, base + U(0,1) * (previous * 3 - base))]
    """
    next_value = min(cap, base + random.random() * (previous * 3 - base))
    return max(base, next_value)


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors"""

    pass


class CircuitBreakerOpenError(CircuitBreakerError):
    """Raised when circuit breaker is open and rejecting requests"""

    pass


class RaceSafeCircuitBreaker:
    """
    Race-safe circuit breaker with async locks, monotonic clock,
    and proper state transition management.
    """

    def __init__(
        self,
        name: str,
        config: BreakerConfig,
        clock: ClockInterface | None = None,
        event_bus=None,
        logger: logging.Logger | None = None,
    ):
        self.name = name
        self.config = config
        self._clock = clock or MonotonicClock()
        self.event_bus = event_bus
        self.logger = logger or logging.getLogger(__name__)

        # State management with async protection
        self._state = BreakerState.CLOSED
        self._state_lock = asyncio.Lock()

        # Failure tracking with monotonic timestamps
        self._failures: list[float] = []  # Ring buffer of failure timestamps
        self._opened_at: float | None = None
        self._backoff_previous = config.open_min_seconds

        # Probe protection
        self._probe_gate = ProbeGate()

        # Metrics
        self._total_calls = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._state_transitions = 0

    @property
    def state(self) -> BreakerState:
        """Get current circuit breaker state (thread-safe read)"""
        return self._state

    async def _set_state(self, new_state: BreakerState, reason: str) -> None:
        """
        Atomically set new state with transition validation and logging.
        Must be called with _state_lock held.
        """
        now = self._clock.now()
        old_state = self._state

        if old_state == new_state:
            return

        # Validate legal state transitions
        legal_transitions = {
            BreakerState.CLOSED: {BreakerState.OPEN},
            BreakerState.OPEN: {BreakerState.HALF_OPEN},
            BreakerState.HALF_OPEN: {BreakerState.CLOSED, BreakerState.OPEN},
        }

        if new_state not in legal_transitions[old_state]:
            raise ValueError(f"Illegal state transition: {old_state} -> {new_state}")

        # Update state atomically
        self._state = new_state
        self._state_transitions += 1

        # Handle state-specific logic
        if new_state == BreakerState.OPEN:
            self._opened_at = now
            # Evolve backoff using decorrelated jitter
            self._backoff_previous = decorrelated_jitter_backoff(
                self._backoff_previous,
                self.config.base_seconds,
                self.config.open_cap_seconds,
            )

        # Emit events and logs
        await self._emit_state_change_event(old_state, new_state, reason, now)
        self.logger.info(
            f"Circuit breaker {self.name} state transition",
            extra={
                "from_state": old_state.value,
                "to_state": new_state.value,
                "reason": reason,
                "timestamp": now,
                "backoff_seconds": (
                    self._backoff_previous if new_state == BreakerState.OPEN else None
                ),
            },
        )

    async def _emit_state_change_event(
        self, old_state: BreakerState, new_state: BreakerState, reason: str, now: float
    ) -> None:
        """Emit state change event to event bus"""
        if self.event_bus:
            try:
                event = create_event(
                    "circuit_breaker_state_changed",
                    breaker_name=self.name,
                    old_state=old_state.value,
                    new_state=new_state.value,
                    reason=reason,
                    timestamp=now,
                    total_calls=self._total_calls,
                    failed_calls=self._failed_calls,
                    source_plugin="race_safe_circuit_breaker",
                )
                await self.event_bus.emit(event)
            except Exception as e:
                self.logger.warning(
                    f"Failed to emit circuit breaker state change event: {e}"
                )

    async def _decay_failure_window(self, now: float) -> None:
        """Remove failures outside the time window (must hold _state_lock)"""
        cutoff = now - self.config.window_seconds
        self._failures = [t for t in self._failures if t >= cutoff]

    async def allow_request(self) -> bool:
        """
        Check if request should be allowed through the circuit breaker.
        Returns True if request can proceed, False if it should be rejected.
        """
        async with self._state_lock:
            now = self._clock.now()
            await self._decay_failure_window(now)

            if self._state == BreakerState.CLOSED:
                # Check if we should trip to OPEN
                if len(self._failures) >= self.config.failure_threshold:
                    await self._set_state(
                        BreakerState.OPEN, "failure_threshold_exceeded"
                    )
                    return False
                return True

            elif self._state == BreakerState.OPEN:
                # Check if cooldown period has elapsed
                if (
                    self._opened_at is not None
                    and (now - self._opened_at) >= self._backoff_previous
                ):
                    await self._set_state(BreakerState.HALF_OPEN, "cooldown_elapsed")
                    return False  # This request is denied, but next can be probe
                return False

            elif self._state == BreakerState.HALF_OPEN:
                return True  # Allow limited probes (bounded by probe gate)

        return False

    async def record_result(
        self, success: bool, exception: Exception | None = None
    ) -> None:
        """Record the result of a request execution"""
        async with self._state_lock:
            now = self._clock.now()
            self._total_calls += 1

            if success:
                self._successful_calls += 1
                # If we're in HALF_OPEN and got success, transition to CLOSED
                if self._state == BreakerState.HALF_OPEN:
                    await self._set_state(BreakerState.CLOSED, "probe_succeeded")
            else:
                # Check if this exception should be counted
                should_count = exception is None or any(
                    isinstance(exception, exc_type)
                    for exc_type in self.config.monitored_exceptions
                )

                if should_count:
                    self._failed_calls += 1
                    self._failures.append(now)

                    # If we're in HALF_OPEN and got failure, transition back to OPEN
                    if self._state == BreakerState.HALF_OPEN:
                        await self._set_state(BreakerState.OPEN, "probe_failed")

    async def execute_with_breaker(self, func: Callable[[], T]) -> T:
        """
        Execute function with circuit breaker protection.
        Handles both sync and async functions.
        """
        # Check if request is allowed
        if not await self.allow_request():
            raise CircuitBreakerOpenError(
                f"Circuit breaker {self.name} is open, rejecting request"
            )

        # Execute the function
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func()
            else:
                result = func()

            await self.record_result(True)
            return result

        except Exception as e:
            await self.record_result(False, e)
            raise

    async def execute_probe(self, probe_func: Callable[[], T]) -> T:
        """
        Execute a probe function in HALF_OPEN state with single-flight protection.
        Should only be called when state is HALF_OPEN.
        """
        if self._state != BreakerState.HALF_OPEN:
            raise ValueError(
                f"Probe can only be executed in HALF_OPEN state, current: {self._state}"
            )

        try:
            return await self._probe_gate.try_probe(probe_func)
        except RuntimeError as e:
            if "Probe already in progress" in str(e):
                # Another probe is running, shed this request
                raise CircuitBreakerOpenError(
                    f"Circuit breaker {self.name} probe already in progress, shedding request"
                ) from None
            raise

    def get_metrics(self) -> dict[str, Any]:
        """Get current circuit breaker metrics"""
        return {
            "name": self.name,
            "state": self._state.value,
            "total_calls": self._total_calls,
            "successful_calls": self._successful_calls,
            "failed_calls": self._failed_calls,
            "state_transitions": self._state_transitions,
            "current_failures": len(self._failures),
            "success_rate": (
                (self._successful_calls / self._total_calls * 100)
                if self._total_calls > 0
                else 100.0
            ),
            "opened_at": self._opened_at,
            "backoff_seconds": self._backoff_previous,
        }

    async def force_state(
        self, state: BreakerState, reason: str = "manual_override"
    ) -> None:
        """Force circuit breaker to specific state (for testing/admin purposes)"""
        async with self._state_lock:
            await self._set_state(state, reason)

    async def reset(self) -> None:
        """Reset circuit breaker to initial state"""
        async with self._state_lock:
            self._failures.clear()
            self._opened_at = None
            self._backoff_previous = self.config.open_min_seconds
            self._total_calls = 0
            self._successful_calls = 0
            self._failed_calls = 0
            await self._set_state(BreakerState.CLOSED, "manual_reset")
