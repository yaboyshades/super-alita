"""Self-Healing Circuit Breakers with constitutional review.

Implements circuit breaker pattern with:
- Fail-fast behavior when services degrade
- Automatic state management (CLOSED, OPEN, HALF_OPEN)
- Constitutional compliance checks before auto-recovery
- Exponential backoff for recovery attempts
- Health metrics and alerting
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    # Failure threshold
    failure_threshold: int = 5  # Failures before opening
    failure_window_seconds: float = 60.0  # Time window for failures

    # Timeout
    timeout_seconds: float = 30.0  # Request timeout

    # Recovery
    recovery_timeout_seconds: float = 60.0  # Time before trying HALF_OPEN
    half_open_max_calls: int = 3  # Test calls in HALF_OPEN
    success_threshold: int = 2  # Successes to close from HALF_OPEN

    # Constitutional
    require_constitutional_review: bool = True  # Gate recovery on compliance
    constitutional_threshold: float = 0.75  # Min compliance to recover


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""

    state: CircuitState
    failure_count: int = 0
    success_count: int = 0
    timeout_count: int = 0
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None
    state_changes: int = 0
    total_calls: int = 0
    rejected_calls: int = 0


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is OPEN."""

    pass


class CircuitBreaker:
    """Self-healing circuit breaker with constitutional compliance gates.

    Prevents cascading failures by failing fast when downstream services
    degrade. Requires constitutional review before auto-recovery.
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        constitutional_validator: Callable[[str], float] | None = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Circuit breaker name (for logging/metrics)
            config: Configuration parameters
            constitutional_validator: Function returning compliance score
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.constitutional_validator = constitutional_validator

        # State
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats(state=self.state)

        # Failure tracking
        self.failures: deque[datetime] = deque(
            maxlen=self.config.failure_threshold * 2
        )

        # Recovery tracking
        self.opened_at: datetime | None = None
        self.half_open_calls = 0
        self.half_open_successes = 0

        # Lock for state transitions
        self._lock = asyncio.Lock()

    async def call(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with circuit breaker protection.

        Args:
            fn: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is OPEN
            asyncio.TimeoutError: If call times out
        """
        async with self._lock:
            self.stats.total_calls += 1

            # Check state
            if self.state == CircuitState.OPEN:
                # Check if recovery timeout elapsed
                if self._should_attempt_recovery():
                    await self._transition_to_half_open()
                else:
                    self.stats.rejected_calls += 1
                    raise CircuitBreakerError(
                        f"Circuit breaker {self.name} is OPEN"
                    )

            # HALF_OPEN: limit test calls
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    self.stats.rejected_calls += 1
                    raise CircuitBreakerError(
                        f"Circuit breaker {self.name} HALF_OPEN max calls"
                    )
                self.half_open_calls += 1

        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                fn(*args, **kwargs),
                timeout=self.config.timeout_seconds,
            )
            await self._on_success()
            return result

        except TimeoutError:
            await self._on_timeout()
            raise

        except Exception as e:
            await self._on_failure()
            raise e

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            now = datetime.now(UTC)
            self.stats.success_count += 1
            self.stats.last_success_time = now

            if self.state == CircuitState.HALF_OPEN:
                self.half_open_successes += 1

                # Check if ready to close
                if self.half_open_successes >= self.config.success_threshold:
                    # Constitutional review required?
                    if (
                        self.config.require_constitutional_review
                        and self.constitutional_validator
                    ):
                        score = await asyncio.to_thread(
                            self.constitutional_validator,
                            f"circuit_breaker_{self.name}_recovery",
                        )
                        if score >= self.config.constitutional_threshold:
                            await self._transition_to_closed()
                        # else: stay HALF_OPEN
                    else:
                        await self._transition_to_closed()

    async def _on_failure(self) -> None:
        """Handle failed call."""
        async with self._lock:
            now = datetime.now(UTC)
            self.stats.failure_count += 1
            self.stats.last_failure_time = now
            self.failures.append(now)

            if self.state == CircuitState.HALF_OPEN:
                # Failure in HALF_OPEN -> back to OPEN
                await self._transition_to_open()

            elif self.state == CircuitState.CLOSED:
                # Check failure threshold
                recent = self._count_recent_failures()
                if recent >= self.config.failure_threshold:
                    await self._transition_to_open()

    async def _on_timeout(self) -> None:
        """Handle timeout."""
        async with self._lock:
            self.stats.timeout_count += 1
            # Treat timeout as failure
            await self._on_failure()

    def _count_recent_failures(self) -> int:
        """Count failures within failure window."""
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=self.config.failure_window_seconds)
        return sum(1 for ts in self.failures if ts >= cutoff)

    def _should_attempt_recovery(self) -> bool:
        """Check if recovery timeout has elapsed."""
        if not self.opened_at:
            return False

        now = datetime.now(UTC)
        elapsed = (now - self.opened_at).total_seconds()
        return elapsed >= self.config.recovery_timeout_seconds

    async def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        self.state = CircuitState.OPEN
        self.opened_at = datetime.now(UTC)
        self.stats.state = self.state
        self.stats.state_changes += 1

    async def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.half_open_successes = 0
        self.stats.state = self.state
        self.stats.state_changes += 1

    async def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.opened_at = None
        self.failures.clear()
        self.stats.state = self.state
        self.stats.state_changes += 1

    async def force_open(self) -> None:
        """Manually open circuit breaker."""
        async with self._lock:
            await self._transition_to_open()

    async def force_close(self) -> None:
        """Manually close circuit breaker."""
        async with self._lock:
            await self._transition_to_closed()

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.stats.failure_count,
            "success_count": self.stats.success_count,
            "timeout_count": self.stats.timeout_count,
            "total_calls": self.stats.total_calls,
            "rejected_calls": self.stats.rejected_calls,
            "state_changes": self.stats.state_changes,
            "recent_failures": self._count_recent_failures(),
            "last_failure": (
                self.stats.last_failure_time.isoformat()
                if self.stats.last_failure_time
                else None
            ),
            "last_success": (
                self.stats.last_success_time.isoformat()
                if self.stats.last_success_time
                else None
            ),
        }


class CircuitBreakerRegistry:
    """Registry managing multiple circuit breakers."""

    def __init__(self):
        """Initialize registry."""
        self.breakers: dict[str, CircuitBreaker] = {}

    def register(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        constitutional_validator: Callable[[str], float] | None = None,
    ) -> CircuitBreaker:
        """Register a circuit breaker.

        Args:
            name: Circuit breaker name
            config: Configuration
            constitutional_validator: Validator function

        Returns:
            Created CircuitBreaker instance
        """
        breaker = CircuitBreaker(name, config, constitutional_validator)
        self.breakers[name] = breaker
        return breaker

    def get(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name."""
        return self.breakers.get(name)

    async def health_check(self) -> dict[str, Any]:
        """Check health of all circuit breakers."""
        return {
            "total_breakers": len(self.breakers),
            "open_breakers": sum(
                1
                for b in self.breakers.values()
                if b.state == CircuitState.OPEN
            ),
            "half_open_breakers": sum(
                1
                for b in self.breakers.values()
                if b.state == CircuitState.HALF_OPEN
            ),
            "breakers": {
                name: breaker.get_stats()
                for name, breaker in self.breakers.items()
            },
        }


# Example usage
async def example_circuit_breaker() -> None:
    """Example demonstrating CircuitBreaker usage."""

    # Flaky service
    call_count = 0

    async def flaky_service() -> str:
        nonlocal call_count
        call_count += 1

        # Fail first 5 calls
        if call_count <= 5:
            raise RuntimeError(f"Service failure {call_count}")

        return f"Success on call {call_count}"

    # Constitutional validator stub
    def validator(context: str) -> float:
        return 0.85  # Always pass for demo

    # Create circuit breaker
    config = CircuitBreakerConfig(
        failure_threshold=3,
        failure_window_seconds=10.0,
        recovery_timeout_seconds=5.0,
        constitutional_threshold=0.75,
    )
    breaker = CircuitBreaker("demo", config, validator)

    # Test calls
    for i in range(10):
        try:
            result = await breaker.call(flaky_service)
            print(f"Call {i+1}: {result}")
        except CircuitBreakerError as e:
            print(f"Call {i+1}: Circuit OPEN - {e}")
        except RuntimeError as e:
            print(f"Call {i+1}: Service error - {e}")

        await asyncio.sleep(1)

    # Stats
    print("\nFinal Stats:")
    print(breaker.get_stats())


if __name__ == "__main__":
    asyncio.run(example_circuit_breaker())
