#!/usr/bin/env python3
"""
DTA 2.0 Reliability Module (Simplified)

Basic reliability patterns for DTA 2.0 components.
This is a simplified version focusing on basic circuit breaker functionality.
"""

import time
from enum import Enum


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""


class AsyncCircuitBreaker:
    """Simple async circuit breaker implementation."""

    def __init__(
        self, name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    async def __aenter__(self):
        """Async context manager entry."""
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (
                self.last_failure_time
                and time.time() - self.last_failure_time > self.recovery_timeout
            ):
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerError(f"Circuit breaker {self.name} is open")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_type is None:
            # Success
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        else:
            # Failure
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

        return False  # Don't suppress exceptions
