from __future__ import annotations

import time


class CircuitBreaker:
    """
    Simple circuit breaker.
    - failure_threshold: failures in window to open
    - recovery_timeout: seconds to half-open
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.state = "closed"  # closed, open, half_open
        self.open_until = 0.0

    def allowed(self) -> bool:
        now = time.time()
        if self.state == "open":
            if now >= self.open_until:
                self.state = "half_open"
                return True
            return False
        return True

    def on_success(self) -> None:
        self.failures = 0
        self.state = "closed"

    def on_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.state = "open"
            self.open_until = time.time() + self.recovery_timeout

    def is_open(self) -> bool:
        return self.state == "open"
