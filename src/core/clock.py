#!/usr/bin/env python3
"""
Monotonic Clock Abstraction
Provides race-safe, deterministic time handling for circuit breakers and performance monitoring.
"""

import time
from abc import ABC, abstractmethod


class ClockInterface(ABC):
    """Abstract clock interface for dependency injection"""

    @abstractmethod
    def now(self) -> float:
        """Return current time in seconds"""
        pass


class MonotonicClock(ClockInterface):
    """Production monotonic clock using time.monotonic()"""

    def now(self) -> float:
        """Return monotonic time in seconds"""
        return time.monotonic()


class FakeClock(ClockInterface):
    """Fake clock for deterministic testing"""

    def __init__(self, initial_time: float = 0.0):
        self._time = float(initial_time)

    def now(self) -> float:
        """Return current fake time"""
        return self._time

    def tick(self, delta: float) -> float:
        """Advance time by delta seconds and return new time"""
        self._time += float(delta)
        return self._time

    def set_time(self, new_time: float) -> float:
        """Set absolute time and return new time"""
        self._time = float(new_time)
        return self._time
