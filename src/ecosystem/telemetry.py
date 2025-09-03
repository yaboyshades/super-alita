# src/ecosystem/telemetry.py
"""
A simple, dependency-free telemetry system for performance monitoring.
Designed to be a lightweight placeholder for a full OpenTelemetry (OTel)
or Prometheus integration.
"""
import time
from contextlib import contextmanager
from typing import Any, Protocol


class IMetricsSink(Protocol):
    """Interface for a system that receives telemetry data."""

    def record_timing(
        self, name: str, value_ms: float, tags: dict[str, Any]
    ) -> None: ...
    def increment_counter(self, name: str, tags: dict[str, Any]) -> None: ...


class NoopMetricsSink(IMetricsSink):
    """A metrics sink that discards all data."""

    def record_timing(self, name: str, value_ms: float, tags: dict[str, Any]) -> None:
        pass

    def increment_counter(self, name: str, tags: dict[str, Any]) -> None:
        pass


class Telemetry:
    """A simple telemetry collector."""

    def __init__(self, sink: IMetricsSink = None):
        self.sink = sink or NoopMetricsSink()
        self.counters: dict[str, int] = {}  # For simple in-memory counting

    @contextmanager
    def timer(self, name: str, tags: dict[str, Any] = None):
        """A context manager to time a block of code."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            self.sink.record_timing(name, duration_ms, tags or {})

    def increment_counter(self, name: str, tags: dict[str, Any] = None) -> None:
        """Increments a named counter."""
        self.counters[name] = self.counters.get(name, 0) + 1
        self.sink.increment_counter(name, tags or {})

    def get_counter(self, name: str) -> int:
        """Gets the current value of an in-memory counter."""
        return self.counters.get(name, 0)
