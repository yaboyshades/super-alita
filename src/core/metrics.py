#!/usr/bin/env python3
"""
Performance Metrics Specification
Defines standard metrics for Super-Alita performance monitoring.
"""

from abc import ABC, abstractmethod


class MetricsCollector(ABC):
    """Abstract interface for metrics collection"""

    @abstractmethod
    def counter(self, name: str, labels: dict[str, str] | None = None) -> "Counter":
        """Get or create a counter metric"""
        pass

    @abstractmethod
    def histogram(self, name: str, labels: dict[str, str] | None = None) -> "Histogram":
        """Get or create a histogram metric"""
        pass

    @abstractmethod
    def gauge(self, name: str, labels: dict[str, str] | None = None) -> "Gauge":
        """Get or create a gauge metric"""
        pass


class Counter(ABC):
    """Counter metric interface"""

    @abstractmethod
    def inc(self, amount: float = 1.0):
        """Increment counter"""
        pass


class Histogram(ABC):
    """Histogram metric interface"""

    @abstractmethod
    def observe(self, value: float):
        """Observe a value"""
        pass


class Gauge(ABC):
    """Gauge metric interface"""

    @abstractmethod
    def set(self, value: float):
        """Set gauge value"""
        pass

    @abstractmethod
    def inc(self, amount: float = 1.0):
        """Increment gauge"""
        pass

    @abstractmethod
    def dec(self, amount: float = 1.0):
        """Decrement gauge"""
        pass


# Standard Super-Alita Metrics
METRICS_SPEC = {
    # Request/Response Metrics
    "request_total": {
        "type": "counter",
        "description": "Total number of requests",
        "labels": ["endpoint", "method", "status"],
    },
    "request_duration_seconds": {
        "type": "histogram",
        "description": "Request duration in seconds",
        "labels": ["endpoint", "method"],
        "buckets": [
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
        ],
    },
    "request_size_bytes": {
        "type": "histogram",
        "description": "Request size in bytes",
        "labels": ["endpoint"],
        "buckets": [100, 1000, 10000, 100000, 1000000],
    },
    "response_size_bytes": {
        "type": "histogram",
        "description": "Response size in bytes",
        "labels": ["endpoint"],
        "buckets": [100, 1000, 10000, 100000, 1000000],
    },
    # Circuit Breaker Metrics
    "circuit_breaker_state": {
        "type": "gauge",
        "description": "Circuit breaker state (0=closed, 1=half_open, 2=open)",
        "labels": ["breaker_name"],
    },
    "circuit_breaker_calls_total": {
        "type": "counter",
        "description": "Total circuit breaker calls",
        "labels": ["breaker_name", "result"],
    },
    "circuit_breaker_failures_total": {
        "type": "counter",
        "description": "Total circuit breaker failures",
        "labels": ["breaker_name"],
    },
    # Cache Metrics
    "cache_operations_total": {
        "type": "counter",
        "description": "Total cache operations",
        "labels": ["cache_name", "operation", "result"],
    },
    "cache_hit_ratio": {
        "type": "gauge",
        "description": "Cache hit ratio (0.0-1.0)",
        "labels": ["cache_name"],
    },
    "cache_size_items": {
        "type": "gauge",
        "description": "Number of items in cache",
        "labels": ["cache_name"],
    },
    "cache_size_bytes": {
        "type": "gauge",
        "description": "Cache size in bytes",
        "labels": ["cache_name"],
    },
    # Event Bus Metrics
    "event_bus_messages_total": {
        "type": "counter",
        "description": "Total event bus messages",
        "labels": ["event_type", "source", "status"],
    },
    "event_bus_processing_duration_seconds": {
        "type": "histogram",
        "description": "Event processing duration",
        "labels": ["event_type"],
        "buckets": [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    },
    "event_bus_queue_size": {
        "type": "gauge",
        "description": "Event bus queue size",
        "labels": ["queue_name"],
    },
    # Plugin Metrics
    "plugin_operations_total": {
        "type": "counter",
        "description": "Total plugin operations",
        "labels": ["plugin_name", "operation", "status"],
    },
    "plugin_active_count": {
        "type": "gauge",
        "description": "Number of active plugins",
        "labels": [],
    },
    "plugin_memory_usage_bytes": {
        "type": "gauge",
        "description": "Plugin memory usage",
        "labels": ["plugin_name"],
    },
    # Resource Pool Metrics
    "pool_connections_active": {
        "type": "gauge",
        "description": "Active pool connections",
        "labels": ["pool_name"],
    },
    "pool_connections_idle": {
        "type": "gauge",
        "description": "Idle pool connections",
        "labels": ["pool_name"],
    },
    "pool_checkout_duration_seconds": {
        "type": "histogram",
        "description": "Pool checkout duration",
        "labels": ["pool_name"],
        "buckets": [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    },
    # System Metrics
    "memory_usage_bytes": {
        "type": "gauge",
        "description": "Memory usage in bytes",
        "labels": ["component"],
    },
    "cpu_usage_percent": {
        "type": "gauge",
        "description": "CPU usage percentage",
        "labels": ["component"],
    },
    "disk_usage_bytes": {
        "type": "gauge",
        "description": "Disk usage in bytes",
        "labels": ["path"],
    },
    "open_file_descriptors": {
        "type": "gauge",
        "description": "Number of open file descriptors",
        "labels": [],
    },
}


class NoOpMetricsCollector(MetricsCollector):
    """No-op metrics collector for testing/disabled metrics"""

    def counter(self, name: str, labels: dict[str, str] | None = None) -> Counter:
        return NoOpCounter()

    def histogram(self, name: str, labels: dict[str, str] | None = None) -> Histogram:
        return NoOpHistogram()

    def gauge(self, name: str, labels: dict[str, str] | None = None) -> Gauge:
        return NoOpGauge()


class NoOpCounter(Counter):
    def inc(self, amount: float = 1.0):
        pass


class NoOpHistogram(Histogram):
    def observe(self, value: float):
        pass


class NoOpGauge(Gauge):
    def set(self, value: float):
        pass

    def inc(self, amount: float = 1.0):
        pass

    def dec(self, amount: float = 1.0):
        pass
