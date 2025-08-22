#!/usr/bin/env python3
"""
DTA 2.0 Monitoring and Metrics Module

Production-grade monitoring, metrics collection, and observability for the
Deep Thinking Architecture. Provides Prometheus metrics, structured logging,
health checks, and analytics dashboards.
"""

import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

# Production monitoring dependencies
try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Fallback metrics classes
    class Counter:
        def __init__(self, *args, **kwargs):
            pass

        def inc(self, amount=1):
            pass

    class Histogram:
        def __init__(self, *args, **kwargs):
            pass

        def observe(self, amount):
            pass

        def time(self):
            return self._timer()

        class _timer:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

    class Gauge:
        def __init__(self, *args, **kwargs):
            pass

        def set(self, value):
            pass


class MetricType(Enum):
    """Types of metrics collected by DTA monitoring."""

    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DTAAlert:
    """Represents a monitoring alert."""

    timestamp: datetime
    level: AlertLevel
    component: str
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary format."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "component": self.component,
            "message": self.message,
            "metrics": self.metrics,
            "resolved": self.resolved,
        }


@dataclass
class DTAHealthStatus:
    """System health status information."""

    overall_status: str
    component_statuses: dict[str, str]
    last_check: datetime
    uptime_seconds: float
    error_rate: float
    avg_response_time_ms: float
    active_alerts: list[DTAAlert] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert health status to dictionary format."""
        return {
            "overall_status": self.overall_status,
            "component_statuses": self.component_statuses,
            "last_check": self.last_check.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "error_rate": self.error_rate,
            "avg_response_time_ms": self.avg_response_time_ms,
            "active_alerts": [alert.to_dict() for alert in self.active_alerts],
        }


class DTAStructuredLogger:
    """Production-grade structured logging for DTA components."""

    def __init__(self, component_name: str, log_level: str = "INFO"):
        self.component_name = component_name
        self.logger = logging.getLogger(f"dta.{component_name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Configure structured formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Add console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with structured data."""
        if kwargs:
            structured_data = json.dumps(kwargs, default=str, separators=(",", ":"))
            return f"{message} | {structured_data}"
        return message

    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self.logger.info(self._format_message(message, **kwargs))

    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self.logger.warning(self._format_message(message, **kwargs))

    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        self.logger.error(self._format_message(message, **kwargs))

    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self.logger.debug(self._format_message(message, **kwargs))

    def audit(self, action: str, user_id: str | None = None, **kwargs):
        """Log audit event with standardized format."""
        audit_data = {
            "audit_action": action,
            "timestamp": datetime.now(UTC).isoformat(),
            "component": self.component_name,
        }
        if user_id:
            audit_data["user_id"] = user_id
        audit_data.update(kwargs)

        self.logger.info(f"AUDIT: {action}", **audit_data)


class DTAMetricsCollector:
    """Prometheus-compatible metrics collector for DTA."""

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        self._metrics: dict[str, Any] = {}
        self._start_time = time.time()

        # Initialize core metrics
        self._init_core_metrics()

    def _init_core_metrics(self):
        """Initialize core DTA metrics."""
        if not PROMETHEUS_AVAILABLE:
            return

        # Request metrics
        self._metrics["requests_total"] = Counter(
            "dta_requests_total",
            "Total number of DTA requests processed",
            ["component", "status"],
            registry=self.registry,
        )

        self._metrics["request_duration"] = Histogram(
            "dta_request_duration_seconds",
            "Time spent processing DTA requests",
            ["component"],
            registry=self.registry,
        )

        # Error metrics
        self._metrics["errors_total"] = Counter(
            "dta_errors_total",
            "Total number of errors encountered",
            ["component", "error_type"],
            registry=self.registry,
        )

        # System metrics
        self._metrics["active_requests"] = Gauge(
            "dta_active_requests",
            "Number of currently active requests",
            ["component"],
            registry=self.registry,
        )

        # Thinking process metrics
        self._metrics["thinking_depth"] = Histogram(
            "dta_thinking_depth_steps",
            "Number of thinking steps in reasoning process",
            ["component"],
            registry=self.registry,
        )

        self._metrics["confidence_score"] = Histogram(
            "dta_confidence_score",
            "Confidence scores from thinking process",
            ["component"],
            registry=self.registry,
        )

        # Cache metrics
        self._metrics["cache_hits"] = Counter(
            "dta_cache_hits_total",
            "Number of cache hits",
            ["cache_type"],
            registry=self.registry,
        )

        self._metrics["cache_misses"] = Counter(
            "dta_cache_misses_total",
            "Number of cache misses",
            ["cache_type"],
            registry=self.registry,
        )

    def inc_requests(self, component: str, status: str = "success"):
        """Increment request counter."""
        if "requests_total" in self._metrics:
            self._metrics["requests_total"].labels(
                component=component, status=status
            ).inc()

    def observe_request_duration(self, component: str, duration: float):
        """Observe request duration."""
        if "request_duration" in self._metrics:
            self._metrics["request_duration"].labels(component=component).observe(
                duration
            )

    def inc_errors(self, component: str, error_type: str):
        """Increment error counter."""
        if "errors_total" in self._metrics:
            self._metrics["errors_total"].labels(
                component=component, error_type=error_type
            ).inc()

    def set_active_requests(self, component: str, count: int):
        """Set number of active requests."""
        if "active_requests" in self._metrics:
            self._metrics["active_requests"].labels(component=component).set(count)

    def observe_thinking_depth(self, component: str, steps: int):
        """Observe thinking process depth."""
        if "thinking_depth" in self._metrics:
            self._metrics["thinking_depth"].labels(component=component).observe(steps)

    def observe_confidence(self, component: str, confidence: float):
        """Observe confidence score."""
        if "confidence_score" in self._metrics:
            self._metrics["confidence_score"].labels(component=component).observe(
                confidence
            )

    def inc_cache_hits(self, cache_type: str):
        """Increment cache hits."""
        if "cache_hits" in self._metrics:
            self._metrics["cache_hits"].labels(cache_type=cache_type).inc()

    def inc_cache_misses(self, cache_type: str):
        """Increment cache misses."""
        if "cache_misses" in self._metrics:
            self._metrics["cache_misses"].labels(cache_type=cache_type).inc()

    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self._start_time


class DTAMonitoring:
    """
    Comprehensive monitoring and observability for DTA 2.0.

    Provides metrics collection, structured logging, health checks,
    alerting, and analytics for production DTA deployments.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)

        # Initialize components
        self.logger = DTAStructuredLogger(
            "monitoring", self.config.get("log_level", "INFO")
        )

        self.metrics = DTAMetricsCollector()
        self.alerts: list[DTAAlert] = []
        self.alert_thresholds = self.config.get(
            "alert_thresholds",
            {
                "error_rate": 0.05,  # 5% error rate threshold
                "response_time_ms": 1000,  # 1 second response time threshold
                "memory_usage_mb": 1000,  # 1GB memory usage threshold
            },
        )

        # Component health tracking
        self._component_health: dict[str, str] = {}
        self._health_lock = threading.RLock()

        # Analytics tracking
        self._analytics_buffer: list[dict[str, Any]] = []
        self._analytics_lock = threading.RLock()

        # Start metrics server if configured
        if self.config.get("metrics_port") and PROMETHEUS_AVAILABLE:
            self._start_metrics_server()

        self.logger.info(
            "DTA Monitoring initialized",
            enabled=self.enabled,
            prometheus_available=PROMETHEUS_AVAILABLE,
        )

    def _start_metrics_server(self):
        """Start Prometheus metrics HTTP server."""
        try:
            port = self.config["metrics_port"]
            start_http_server(port, registry=self.metrics.registry)
            self.logger.info(f"Metrics server started on port {port}")
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")

    @asynccontextmanager
    async def track_request(self, component: str, operation: str):
        """Context manager to track request metrics and timing."""
        start_time = time.time()
        request_id = f"{component}-{operation}-{int(start_time * 1000) % 10000}"

        # Increment active requests
        self.metrics.set_active_requests(component, 1)

        self.logger.debug(
            f"Request started: {request_id}",
            component=component,
            operation=operation,
            request_id=request_id,
        )

        try:
            yield request_id

            # Track successful completion
            duration = time.time() - start_time
            self.metrics.inc_requests(component, "success")
            self.metrics.observe_request_duration(component, duration)

            self.logger.debug(
                f"Request completed: {request_id}",
                component=component,
                operation=operation,
                request_id=request_id,
                duration_ms=duration * 1000,
            )

        except Exception as e:
            # Track error
            duration = time.time() - start_time
            error_type = type(e).__name__
            self.metrics.inc_requests(component, "error")
            self.metrics.inc_errors(component, error_type)
            self.metrics.observe_request_duration(component, duration)

            self.logger.error(
                f"Request failed: {request_id}",
                component=component,
                operation=operation,
                request_id=request_id,
                error_type=error_type,
                error_message=str(e),
                duration_ms=duration * 1000,
            )

            # Check if error rate threshold is exceeded
            await self._check_error_rate_alert(component)

            raise

        finally:
            # Decrement active requests
            self.metrics.set_active_requests(component, 0)

    async def record_thinking_process(
        self,
        component: str,
        thinking_steps: int,
        confidence: float,
        reasoning_quality: str | None = None,
    ):
        """Record metrics from thinking process."""
        if not self.enabled:
            return

        self.metrics.observe_thinking_depth(component, thinking_steps)
        self.metrics.observe_confidence(component, confidence)

        # Log detailed thinking analytics
        analytics_data = {
            "event_type": "thinking_process",
            "component": component,
            "thinking_steps": thinking_steps,
            "confidence": confidence,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        if reasoning_quality:
            analytics_data["reasoning_quality"] = reasoning_quality

        await self._record_analytics(analytics_data)

        self.logger.info(
            "Thinking process recorded",
            component=component,
            thinking_steps=thinking_steps,
            confidence=confidence,
        )

    async def record_cache_operation(
        self,
        cache_type: str,
        operation: str,
        hit: bool,
        latency_ms: float | None = None,
    ):
        """Record cache operation metrics."""
        if not self.enabled:
            return

        if hit:
            self.metrics.inc_cache_hits(cache_type)
        else:
            self.metrics.inc_cache_misses(cache_type)

        analytics_data = {
            "event_type": "cache_operation",
            "cache_type": cache_type,
            "operation": operation,
            "hit": hit,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        if latency_ms is not None:
            analytics_data["latency_ms"] = latency_ms

        await self._record_analytics(analytics_data)

    async def set_component_health(
        self, component: str, status: str, details: dict[str, Any] | None = None
    ):
        """Update component health status."""
        with self._health_lock:
            self._component_health[component] = status

        self.logger.info(
            f"Component health updated: {component} -> {status}",
            component=component,
            status=status,
            details=details or {},
        )

        # Check for health-based alerts
        if status in ["unhealthy", "critical", "error"]:
            await self._create_alert(
                AlertLevel.WARNING if status == "unhealthy" else AlertLevel.ERROR,
                component,
                f"Component {component} status: {status}",
                {"status": status, "details": details},
            )

    async def get_health_status(self) -> DTAHealthStatus:
        """Get comprehensive system health status."""
        current_time = datetime.now(UTC)

        with self._health_lock:
            component_statuses = self._component_health.copy()

        # Calculate overall status
        if not component_statuses:
            overall_status = "unknown"
        elif any(
            status in ["critical", "error"] for status in component_statuses.values()
        ):
            overall_status = "critical"
        elif any(status == "unhealthy" for status in component_statuses.values()):
            overall_status = "degraded"
        elif all(status == "healthy" for status in component_statuses.values()):
            overall_status = "healthy"
        else:
            overall_status = "unknown"

        # Get active alerts
        active_alerts = [alert for alert in self.alerts if not alert.resolved]

        # Calculate basic error rate and response time metrics
        error_rate = 0.0  # Will be enhanced when metrics collection is expanded
        avg_response_time = 100.0  # Default response time in ms

        return DTAHealthStatus(
            overall_status=overall_status,
            component_statuses=component_statuses,
            last_check=current_time,
            uptime_seconds=self.metrics.get_uptime(),
            error_rate=error_rate,
            avg_response_time_ms=avg_response_time,
            active_alerts=active_alerts,
        )

    async def _create_alert(
        self, level: AlertLevel, component: str, message: str, metrics: dict[str, Any]
    ):
        """Create and track a new alert."""
        alert = DTAAlert(
            timestamp=datetime.now(UTC),
            level=level,
            component=component,
            message=message,
            metrics=metrics,
        )

        self.alerts.append(alert)

        # Log alert
        self.logger.warning(
            f"Alert created: {message}",
            level=level.value,
            component=component,
            metrics=metrics,
        )

        # Keep only recent alerts (last 1000)
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]

    async def _check_error_rate_alert(self, component: str):
        """Check if error rate threshold is exceeded."""
        # Get recent alerts for this component
        recent_alerts = [
            alert
            for alert in self.alerts[-100:]  # Last 100 alerts
            if alert.component == component
            and alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]
        ]

        # Calculate error rate threshold check
        error_threshold = 0.1  # 10% error rate threshold
        if len(recent_alerts) > 10:  # Only check if we have enough data points
            error_rate = len(recent_alerts) / 100  # Rate over last 100 events
            if error_rate > error_threshold:
                await self._create_alert(
                    AlertLevel.ERROR,
                    component,
                    f"High error rate detected: {error_rate:.2%}",
                    {"error_rate": error_rate, "recent_errors": len(recent_alerts)},
                )

    async def _record_analytics(self, data: dict[str, Any]):
        """Record analytics data for later processing."""
        with self._analytics_lock:
            self._analytics_buffer.append(data)

            # Keep buffer manageable
            if len(self._analytics_buffer) > 10000:
                self._analytics_buffer = self._analytics_buffer[-5000:]

    async def get_analytics_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get analytics summary for specified time period."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

        with self._analytics_lock:
            recent_events = [
                event
                for event in self._analytics_buffer
                if datetime.fromisoformat(event["timestamp"]) > cutoff_time
            ]

        # Calculate summary statistics
        thinking_events = [
            e for e in recent_events if e.get("event_type") == "thinking_process"
        ]
        cache_events = [
            e for e in recent_events if e.get("event_type") == "cache_operation"
        ]

        summary = {
            "period_hours": hours,
            "total_events": len(recent_events),
            "thinking_processes": len(thinking_events),
            "cache_operations": len(cache_events),
            "avg_confidence": sum(e.get("confidence", 0) for e in thinking_events)
            / max(len(thinking_events), 1),
            "avg_thinking_steps": sum(
                e.get("thinking_steps", 0) for e in thinking_events
            )
            / max(len(thinking_events), 1),
            "cache_hit_rate": sum(1 for e in cache_events if e.get("hit", False))
            / max(len(cache_events), 1),
        }

        return summary

    async def shutdown(self):
        """Gracefully shutdown monitoring."""
        self.logger.info("DTA Monitoring shutting down")

        # Final analytics export could be added here
        with self._analytics_lock:
            final_event_count = len(self._analytics_buffer)

        self.logger.info(
            "DTA Monitoring shutdown complete",
            final_analytics_events=final_event_count,
            total_alerts=len(self.alerts),
        )


# Utility functions for easy monitoring setup
def create_monitoring(config: dict[str, Any] | None = None) -> DTAMonitoring:
    """Create a DTAMonitoring instance with default configuration."""
    default_config = {
        "enabled": True,
        "log_level": "INFO",
        "metrics_port": None,  # Set to enable Prometheus metrics server
        "alert_thresholds": {
            "error_rate": 0.05,
            "response_time_ms": 1000,
            "memory_usage_mb": 1000,
        },
    }

    if config:
        if hasattr(config, "__dict__"):
            # Handle dataclass or object - convert to dict
            if hasattr(config, "to_dict"):
                config_dict = config.to_dict()
            else:
                config_dict = {
                    k: v for k, v in config.__dict__.items() if not k.startswith("_")
                }
            default_config.update(config_dict)
        else:
            # Handle dict
            default_config.update(config)

    return DTAMonitoring(default_config)


# Example usage and testing
async def example_monitoring_usage():
    """Example of how to use DTA monitoring."""

    # Create monitoring instance
    monitoring = create_monitoring(
        {
            "log_level": "DEBUG",
            "metrics_port": 8000,  # Enable Prometheus metrics
        }
    )

    # Track a request
    async with monitoring.track_request("preprocessor", "analyze_intent"):
        # Simulate some work
        await asyncio.sleep(0.1)

        # Record thinking process
        await monitoring.record_thinking_process(
            "preprocessor", thinking_steps=5, confidence=0.85, reasoning_quality="high"
        )

        # Record cache operation
        await monitoring.record_cache_operation(
            "response_cache", "get", hit=True, latency_ms=5.2
        )

    # Update component health
    await monitoring.set_component_health("preprocessor", "healthy")
    await monitoring.set_component_health("cache", "healthy")

    # Get health status
    health = await monitoring.get_health_status()
    print(f"System health: {health.overall_status}")

    # Get analytics
    analytics = await monitoring.get_analytics_summary(hours=1)
    print(f"Analytics: {analytics}")

    # Shutdown
    await monitoring.shutdown()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_monitoring_usage())
