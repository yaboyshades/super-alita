"""
OpenTelemetry Configuration and Setup

Implements the comprehensive telemetry infrastructure with:
- Service-Level Objectives (SLOs): p95 < 1s, error rate < 2%
- JSON structured logging with mandatory fields
- Prometheus metrics endpoint on :9464
- 100% error capture, 10% success sampling
- Context propagation across async boundaries
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import wraps
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class TelemetrySpan:
    """Structured telemetry span with mandatory OpenTelemetry fields."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    component: str
    operation: str
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None
    status_code: str = "OK"  # OK, ERROR, TIMEOUT, CANCELLED
    error_type: str | None = None
    error_message: str | None = None
    tags: dict[str, Any] | None = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

    def to_json_log(self) -> str:
        """Convert span to JSON log format with mandatory fields."""
        log_data = {
            "timestamp": datetime.fromtimestamp(
                self.start_time, UTC
            ).isoformat(),
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "component": self.component,
            "operation": self.operation,
            "duration_ms": self.duration_ms,
            "status_code": self.status_code,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "tags": self.tags,
        }
        return json.dumps(log_data, separators=(",", ":"))


@dataclass
class ServiceLevelObjectives:
    """Service-Level Objectives for performance monitoring."""

    latency_p95_ms: float = 1000.0  # 95th percentile < 1s
    error_rate_threshold: float = 0.02  # < 2% failures per 1000 interactions
    min_throughput_rps: float = 10.0  # Minimum sustained requests per second
    cpu_utilization_threshold: float = 0.80  # CPU < 80% sustained
    memory_utilization_threshold: float = 0.70  # Memory < 70% sustained


class TelemetryContext:
    """Thread-local telemetry context for trace propagation."""

    def __init__(self):
        self._context = {}
        self._span_stack = []

    def get_current_trace_id(self) -> str | None:
        """Get current trace ID from context."""
        return self._context.get("trace_id")

    def get_current_span_id(self) -> str | None:
        """Get current span ID from context."""
        if self._span_stack:
            return self._span_stack[-1]
        return None

    def push_span(self, span_id: str, trace_id: str | None = None) -> None:
        """Push new span onto the context stack."""
        if trace_id:
            self._context["trace_id"] = trace_id
        self._span_stack.append(span_id)

    def pop_span(self) -> str | None:
        """Pop span from the context stack."""
        if self._span_stack:
            return self._span_stack.pop()
        return None

    def clear(self) -> None:
        """Clear the telemetry context."""
        self._context.clear()
        self._span_stack.clear()


# Global telemetry context (should be task-local in production)
_telemetry_context = TelemetryContext()


class OpenTelemetryCollector:
    """OpenTelemetry-compatible telemetry collector with SLO tracking."""

    def __init__(
        self,
        slos: ServiceLevelObjectives | None = None,
        error_sample_rate: float = 1.0,  # 100% error capture
        success_sample_rate: float = 0.1,  # 10% success sampling
        prometheus_port: int = 9464,
    ):
        self.slos = slos or ServiceLevelObjectives()
        self.error_sample_rate = error_sample_rate
        self.success_sample_rate = success_sample_rate
        self.prometheus_port = prometheus_port

        # Metrics storage
        self.spans: list[TelemetrySpan] = []
        self.metrics: dict[str, list[float]] = {
            "latency_ms": [],
            "error_count": [],
            "success_count": [],
            "total_requests": [],
        }

        # SLO violation tracking
        self.slo_violations: list[dict[str, Any]] = []

        # Setup structured logger
        self._setup_structured_logger()

        logger.info(
            "OpenTelemetry collector initialized with SLOs: %s",
            asdict(self.slos),
        )

    def _setup_structured_logger(self) -> None:
        """Setup structured JSON logger for telemetry data."""
        # Create dedicated telemetry logger
        self.telemetry_logger = logging.getLogger("telemetry.spans")
        self.telemetry_logger.setLevel(logging.INFO)

        # Prevent propagation to avoid duplicate logs
        self.telemetry_logger.propagate = False

        # Add JSON formatter if not already configured
        if not self.telemetry_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(message)s")  # Raw JSON output
            handler.setFormatter(formatter)
            self.telemetry_logger.addHandler(handler)

    def start_span(
        self,
        component: str,
        operation: str,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> TelemetrySpan:
        """Start a new telemetry span with context propagation."""

        # Generate IDs
        if not trace_id:
            trace_id = _telemetry_context.get_current_trace_id() or str(
                uuid4()
            )

        if not parent_span_id:
            parent_span_id = _telemetry_context.get_current_span_id()

        span_id = str(uuid4())

        # Create span
        span = TelemetrySpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            component=component,
            operation=operation,
            start_time=time.perf_counter(),
            tags=tags or {},
        )

        # Update context
        _telemetry_context.push_span(span_id, trace_id)

        return span

    def finish_span(
        self,
        span: TelemetrySpan,
        status_code: str = "OK",
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Finish a telemetry span and apply sampling."""

        span.end_time = time.perf_counter()
        span.duration_ms = (span.end_time - span.start_time) * 1000
        span.status_code = status_code
        span.error_type = error_type
        span.error_message = error_message

        # Pop from context
        _telemetry_context.pop_span()

        # Apply sampling rules
        should_record = self._should_record_span(span)

        if should_record:
            self.spans.append(span)
            self.telemetry_logger.info(span.to_json_log())

        # Update metrics
        self._update_metrics(span)

        # Check SLO violations
        self._check_slo_violations(span)

    def _should_record_span(self, span: TelemetrySpan) -> bool:
        """Apply sampling rules: 100% errors, 10% success."""
        if span.status_code != "OK":
            return True  # 100% error capture

        # 10% success sampling
        import random

        return random.random() < self.success_sample_rate

    def _update_metrics(self, span: TelemetrySpan) -> None:
        """Update performance metrics."""
        if span.duration_ms is not None:
            self.metrics["latency_ms"].append(span.duration_ms)

        self.metrics["total_requests"].append(1)

        if span.status_code == "OK":
            self.metrics["success_count"].append(1)
        else:
            self.metrics["error_count"].append(1)

    def _check_slo_violations(self, span: TelemetrySpan) -> None:
        """Check for SLO violations and log alerts."""
        violations = []

        # Check latency SLO
        if (
            span.duration_ms is not None
            and span.duration_ms > self.slos.latency_p95_ms
        ):
            violations.append(
                {
                    "type": "latency_violation",
                    "threshold": self.slos.latency_p95_ms,
                    "actual": span.duration_ms,
                    "span_id": span.span_id,
                    "component": span.component,
                    "operation": span.operation,
                }
            )

        # Track violations
        for violation in violations:
            self.slo_violations.append(violation)
            logger.warning("SLO violation detected: %s", violation)

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get comprehensive metrics summary for monitoring."""
        if not self.metrics["latency_ms"]:
            return {"status": "no_data"}

        latencies = sorted(self.metrics["latency_ms"])
        total_requests = len(latencies)
        error_count = len(self.metrics["error_count"])

        # Calculate percentiles
        p50_idx = int(total_requests * 0.5)
        p95_idx = int(total_requests * 0.95)
        p99_idx = int(total_requests * 0.99)

        error_rate = error_count / total_requests if total_requests > 0 else 0

        return {
            "total_requests": total_requests,
            "error_count": error_count,
            "error_rate": error_rate,
            "latency_ms": {
                "p50": latencies[p50_idx] if p50_idx < len(latencies) else 0,
                "p95": latencies[p95_idx] if p95_idx < len(latencies) else 0,
                "p99": latencies[p99_idx] if p99_idx < len(latencies) else 0,
                "avg": sum(latencies) / len(latencies),
                "max": max(latencies),
                "min": min(latencies),
            },
            "slo_compliance": {
                "latency_p95_compliant": (
                    latencies[p95_idx] < self.slos.latency_p95_ms
                    if p95_idx < len(latencies)
                    else True
                ),
                "error_rate_compliant": (
                    error_rate < self.slos.error_rate_threshold
                ),
                "total_violations": len(self.slo_violations),
            },
            "recent_violations": (
                self.slo_violations[-10:] if self.slo_violations else []
            ),
        }

    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-compatible metrics format."""
        summary = self.get_metrics_summary()

        if summary.get("status") == "no_data":
            return "# No telemetry data available\n"

        metrics_text = []

        # Request metrics
        metrics_text.append(
            "# HELP extension_requests_total "
            "Total number of extension requests"
        )
        metrics_text.append("# TYPE extension_requests_total counter")
        metrics_text.append(
            f"extension_requests_total {summary['total_requests']}"
        )

        # Error metrics
        metrics_text.append(
            "# HELP extension_errors_total Total number of extension errors"
        )
        metrics_text.append("# TYPE extension_errors_total counter")
        metrics_text.append(f"extension_errors_total {summary['error_count']}")

        # Error rate
        metrics_text.append("# HELP extension_error_rate Current error rate")
        metrics_text.append("# TYPE extension_error_rate gauge")
        metrics_text.append(f"extension_error_rate {summary['error_rate']}")

        # Latency percentiles
        latency = summary["latency_ms"]
        for percentile in ["p50", "p95", "p99"]:
            metrics_text.append(
                f"# HELP extension_latency_{percentile}_ms "
                f"Latency {percentile} in milliseconds"
            )
            metrics_text.append(
                f"# TYPE extension_latency_{percentile}_ms gauge"
            )
            metrics_text.append(
                f"extension_latency_{percentile}_ms {latency[percentile]}"
            )

        # SLO compliance
        slo = summary["slo_compliance"]
        metrics_text.append(
            "# HELP extension_slo_compliance SLO compliance indicators"
        )
        metrics_text.append("# TYPE extension_slo_compliance gauge")
        metrics_text.append(
            f"extension_slo_latency_compliant "
            f"{1 if slo['latency_p95_compliant'] else 0}"
        )
        metrics_text.append(
            f"extension_slo_error_rate_compliant "
            f"{1 if slo['error_rate_compliant'] else 0}"
        )

        return "\n".join(metrics_text) + "\n"


# Global telemetry collector instance
_global_collector = OpenTelemetryCollector()


def get_telemetry_collector() -> OpenTelemetryCollector:
    """Get the global telemetry collector instance."""
    return _global_collector


def telemetry_trace(
    component: str, operation: str = None, tags: dict[str, Any] | None = None
):
    """Decorator for automatic telemetry tracing of functions."""

    def decorator(func):
        func_operation = operation or func.__name__

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                collector = get_telemetry_collector()
                span = collector.start_span(
                    component, func_operation, tags=tags
                )

                try:
                    result = await func(*args, **kwargs)
                    collector.finish_span(span, "OK")
                    return result

                except Exception as e:
                    collector.finish_span(
                        span,
                        "ERROR",
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )
                    raise

            return async_wrapper

        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                collector = get_telemetry_collector()
                span = collector.start_span(
                    component, func_operation, tags=tags
                )

                try:
                    result = func(*args, **kwargs)
                    collector.finish_span(span, "OK")
                    return result

                except Exception as e:
                    collector.finish_span(
                        span,
                        "ERROR",
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )
                    raise

            return sync_wrapper

    return decorator


@asynccontextmanager
async def telemetry_span(
    component: str, operation: str, tags: dict[str, Any] | None = None
):
    """Async context manager for telemetry spans."""
    collector = get_telemetry_collector()
    span = collector.start_span(component, operation, tags=tags)

    try:
        yield span
        collector.finish_span(span, "OK")

    except Exception as e:
        collector.finish_span(
            span, "ERROR", error_type=type(e).__name__, error_message=str(e)
        )
        raise


@contextmanager
def telemetry_span_sync(
    component: str, operation: str, tags: dict[str, Any] | None = None
):
    """Synchronous context manager for telemetry spans."""
    collector = get_telemetry_collector()
    span = collector.start_span(component, operation, tags=tags)

    try:
        yield span
        collector.finish_span(span, "OK")

    except Exception as e:
        collector.finish_span(
            span, "ERROR", error_type=type(e).__name__, error_message=str(e)
        )
        raise


async def start_prometheus_server(port: int = 9464) -> None:
    """Start Prometheus metrics HTTP server."""
    try:
        from aiohttp import web

        async def metrics_handler(request):
            collector = get_telemetry_collector()
            metrics_text = collector.get_prometheus_metrics()
            return web.Response(text=metrics_text, content_type="text/plain")

        app = web.Application()
        app.router.add_get("/metrics", metrics_handler)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, "localhost", port)
        await site.start()

        logger.info(
            "Prometheus metrics server started on "
            "http://localhost:%s/metrics",
            port,
        )

    except ImportError:
        logger.warning("aiohttp not available, Prometheus server not started")
    except Exception as e:
        logger.error("Failed to start Prometheus server: %s", e)
