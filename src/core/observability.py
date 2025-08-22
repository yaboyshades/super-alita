"""
Observability Layer for Super Alita REUG v9.0

Provides structured event sourcing, correlation tracking, and tracing infrastructure
for enhanced debugging and monitoring of the cognitive execution flow.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class ObservabilityLevel(Enum):
    """Observability detail levels"""

    MINIMAL = auto()  # Essential events only
    STANDARD = auto()  # Core execution flow
    DETAILED = auto()  # Detailed state transitions
    VERBOSE = auto()  # All events including internal operations


@dataclass
class CorrelationContext:
    """Correlation context for tracking related events"""

    correlation_id: str
    session_id: str | None = None
    turn_id: str | None = None
    user_id: str | None = None
    parent_correlation_id: str | None = None
    trace_depth: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Metadata
    tags: dict[str, str] = field(default_factory=dict)
    baggage: dict[str, Any] = field(default_factory=dict)

    def derive_child(self, operation: str) -> "CorrelationContext":
        """Create a child correlation context for sub-operations"""
        return CorrelationContext(
            correlation_id=f"{self.correlation_id}.{operation}_{uuid.uuid4().hex[:8]}",
            session_id=self.session_id,
            turn_id=self.turn_id,
            user_id=self.user_id,
            parent_correlation_id=self.correlation_id,
            trace_depth=self.trace_depth + 1,
            tags=self.tags.copy(),
            baggage=self.baggage.copy(),
        )


@dataclass
class StructuredEvent:
    """Structured event for event sourcing"""

    # Core identification
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    event_type: str = ""
    event_category: str = "general"  # cognitive, system, plugin, error, etc.

    # Correlation tracking
    correlation_id: str = ""
    session_id: str | None = None
    turn_id: str | None = None

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float | None = None

    # Source tracking
    source_component: str = ""
    source_plugin: str | None = None
    source_state: str | None = None

    # Event data
    payload: dict[str, Any] = field(default_factory=dict)

    # Status and errors
    success: bool = True
    error_type: str | None = None
    error_message: str | None = None
    error_stack: str | None = None

    # Observability metadata
    observability_level: ObservabilityLevel = ObservabilityLevel.STANDARD
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_category": self.event_category,
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "source_component": self.source_component,
            "source_plugin": self.source_plugin,
            "source_state": self.source_state,
            "payload": self.payload,
            "success": self.success,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_stack": self.error_stack,
            "observability_level": self.observability_level.name,
            "tags": self.tags,
        }


class ObservabilityManager:
    """
    Central observability manager for structured event sourcing and tracing

    Features:
    - Correlation ID propagation
    - Structured event sourcing
    - Execution tracing
    - Performance monitoring
    - Error tracking
    """

    def __init__(self, level: ObservabilityLevel = ObservabilityLevel.STANDARD):
        self.level = level
        self.logger = logging.getLogger(__name__)

        # Event storage
        self.events: list[StructuredEvent] = []
        self.max_events = 10000  # Circular buffer

        # Active correlations
        self.active_correlations: dict[str, CorrelationContext] = {}

        # Event handlers
        self.event_handlers: list[callable] = []

        # Performance tracking
        self.operation_times: dict[str, list[float]] = {}

        self.logger.info(f"Observability manager initialized with level: {level.name}")

    def create_correlation_context(
        self,
        operation: str,
        session_id: str | None = None,
        turn_id: str | None = None,
        user_id: str | None = None,
        parent_correlation_id: str | None = None,
    ) -> CorrelationContext:
        """Create a new correlation context"""
        correlation_id = f"{operation}_{uuid.uuid4().hex[:12]}"

        context = CorrelationContext(
            correlation_id=correlation_id,
            session_id=session_id,
            turn_id=turn_id,
            user_id=user_id,
            parent_correlation_id=parent_correlation_id,
        )

        self.active_correlations[correlation_id] = context

        # Emit correlation started event
        self.emit_event(
            event_type="correlation_started",
            event_category="observability",
            correlation_id=correlation_id,
            source_component="observability_manager",
            payload={
                "operation": operation,
                "trace_depth": context.trace_depth,
                "parent_correlation_id": parent_correlation_id,
            },
            observability_level=ObservabilityLevel.DETAILED,
        )

        return context

    @asynccontextmanager
    async def trace_operation(
        self,
        operation: str,
        correlation_context: CorrelationContext | None = None,
        **metadata,
    ) -> AsyncIterator[CorrelationContext]:
        """Context manager for tracing operations with automatic timing"""
        if correlation_context is None:
            correlation_context = self.create_correlation_context(operation)
        else:
            correlation_context = correlation_context.derive_child(operation)

        start_time = datetime.now(UTC)

        # Emit operation started event
        self.emit_event(
            event_type="operation_started",
            event_category="tracing",
            correlation_id=correlation_context.correlation_id,
            source_component="trace_operation",
            payload={
                "operation": operation,
                "metadata": metadata,
                "trace_depth": correlation_context.trace_depth,
            },
            observability_level=ObservabilityLevel.DETAILED,
        )

        try:
            yield correlation_context

            # Success case
            end_time = datetime.now(UTC)
            duration_ms = (end_time - start_time).total_seconds() * 1000

            self.emit_event(
                event_type="operation_completed",
                event_category="tracing",
                correlation_id=correlation_context.correlation_id,
                source_component="trace_operation",
                duration_ms=duration_ms,
                payload={
                    "operation": operation,
                    "duration_ms": duration_ms,
                    "success": True,
                },
                observability_level=ObservabilityLevel.DETAILED,
            )

            # Track performance
            if operation not in self.operation_times:
                self.operation_times[operation] = []
            self.operation_times[operation].append(duration_ms)

        except Exception as e:
            # Error case
            end_time = datetime.now(UTC)
            duration_ms = (end_time - start_time).total_seconds() * 1000

            self.emit_event(
                event_type="operation_failed",
                event_category="error",
                correlation_id=correlation_context.correlation_id,
                source_component="trace_operation",
                duration_ms=duration_ms,
                success=False,
                error_type=type(e).__name__,
                error_message=str(e),
                payload={
                    "operation": operation,
                    "duration_ms": duration_ms,
                    "error_details": {"type": type(e).__name__, "message": str(e)},
                },
                observability_level=ObservabilityLevel.STANDARD,
            )
            raise

        finally:
            # Cleanup active correlation
            if correlation_context.correlation_id in self.active_correlations:
                del self.active_correlations[correlation_context.correlation_id]

    def emit_event(
        self,
        event_type: str,
        event_category: str = "general",
        correlation_id: str = "",
        session_id: str | None = None,
        turn_id: str | None = None,
        source_component: str = "",
        source_plugin: str | None = None,
        source_state: str | None = None,
        payload: dict[str, Any] | None = None,
        duration_ms: float | None = None,
        success: bool = True,
        error_type: str | None = None,
        error_message: str | None = None,
        error_stack: str | None = None,
        observability_level: ObservabilityLevel = ObservabilityLevel.STANDARD,
        tags: dict[str, str] | None = None,
    ) -> StructuredEvent:
        """Emit a structured event"""

        # Skip if below current observability level
        if observability_level.value > self.level.value:
            return None

        event = StructuredEvent(
            event_type=event_type,
            event_category=event_category,
            correlation_id=correlation_id,
            session_id=session_id,
            turn_id=turn_id,
            source_component=source_component,
            source_plugin=source_plugin,
            source_state=source_state,
            payload=payload or {},
            duration_ms=duration_ms,
            success=success,
            error_type=error_type,
            error_message=error_message,
            error_stack=error_stack,
            observability_level=observability_level,
            tags=tags or {},
        )

        # Store event (circular buffer)
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)

        # Notify handlers
        for handler in self.event_handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.warning(f"Event handler failed: {e}")

        return event

    def add_event_handler(self, handler: callable):
        """Add an event handler for real-time processing"""
        self.event_handlers.append(handler)

    def get_events(
        self,
        correlation_id: str | None = None,
        session_id: str | None = None,
        event_category: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[StructuredEvent]:
        """Query stored events with filtering"""
        filtered_events = self.events

        if correlation_id:
            filtered_events = [
                e for e in filtered_events if e.correlation_id == correlation_id
            ]

        if session_id:
            filtered_events = [e for e in filtered_events if e.session_id == session_id]

        if event_category:
            filtered_events = [
                e for e in filtered_events if e.event_category == event_category
            ]

        if since:
            filtered_events = [e for e in filtered_events if e.timestamp >= since]

        # Sort by timestamp (most recent first)
        filtered_events.sort(key=lambda e: e.timestamp, reverse=True)

        if limit:
            filtered_events = filtered_events[:limit]

        return filtered_events

    def get_performance_stats(self, operation: str | None = None) -> dict[str, Any]:
        """Get performance statistics for operations"""
        if operation:
            times = self.operation_times.get(operation, [])
            if not times:
                return {"operation": operation, "count": 0}

            return {
                "operation": operation,
                "count": len(times),
                "avg_duration_ms": sum(times) / len(times),
                "min_duration_ms": min(times),
                "max_duration_ms": max(times),
                "total_duration_ms": sum(times),
            }
        else:
            return {
                "total_operations": len(self.operation_times),
                "operations": [
                    self.get_performance_stats(op) for op in self.operation_times
                ],
            }

    def get_active_traces(self) -> dict[str, CorrelationContext]:
        """Get currently active correlation contexts"""
        return self.active_correlations.copy()

    def export_events(self, format_type: str = "json") -> str | dict:
        """Export events for external analysis"""
        if format_type == "json":
            return [event.to_dict() for event in self.events]
        elif format_type == "csv":
            # CSV export implementation
            import csv
            import io

            output = io.StringIO()

            if not self.events:
                return "event_id,event_type,event_category,correlation_id,timestamp,source_component,success\n"

            fieldnames = [
                "event_id",
                "event_type",
                "event_category",
                "correlation_id",
                "session_id",
                "turn_id",
                "timestamp",
                "duration_ms",
                "source_component",
                "source_plugin",
                "source_state",
                "success",
                "error_type",
                "error_message",
                "observability_level",
            ]

            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            for event in self.events:
                row = {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "event_category": event.event_category,
                    "correlation_id": event.correlation_id,
                    "session_id": event.session_id or "",
                    "turn_id": event.turn_id or "",
                    "timestamp": event.timestamp.isoformat(),
                    "duration_ms": event.duration_ms or "",
                    "source_component": event.source_component,
                    "source_plugin": event.source_plugin or "",
                    "source_state": event.source_state or "",
                    "success": event.success,
                    "error_type": event.error_type or "",
                    "error_message": event.error_message or "",
                    "observability_level": event.observability_level.name,
                }
                writer.writerow(row)

            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")


# Global observability manager instance
observability = ObservabilityManager()


def get_observability_manager() -> ObservabilityManager:
    """Get the global observability manager"""
    return observability


def set_observability_level(level: ObservabilityLevel) -> None:
    """Set the global observability level"""
    observability.level = level
    observability.logger.info(f"Observability level set to: {level.name}")


# Decorators for easy tracing
def trace_async(operation_name: str | None = None):
    """Decorator for tracing async functions"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            async with observability.trace_operation(op_name):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def trace_sync(operation_name: str | None = None):
    """Decorator for tracing sync functions"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            # For sync functions, we'll create a simple timing mechanism
            start_time = datetime.now(UTC)

            try:
                result = func(*args, **kwargs)

                end_time = datetime.now(UTC)
                duration_ms = (end_time - start_time).total_seconds() * 1000

                observability.emit_event(
                    event_type="sync_operation_completed",
                    event_category="tracing",
                    source_component="trace_sync_decorator",
                    payload={
                        "operation": op_name,
                        "duration_ms": duration_ms,
                        "success": True,
                    },
                    duration_ms=duration_ms,
                    observability_level=ObservabilityLevel.DETAILED,
                )

                return result

            except Exception as e:
                end_time = datetime.now(UTC)
                duration_ms = (end_time - start_time).total_seconds() * 1000

                observability.emit_event(
                    event_type="sync_operation_failed",
                    event_category="error",
                    source_component="trace_sync_decorator",
                    payload={
                        "operation": op_name,
                        "duration_ms": duration_ms,
                        "error_details": {"type": type(e).__name__, "message": str(e)},
                    },
                    duration_ms=duration_ms,
                    success=False,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    observability_level=ObservabilityLevel.STANDARD,
                )

                raise

        return wrapper

    return decorator
