"""
Telemetry Bridge

Collects real-time telemetry data from various system components including
host API calls, WASM operations, and LSP communications.
"""

import asyncio
import contextlib
import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TelemetryEvent:
    """Individual telemetry event data."""

    source: str  # host_api, wasm, lsp, extension
    event_type: str  # call, response, error, metric
    timestamp: datetime
    data: dict[str, Any]
    duration_ms: float | None = None
    success: bool = True
    tags: dict[str, str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class HostAPICall:
    """Host API call telemetry data."""

    function_name: str
    parameters: dict[str, Any]
    start_time: datetime
    end_time: datetime | None = None
    result: Any = None
    error: str | None = None
    constitutional_impact: str | None = None


@dataclass
class WASMOperation:
    """WASM operation telemetry data."""

    component_name: str
    operation_type: str  # predict, analyze, validate
    input_size_bytes: int
    start_time: datetime
    end_time: datetime | None = None
    memory_usage_bytes: int | None = None
    execution_result: Any = None
    prediction_accuracy: float | None = None


@dataclass
class LSPMessage:
    """LSP message telemetry data."""

    message_type: str  # request, response, notification
    method: str
    message_size_bytes: int
    timestamp: datetime
    latency_ms: float | None = None
    diagnostic_count: int | None = None
    constitutional_relevance: str | None = None


class TelemetryBridge:
    """
    Telemetry bridge for collecting real-time performance and constitutional
    compliance data from all system components.

    Implements Article I: Library-First through standard telemetry patterns.
    Implements Article III: Simplicity through focused data collection.
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        flush_interval_seconds: int = 30,
        enable_constitutional_tracking: bool = True,
    ):
        self.buffer_size = buffer_size
        self.flush_interval_seconds = flush_interval_seconds
        self.enable_constitutional_tracking = enable_constitutional_tracking

        # Event storage
        self.events: deque = deque(maxlen=buffer_size)
        self.host_api_calls: deque = deque(maxlen=1000)
        self.wasm_operations: deque = deque(maxlen=1000)
        self.lsp_messages: deque = deque(maxlen=1000)

        # Event handlers and filters
        self.event_handlers: list[Callable] = []
        self.filters: list[Callable] = []

        # Background processing
        self._active = False
        self._flush_task: asyncio.Task | None = None
        self._lock = threading.Lock()

        # Constitutional tracking
        self.constitutional_events: deque = deque(maxlen=500)

        logger.info(
            "Telemetry Bridge initialized with constitutional tracking enabled"
        )

    async def start(self) -> None:
        """Start telemetry collection."""
        if self._active:
            logger.warning("Telemetry Bridge already active")
            return

        self._active = True
        self._flush_task = asyncio.create_task(self._background_flush_loop())
        logger.info("Telemetry Bridge started")

    async def stop(self) -> None:
        """Stop telemetry collection."""
        if not self._active:
            return

        self._active = False
        if self._flush_task:
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task

        # Final flush
        await self._flush_events()
        logger.info("Telemetry Bridge stopped")

    def track_host_api_call(
        self,
        function_name: str,
        parameters: dict[str, Any],
        constitutional_impact: str | None = None,
    ) -> HostAPICall:
        """Start tracking a host API call."""
        call = HostAPICall(
            function_name=function_name,
            parameters=parameters,
            start_time=datetime.now(UTC),
            constitutional_impact=constitutional_impact,
        )

        with self._lock:
            self.host_api_calls.append(call)

        # Create telemetry event
        self._add_event(
            source="host_api",
            event_type="call_start",
            data={
                "function_name": function_name,
                "parameter_count": len(parameters),
                "constitutional_impact": constitutional_impact,
            },
            tags={"function": function_name},
        )

        logger.debug(f"Started tracking host API call: {function_name}")
        return call

    def complete_host_api_call(
        self, call: HostAPICall, result: Any = None, error: str | None = None
    ) -> None:
        """Complete tracking of a host API call."""
        call.end_time = datetime.now(UTC)
        call.result = result
        call.error = error

        duration_ms = (call.end_time - call.start_time).total_seconds() * 1000

        # Create completion event
        self._add_event(
            source="host_api",
            event_type="call_complete",
            data={
                "function_name": call.function_name,
                "duration_ms": duration_ms,
                "success": error is None,
                "error": error,
                "constitutional_impact": call.constitutional_impact,
            },
            duration_ms=duration_ms,
            success=error is None,
            tags={"function": call.function_name},
        )

        # Track constitutional impact
        if self.enable_constitutional_tracking and call.constitutional_impact:
            self._track_constitutional_event(
                "host_api_call", call.constitutional_impact, duration_ms
            )

    def track_wasm_operation(
        self, component_name: str, operation_type: str, input_size_bytes: int
    ) -> WASMOperation:
        """Start tracking a WASM operation."""
        operation = WASMOperation(
            component_name=component_name,
            operation_type=operation_type,
            input_size_bytes=input_size_bytes,
            start_time=datetime.now(UTC),
        )

        with self._lock:
            self.wasm_operations.append(operation)

        # Create telemetry event
        self._add_event(
            source="wasm",
            event_type="operation_start",
            data={
                "component_name": component_name,
                "operation_type": operation_type,
                "input_size_bytes": input_size_bytes,
            },
            tags={"component": component_name, "operation": operation_type},
        )

        logger.debug(
            f"Started tracking WASM operation: {component_name}.{operation_type}"
        )
        return operation

    def complete_wasm_operation(
        self,
        operation: WASMOperation,
        result: Any = None,
        memory_usage_bytes: int | None = None,
        prediction_accuracy: float | None = None,
    ) -> None:
        """Complete tracking of a WASM operation."""
        operation.end_time = datetime.now(UTC)
        operation.execution_result = result
        operation.memory_usage_bytes = memory_usage_bytes
        operation.prediction_accuracy = prediction_accuracy

        duration_ms = (
            operation.end_time - operation.start_time
        ).total_seconds() * 1000

        # Create completion event
        self._add_event(
            source="wasm",
            event_type="operation_complete",
            data={
                "component_name": operation.component_name,
                "operation_type": operation.operation_type,
                "duration_ms": duration_ms,
                "memory_usage_bytes": memory_usage_bytes,
                "prediction_accuracy": prediction_accuracy,
                "input_size_bytes": operation.input_size_bytes,
            },
            duration_ms=duration_ms,
            tags={
                "component": operation.component_name,
                "operation": operation.operation_type,
            },
        )

    def track_lsp_message(
        self,
        message_type: str,
        method: str,
        message_size_bytes: int,
        constitutional_relevance: str | None = None,
    ) -> LSPMessage:
        """Track an LSP message."""
        message = LSPMessage(
            message_type=message_type,
            method=method,
            message_size_bytes=message_size_bytes,
            timestamp=datetime.now(UTC),
            constitutional_relevance=constitutional_relevance,
        )

        with self._lock:
            self.lsp_messages.append(message)

        # Create telemetry event
        self._add_event(
            source="lsp",
            event_type="message",
            data={
                "message_type": message_type,
                "method": method,
                "message_size_bytes": message_size_bytes,
                "constitutional_relevance": constitutional_relevance,
            },
            tags={"method": method, "type": message_type},
        )

        # Track constitutional relevance
        if self.enable_constitutional_tracking and constitutional_relevance:
            self._track_constitutional_event(
                "lsp_message", constitutional_relevance, 0
            )

        return message

    def update_lsp_message(
        self,
        message: LSPMessage,
        latency_ms: float | None = None,
        diagnostic_count: int | None = None,
    ) -> None:
        """Update LSP message with response data."""
        message.latency_ms = latency_ms
        message.diagnostic_count = diagnostic_count

        # Create update event
        self._add_event(
            source="lsp",
            event_type="message_update",
            data={
                "method": message.method,
                "latency_ms": latency_ms,
                "diagnostic_count": diagnostic_count,
                "constitutional_relevance": message.constitutional_relevance,
            },
            duration_ms=latency_ms,
            tags={"method": message.method},
        )

    def add_event_handler(
        self, handler: Callable[[TelemetryEvent], None]
    ) -> None:
        """Add an event handler."""
        self.event_handlers.append(handler)
        logger.info(f"Added telemetry event handler: {handler.__name__}")

    def add_filter(
        self, filter_func: Callable[[TelemetryEvent], bool]
    ) -> None:
        """Add an event filter."""
        self.filters.append(filter_func)
        logger.info(f"Added telemetry filter: {filter_func.__name__}")

    def get_telemetry_summary(self) -> dict[str, Any]:
        """Get current telemetry summary."""
        with self._lock:
            current_time = datetime.now(UTC)

            # Recent events (last 10 minutes)
            recent_cutoff = current_time.timestamp() - 600
            recent_events = [
                e
                for e in self.events
                if e.timestamp.timestamp() > recent_cutoff
            ]

            # Calculate statistics
            host_api_stats = self._calculate_host_api_stats()
            wasm_stats = self._calculate_wasm_stats()
            lsp_stats = self._calculate_lsp_stats()
            constitutional_stats = self._calculate_constitutional_stats()

        return {
            "timestamp": current_time.isoformat(),
            "total_events": len(self.events),
            "recent_events": len(recent_events),
            "event_rate_per_minute": len(recent_events)
            * 6,  # Extrapolate from 10 minutes
            "host_api_statistics": host_api_stats,
            "wasm_statistics": wasm_stats,
            "lsp_statistics": lsp_stats,
            "constitutional_statistics": constitutional_stats,
            "buffer_utilization": {
                "events": len(self.events) / self.buffer_size,
                "host_api_calls": len(self.host_api_calls) / 1000,
                "wasm_operations": len(self.wasm_operations) / 1000,
                "lsp_messages": len(self.lsp_messages) / 1000,
            },
        }

    def get_constitutional_compliance_data(self) -> dict[str, Any]:
        """Get constitutional compliance data from telemetry."""
        with self._lock:
            constitutional_events = list(self.constitutional_events)

        if not constitutional_events:
            return {
                "status": "no_data",
                "message": "No constitutional events tracked",
            }

        # Analyze constitutional impact patterns
        impact_categories = {}
        for event in constitutional_events:
            category = event.get("category", "unknown")
            if category not in impact_categories:
                impact_categories[category] = {"count": 0, "total_duration": 0}
            impact_categories[category]["count"] += 1
            impact_categories[category]["total_duration"] += event.get(
                "duration_ms", 0
            )

        return {
            "total_constitutional_events": len(constitutional_events),
            "impact_categories": impact_categories,
            "compliance_indicators": self._extract_compliance_indicators(
                constitutional_events
            ),
            "recent_violations": self._extract_recent_violations(
                constitutional_events
            ),
        }

    def _add_event(
        self,
        source: str,
        event_type: str,
        data: dict[str, Any],
        duration_ms: float | None = None,
        success: bool = True,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Add a telemetry event."""
        event = TelemetryEvent(
            source=source,
            event_type=event_type,
            timestamp=datetime.now(UTC),
            data=data,
            duration_ms=duration_ms,
            success=success,
            tags=tags or {},
        )

        # Apply filters
        for filter_func in self.filters:
            if not filter_func(event):
                return

        # Add to buffer
        with self._lock:
            self.events.append(event)

        # Notify handlers
        for handler in self.event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Telemetry handler error: {e}")

    def _track_constitutional_event(
        self, source: str, impact_description: str, duration_ms: float
    ) -> None:
        """Track constitutional compliance events."""
        constitutional_event = {
            "source": source,
            "impact": impact_description,
            "duration_ms": duration_ms,
            "timestamp": datetime.now(UTC).isoformat(),
            "category": self._categorize_constitutional_impact(
                impact_description
            ),
        }

        with self._lock:
            self.constitutional_events.append(constitutional_event)

    def _categorize_constitutional_impact(
        self, impact_description: str
    ) -> str:
        """Categorize constitutional impact for analysis."""
        impact_lower = impact_description.lower()

        if "library" in impact_lower or "dependency" in impact_lower:
            return "article_i_library_first"
        elif "test" in impact_lower or "validation" in impact_lower:
            return "article_ii_test_first"
        elif "complexity" in impact_lower or "simplicity" in impact_lower:
            return "article_iii_simplicity"
        elif "integration" in impact_lower or "interface" in impact_lower:
            return "article_iv_integration_first"
        elif "documentation" in impact_lower or "clarity" in impact_lower:
            return "article_v_clarity"
        elif "version" in impact_lower or "compatibility" in impact_lower:
            return "article_vi_versioning"
        else:
            return "general_constitutional"

    async def _background_flush_loop(self) -> None:
        """Background loop for flushing events."""
        while self._active:
            try:
                await asyncio.sleep(self.flush_interval_seconds)
                await self._flush_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Telemetry flush error: {e}")

    async def _flush_events(self) -> None:
        """Flush events to handlers/storage."""
        # For now, just log a summary. In production, this would
        # send to monitoring systems, databases, etc.
        summary = self.get_telemetry_summary()
        logger.info(
            f"Telemetry flush: {summary['total_events']} events, "
            f"rate: {summary['event_rate_per_minute']}/min"
        )

    def _calculate_host_api_stats(self) -> dict[str, Any]:
        """Calculate host API call statistics."""
        completed_calls = [
            call for call in self.host_api_calls if call.end_time
        ]
        if not completed_calls:
            return {"no_data": True}

        total_duration = sum(
            (call.end_time - call.start_time).total_seconds() * 1000
            for call in completed_calls
        )
        successful_calls = sum(
            1 for call in completed_calls if call.error is None
        )

        return {
            "total_calls": len(completed_calls),
            "success_rate": successful_calls / len(completed_calls),
            "average_duration_ms": total_duration / len(completed_calls),
            "constitutional_impact_calls": sum(
                1 for call in completed_calls if call.constitutional_impact
            ),
        }

    def _calculate_wasm_stats(self) -> dict[str, Any]:
        """Calculate WASM operation statistics."""
        completed_ops = [op for op in self.wasm_operations if op.end_time]
        if not completed_ops:
            return {"no_data": True}

        total_duration = sum(
            (op.end_time - op.start_time).total_seconds() * 1000
            for op in completed_ops
        )

        return {
            "total_operations": len(completed_ops),
            "average_duration_ms": total_duration / len(completed_ops),
            "average_memory_usage_bytes": sum(
                op.memory_usage_bytes or 0 for op in completed_ops
            )
            / len(completed_ops),
            "operations_by_type": self._group_by_field(
                completed_ops, "operation_type"
            ),
        }

    def _calculate_lsp_stats(self) -> dict[str, Any]:
        """Calculate LSP message statistics."""
        if not self.lsp_messages:
            return {"no_data": True}

        messages_with_latency = [
            msg for msg in self.lsp_messages if msg.latency_ms
        ]

        return {
            "total_messages": len(self.lsp_messages),
            "average_message_size_bytes": sum(
                msg.message_size_bytes for msg in self.lsp_messages
            )
            / len(self.lsp_messages),
            "average_latency_ms": (
                sum(msg.latency_ms for msg in messages_with_latency)
                / len(messages_with_latency)
                if messages_with_latency
                else 0
            ),
            "messages_by_method": self._group_by_field(
                self.lsp_messages, "method"
            ),
            "constitutional_relevant_messages": sum(
                1 for msg in self.lsp_messages if msg.constitutional_relevance
            ),
        }

    def _calculate_constitutional_stats(self) -> dict[str, Any]:
        """Calculate constitutional compliance statistics."""
        if not self.constitutional_events:
            return {"no_data": True}

        events = list(self.constitutional_events)
        category_counts = {}
        for event in events:
            category = event.get("category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1

        return {
            "total_constitutional_events": len(events),
            "events_by_category": category_counts,
            "recent_events_count": len(
                [
                    e
                    for e in events
                    if (
                        datetime.now(UTC)
                        - datetime.fromisoformat(
                            e["timestamp"].replace("Z", "+00:00")
                        )
                    ).total_seconds()
                    < 3600
                ]
            ),
        }

    def _group_by_field(self, items: list[Any], field: str) -> dict[str, int]:
        """Group items by a field value."""
        groups = {}
        for item in items:
            value = getattr(item, field, "unknown")
            groups[value] = groups.get(value, 0) + 1
        return groups

    def _extract_compliance_indicators(
        self, events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Extract compliance indicators from constitutional events."""
        return {
            "total_compliance_checks": len(events),
            "compliance_categories": list(
                {e.get("category", "unknown") for e in events}
            ),
            "average_check_duration_ms": (
                sum(e.get("duration_ms", 0) for e in events) / len(events)
                if events
                else 0
            ),
        }

    def _extract_recent_violations(
        self, events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Extract recent constitutional violations."""
        # For now, assume violations are events with certain patterns
        violations = []
        for event in events[-10:]:  # Last 10 events
            if "violation" in event.get("impact", "").lower():
                violations.append(
                    {
                        "source": event.get("source"),
                        "impact": event.get("impact"),
                        "timestamp": event.get("timestamp"),
                        "category": event.get("category"),
                    }
                )
        return violations
