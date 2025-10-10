"""Unified Orchestrator - the routing brain for all component interactions.

Single event loop that:
- Routes events through constitutional middleware
- Manages component lifecycle via registry
- Propagates correlation IDs and OpenTelemetry spans
- Applies compliance taps on all operations
- Coordinates adapters and core services
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore[assignment]

from src.contracts import (
    Adapter,
    Compliance,
    HealthStatus,
    Memory,
    UnifiedEvent,
)
from src.orchestration.component_registry import ComponentRegistry
from src.orchestration.constitutional_middleware import (
    ConstitutionalMiddleware,
    InterceptAction,
)
from src.orchestration.event_store import EventStore

logger = logging.getLogger(__name__)


class EventOrchestrator:
    """Central orchestrator coordinating all system components.

    Responsibilities:
    - Event routing with constitutional validation
    - Correlation ID propagation
    - OpenTelemetry span management
    - Component health monitoring
    - Graceful startup/shutdown
    - Adapter coordination
    """

    def __init__(
        self,
        event_store: EventStore,
        registry: ComponentRegistry,
        middleware: ConstitutionalMiddleware,
        memory: Memory,
        compliance: Compliance,
        adapters: dict[str, Adapter],
        enable_tracing: bool = True,
    ):
        """Initialize orchestrator.

        Args:
            event_store: Event sourcing store
            registry: Component registry
            middleware: Constitutional middleware
            memory: Memory service
            compliance: Compliance service
            adapters: Dict of adapter name -> adapter instance
            enable_tracing: Enable OpenTelemetry tracing
        """
        self.event_store = event_store
        self.registry = registry
        self.middleware = middleware
        self.memory = memory
        self.compliance = compliance
        self.adapters = adapters
        self.enable_tracing = enable_tracing and OTEL_AVAILABLE

        # Event routing
        self._handlers: dict[
            str, list[Callable[[UnifiedEvent], Coroutine]]
        ] = {}
        self._running = False
        self._event_queue: asyncio.Queue[UnifiedEvent] = asyncio.Queue()

        # Metrics
        self.events_processed = 0
        self.events_blocked = 0
        self.events_transformed = 0

        # Tracer
        if self.enable_tracing:
            self.tracer = trace.get_tracer(__name__)  # type: ignore[union-attr]
        else:
            self.tracer = None

    async def boot(self) -> None:
        """Boot the orchestrator and announce readiness."""
        logger.info("Booting event orchestrator...")

        # Connect event store
        await self.event_store.connect()

        # Register adapters
        for name, adapter in self.adapters.items():
            await self._subscribe_adapter(name, adapter)

        # Emit boot event
        boot_event = UnifiedEvent(
            event_type="boot",
            source="orchestrator",
            payload={
                "adapters": list(self.adapters.keys()),
                "status": "ready",
            },
        )

        await self._emit_internal(boot_event)
        logger.info("Event orchestrator boot complete")

    async def run(self) -> None:
        """Run the main event loop."""
        self._running = True
        logger.info("Starting orchestrator event loop...")

        try:
            while self._running:
                evt = await self._event_queue.get()
                await self._process_event(evt)
                self._event_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Orchestrator event loop cancelled")
        except Exception as e:
            logger.exception(f"Fatal error in orchestrator: {e}")
            raise
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully shutdown orchestrator."""
        logger.info("Shutting down orchestrator...")
        self._running = False

        shutdown_event = UnifiedEvent(
            event_type="shutdown",
            source="orchestrator",
            payload={"reason": "graceful_shutdown"},
        )
        await self._emit_internal(shutdown_event)
        await self.event_store.disconnect()
        logger.info("Orchestrator shutdown complete")

    async def emit(self, evt: UnifiedEvent) -> None:
        """Emit an event into the orchestrator."""
        await self._event_queue.put(evt)

    async def _emit_internal(self, evt: UnifiedEvent) -> None:
        """Emit without queueing (for system events)."""
        await self._process_event(evt)

    async def _process_event(self, evt: UnifiedEvent) -> None:
        """Process a single event through the pipeline."""
        span_context = None

        try:
            # Start OpenTelemetry span
            if self.enable_tracing and self.tracer:
                span_context = self.tracer.start_span(  # type: ignore[union-attr]
                    f"event.{evt.event_type}",
                    attributes={
                        "event.id": evt.event_id,
                        "event.type": evt.event_type,
                        "event.source": evt.source,
                        "event.target": evt.target or "broadcast",
                        "correlation.id": evt.corr_id,
                    },
                )
                span_context.__enter__()

            # Constitutional middleware validation
            result = await self.middleware.intercept(evt)

            if result.action == InterceptAction.BLOCK:
                self.events_blocked += 1
                logger.warning(
                    f"Event {evt.event_id} BLOCKED: {result.message}",
                    extra={
                        "corr_id": evt.corr_id,
                        "violations": len(result.violations),
                    },
                )
                if span_context:
                    span_context.set_status(  # type: ignore[union-attr]
                        Status(StatusCode.ERROR, "Constitutional violation")
                    )
                return

            if result.action == InterceptAction.TRANSFORM:
                self.events_transformed += 1
                evt = result.transformed_event  # type: ignore[assignment]

            # Persist to event store
            await self.event_store.append("unified-stream", [evt])

            # Optional compliance tap
            if evt.event_type in ["code_generate", "sdd_specify", "sdd_plan"]:
                await self._compliance_tap(evt)

            # Route to handlers
            await self._route(evt)

            self.events_processed += 1

            if span_context:
                span_context.set_status(Status(StatusCode.OK))  # type: ignore[union-attr]

        except Exception as e:
            logger.exception(f"Error processing event {evt.event_id}: {e}")
            if span_context:
                span_context.set_status(Status(StatusCode.ERROR, str(e)))  # type: ignore[union-attr]
            raise
        finally:
            if span_context:
                span_context.__exit__(None, None, None)

    async def _route(self, evt: UnifiedEvent) -> None:
        """Route event to registered handlers."""
        # Route to adapters
        for name, adapter in self.adapters.items():
            if evt.target is None or evt.target == name:
                try:
                    await adapter.handle(evt)
                except Exception as e:
                    logger.exception(f"Adapter {name} failed: {e}")

        # Route to registered handlers
        handlers = self._handlers.get(evt.event_type, [])
        for handler in handlers:
            try:
                await handler(evt)
            except Exception as e:
                logger.exception(f"Handler failed for {evt.event_type}: {e}")

    async def _compliance_tap(self, evt: UnifiedEvent) -> None:
        """Tap event for compliance analysis."""
        try:
            artifact = evt.payload.get("code") or evt.payload.get("spec") or ""
            if artifact:
                result = await self.compliance.validate(
                    artifact=str(artifact),
                    kind=evt.event_type,
                    corr_id=evt.corr_id,
                )
                logger.info(
                    f"Compliance: {evt.event_type} score={result.get('score', 0):.2f}"
                )
        except Exception as e:
            logger.warning(f"Compliance tap failed: {e}")

    async def _subscribe_adapter(self, name: str, adapter: Adapter) -> None:
        """Subscribe an adapter to events."""
        logger.info(f"Subscribing adapter: {name}")
        ready_event = UnifiedEvent(
            event_type="component_ready",
            source="orchestrator",
            target=name,
            payload={"component": name, "type": "adapter"},
        )
        await self._emit_internal(ready_event)

    def subscribe(
        self, event_type: str, handler: Callable[[UnifiedEvent], Coroutine]
    ) -> None:
        """Subscribe a handler to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    async def check_health(self) -> dict[str, HealthStatus]:
        """Check health of all components."""
        health = {}

        for name, adapter in self.adapters.items():
            health[f"adapter.{name}"] = await adapter.health_check()

        health.update(await self.registry.check_all_health())
        health["memory"] = await self.memory.health_check()
        health["compliance"] = await self.compliance.health_check()

        try:
            await self.event_store.get_stream_info("unified-stream")
            health["event_store"] = HealthStatus(
                component="event_store", status="healthy"
            )
        except Exception as e:
            health["event_store"] = HealthStatus(
                component="event_store", status="unhealthy", message=str(e)
            )

        return health

    def get_metrics(self) -> dict[str, Any]:
        """Get orchestrator metrics."""
        return {
            "events_processed": self.events_processed,
            "events_blocked": self.events_blocked,
            "events_transformed": self.events_transformed,
            "queue_size": self._event_queue.qsize(),
            "adapters": list(self.adapters.keys()),
            "running": self._running,
        }
