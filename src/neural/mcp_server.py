#!/usr/bin/env python3
"""
MCP-style server with ability contracts for Neural Atoms
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from .store import MessageStore

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Histogram, start_http_server

    # Prometheus metrics
    process_time = Histogram(
        "neural_atom_processing_seconds", "Processing time per atom event"
    )
    prometheus_available = True
except ImportError:
    prometheus_available = False

    # Mock histogram for graceful degradation
    class MockHistogram:
        def time(self):
            def decorator(func: Callable) -> Callable:
                return func

            return decorator

    process_time = MockHistogram()


class MCPServer:
    """MCP-style server with ability contracts"""

    def __init__(self, store: MessageStore, metrics_port: int = 8000):
        self.store = store
        self.handlers: dict[str, Callable] = {}
        self.metrics_port = metrics_port
        self.running = False

    def ability(self, event_type: str):
        """Decorator for registering ability handlers"""

        def decorator(func: Callable):
            self.handlers[event_type] = func
            logger.info(f"Registered ability handler for: {event_type}")
            return func

        return decorator

    async def handle_event(
        self, event_type: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle event with persistence and metrics"""
        logger.debug(f"Handling event: {event_type}")

        try:
            # Persist event first
            self.store.persist(event_type, payload)

            # Handle event
            if event_type in self.handlers:
                if prometheus_available:
                    with process_time.time():
                        result = await self._call_handler(
                            self.handlers[event_type], payload
                        )
                else:
                    result = await self._call_handler(
                        self.handlers[event_type], payload
                    )
                logger.debug(f"Event {event_type} handled successfully")
                return result
            else:
                logger.warning(f"No handler registered for event type: {event_type}")
                return {"status": "error", "message": f"No handler for {event_type}"}

        except Exception as e:
            logger.error(f"Error handling event {event_type}: {e}")
            return {"status": "error", "message": str(e)}

    async def _call_handler(
        self, handler: Callable, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Call handler with proper async handling"""
        if asyncio.iscoroutinefunction(handler):
            return await handler(payload)
        else:
            return handler(payload)

    async def start_server(self):
        """Start Prometheus metrics server"""
        if not self.running and prometheus_available:
            start_http_server(self.metrics_port)
            self.running = True
            logger.info(
                f"Prometheus metrics server started on port {self.metrics_port}"
            )
        elif not prometheus_available:
            logger.info("Prometheus not available, metrics server not started")
        else:
            logger.debug("Metrics server already running")

    def shutdown(self):
        """Shutdown the MCP server"""
        if self.running:
            self.running = False
            logger.info("MCP server shutdown")
