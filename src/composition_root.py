"""Composition Root - unified architecture bootstrap.

Single initialization point for:
- Event Store (Redis Streams)
- Component Registry (Hot-swappable)
- Constitutional Middleware
- Distributed Locks
- Memory & Compliance Services
- Adapters (Codex, SuperAlita, CMA)
- Event Orchestrator

This is the ONLY place where components are instantiated and wired together.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Import core infrastructure
from src.adapters.cma_adapter import CMAAdapter
from src.adapters.codex_adapter import CodexAdapter
from src.adapters.super_alita_adapter import SuperAlitaAdapter
from src.contracts import Compliance, HealthStatus, Memory
from src.orchestration.component_registry import (
    ComponentDescriptor,
    ComponentRegistry,
)
from src.orchestration.constitutional_middleware import (
    Constitution,
    ConstitutionalArticle,
    ConstitutionalMiddleware,
)
from src.orchestration.event_orchestrator import EventOrchestrator
from src.orchestration.event_store import EventStore


# Stub implementations for Memory and Compliance
# (Replace with real implementations)
class InMemoryMemory(Memory):
    """In-memory stub for Memory service."""

    def __init__(self):
        self.store: dict[str, Any] = {}

    async def put(self, item: dict[str, Any], corr_id: str) -> str:
        item_id = corr_id
        self.store[item_id] = item
        return item_id

    async def search(
        self, query: str, k: int = 8, corr_id: str | None = None
    ) -> list[dict[str, Any]]:
        # Simplified search
        return list(self.store.values())[:k]

    async def health_check(self) -> HealthStatus:
        return HealthStatus(
            component="memory",
            status="healthy",
            details={"items": len(self.store)},
        )


class BasicCompliance(Compliance):
    """Basic stub for Compliance service."""

    async def validate(
        self, artifact: str, kind: str, corr_id: str
    ) -> dict[str, Any]:
        # Simplified validation
        score = 0.85 if len(artifact) > 0 else 0.0
        return {"score": score, "violations": [], "details": {}}

    async def health_check(self) -> HealthStatus:
        return HealthStatus(component="compliance", status="healthy")


# Constitutional article validators (simplified)
async def validate_test_first(evt) -> float:
    """Validate Test-First principle."""
    if evt.event_type == "code_generate":
        if "test_requirements" in evt.payload:
            return 1.0
        return 0.3
    return 1.0


async def validate_simplicity(evt) -> float:
    """Validate Simplicity Gate."""
    if evt.event_type == "code_generate":
        complexity = evt.payload.get("complexity", 5)
        if complexity <= 10:
            return 1.0
        return 0.5
    return 1.0


class CompositionRoot:
    """Composition root - wires all components together.

    Usage:
        root = CompositionRoot()
        await root.initialize()
        await root.start()
    """

    def __init__(self):
        """Initialize composition root (no components created yet)."""
        self.orchestrator: EventOrchestrator | None = None
        self.event_store: EventStore | None = None
        self.registry: ComponentRegistry | None = None

    async def initialize(self) -> None:
        """Initialize all components in dependency order."""
        logger.info("=== Composition Root: Initializing ===")

        # 1. Configuration
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

        # 2. Event Store
        logger.info("Creating Event Store...")
        self.event_store = EventStore(redis_url)

        # 3. Component Registry
        logger.info("Creating Component Registry...")
        self.registry = ComponentRegistry()

        # 4. Constitution & Middleware
        logger.info("Creating Constitutional Framework...")
        constitution = Constitution()

        # Add constitutional articles
        constitution.add_article(
            ConstitutionalArticle(
                id="Article_II",
                name="Test-First",
                description="All code must have test requirements",
                validate=validate_test_first,
                threshold=0.75,
            )
        )

        constitution.add_article(
            ConstitutionalArticle(
                id="Article_III",
                name="Simplicity Gate",
                description="Functions ≤50 lines, complexity ≤10",
                validate=validate_simplicity,
                threshold=0.75,
            )
        )

        middleware = ConstitutionalMiddleware(constitution, strict_mode=False)

        # 5. Core Services
        logger.info("Creating Core Services...")
        memory = InMemoryMemory()
        compliance = BasicCompliance()

        # 6. Adapters
        logger.info("Creating Adapters...")
        # Stub event bus for adapters (in real impl, use proper EventBus)
        stub_bus = None

        codex = CodexAdapter(stub_bus)
        super_alita = SuperAlitaAdapter(stub_bus)
        cma = CMAAdapter(stub_bus)

        adapters = {
            "codex": codex,
            "super_alita": super_alita,
            "cma": cma,
        }

        # 7. Event Orchestrator (the routing brain)
        logger.info("Creating Event Orchestrator...")
        self.orchestrator = EventOrchestrator(
            event_store=self.event_store,
            registry=self.registry,
            middleware=middleware,
            memory=memory,
            compliance=compliance,
            adapters=adapters,
            enable_tracing=False,  # Set True if OpenTelemetry available
        )

        # 8. Register components
        logger.info("Registering Components...")
        for name, adapter in adapters.items():
            desc = ComponentDescriptor(
                name=name,
                provides={f"{name}.handle"},
                requires={"event_store"},
                health_check=adapter.health_check,
            )
            await self.registry.register(desc, adapter)

        logger.info("=== Composition Root: Initialization Complete ===")

    async def start(self) -> None:
        """Start the orchestrator event loop."""
        if not self.orchestrator:
            raise RuntimeError("Composition root not initialized")

        logger.info("=== Starting Orchestrator ===")
        await self.orchestrator.boot()

        # Run orchestrator (blocks until shutdown)
        await self.orchestrator.run()

    async def health_check(self) -> dict[str, Any]:
        """Check health of all components."""
        if not self.orchestrator:
            return {"status": "not_initialized"}

        health = await self.orchestrator.check_health()
        metrics = self.orchestrator.get_metrics()

        return {
            "status": "running",
            "health": health,
            "metrics": metrics,
        }


# Entry point for CLI usage
async def main():
    """Main entry point."""
    root = CompositionRoot()
    await root.initialize()

    # Start orchestrator (runs until Ctrl+C)
    try:
        await root.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
