from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .config import BootstrapConfig, ComponentSettings
from .infrastructure import ConstitutionalEventBus, SystemInfrastructure


@dataclass(slots=True)
class ComponentProcess:
    """Runtime metadata for a launched component."""

    name: str
    command: tuple[str, ...]
    status: str = "not_started"
    started_at: float | None = None
    ready_at: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def mark_started(self) -> None:
        self.status = "starting"
        self.started_at = time.time()

    def mark_ready(self) -> None:
        self.status = "ready"
        self.ready_at = time.time()

    def mark_failed(self, reason: str) -> None:
        self.status = "failed"
        self.metadata["failure_reason"] = reason


class ComponentOrchestrator:
    """Responsible for bringing all configured components online."""

    def __init__(
        self,
        infrastructure: SystemInfrastructure,
        config: BootstrapConfig,
    ) -> None:
        self._infrastructure = infrastructure
        self._config = config
        self._eventbus: ConstitutionalEventBus = infrastructure.eventbus

    async def spawn_all(self) -> dict[str, ComponentProcess]:
        tasks = [
            self._spawn_component(name, settings)
            for name, settings in self._config.components.items()
        ]
        results = await asyncio.gather(*tasks)
        registry = {proc.name: proc for proc in results}
        await self.verify_eventbus_connectivity(registry)
        if (
            self._infrastructure.advanced
            and self._infrastructure.advanced.has_capabilities()
        ):
            await self._infrastructure.advanced.on_components_registered(
                registry
            )
        return registry

    async def _spawn_component(
        self,
        name: str,
        settings: ComponentSettings,
    ) -> ComponentProcess:
        process = ComponentProcess(name=name, command=settings.process_command)
        process.mark_started()

        # Emit bootstrapping event for observability.
        await self._eventbus.publish_raw(
            event_type="component_ready",
            source=name,
            payload={
                "command": list(settings.process_command),
                "options": settings.options,
                "healthcheck": settings.healthcheck_endpoint,
            },
        )

        # Simulate readiness confirmation.
        await asyncio.sleep(0.1)
        process.mark_ready()
        process.metadata.update(settings.options)
        return process

    async def verify_eventbus_connectivity(
        self,
        processes: dict[str, ComponentProcess],
    ) -> None:
        # Round-trip a heartbeat event to ensure event bus is receptive.
        ready_event = asyncio.Event()

        async def _handler(
            _event,
        ) -> None:  # pragma: no cover - exercised in integration tests
            ready_event.set()

        await self._eventbus.subscribe("heartbeat", _handler)
        await self._eventbus.publish_raw(
            event_type="heartbeat",
            source="component_orchestrator",
            payload={"components": list(processes)},
        )
        await asyncio.wait_for(ready_event.wait(), timeout=1.0)
