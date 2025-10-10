from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - optional dependency
    import redis.asyncio as redis_async
except Exception:  # pragma: no cover
    redis_async = None

from src.contracts import UnifiedEvent
from src.core.in_memory_event_bus import InMemoryEventBus
from src.orchestration.observability import OrchestatorObserver
from src.unified_intelligence.constitutional_engine import ConstitutionalEngine

from .advanced import AdvancedCapabilityHarness
from .config import BootstrapConfig, InfrastructureConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SystemInfrastructure:
    """Resolved infrastructure objects used by the unified consciousness runtime."""

    eventbus: ConstitutionalEventBus
    memory: NeuralMemoryFederation
    constitutional: ConstitutionalEngine
    observability: OrchestatorObserver
    config: BootstrapConfig
    advanced: AdvancedCapabilityHarness | None = None


class ConstitutionalEventBus:
    """Event bus wrapper that enforces constitutional channels."""

    def __init__(self, config: InfrastructureConfig):
        self._config = config
        self._backend: Any | None = None
        self._channels: set[str] = set()
        self._is_running = False

    async def initialize_constitutional_channels(self) -> None:
        """Bring the event bus online using Redis when available, otherwise fallback to in-memory."""

        redis_cfg = self._config.redis
        backend = None
        if redis_async is not None:
            try:
                backend = redis_async.Redis(
                    host=redis_cfg.host,
                    port=redis_cfg.port,
                    db=redis_cfg.db,
                    decode_responses=False,
                    max_connections=self._config.redis.max_connections,
                )
                await backend.ping()
                logger.info("Connected to Redis at %s", redis_cfg.url)
            except Exception:  # pragma: no cover - depends on environment
                backend = None

        if backend is not None:
            from src.core.redis_event_bus import RedisEventBus

            self._backend = RedisEventBus(
                host=redis_cfg.host,
                port=redis_cfg.port,
                db=redis_cfg.db,
            )
            await self._backend.start()
            self._is_running = True
            logger.info("RedisEventBus ready with constitutional channels")
        else:
            self._backend = InMemoryEventBus()
            await self._backend.start()
            self._is_running = True
            logger.info("In-memory event bus ready (Redis unavailable)")

        for channel in (
            "boot",
            "shutdown",
            "component_ready",
            "component_degraded",
            "constitutional_violation",
            "consciousness_metric",
        ):
            self._channels.add(channel)

    async def publish(self, event: UnifiedEvent) -> None:
        if not self._is_running or self._backend is None:
            raise RuntimeError("Event bus not running")
        await self._backend.emit(
            event.event_type,
            source_plugin=event.source,
            payload=event.payload,
        )

    async def publish_raw(
        self,
        *,
        event_type: str,
        source: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        evt = UnifiedEvent(
            event_type=event_type,  # type: ignore[arg-type]
            source=source,  # type: ignore[arg-type]
            payload=payload or {},
        )
        await self.publish(evt)

    async def subscribe(self, event_type: str, handler: Any) -> None:
        if not self._is_running or self._backend is None:
            raise RuntimeError("Event bus not running")
        await self._backend.subscribe(event_type, handler)

    async def shutdown(self) -> None:
        if self._backend is None:
            return
        stop = getattr(self._backend, "stop", None)
        if stop is not None:
            await stop()
        self._backend = None
        self._is_running = False


class NeuralMemoryFederation:
    """Simple in-memory neural federation facade."""

    def __init__(self) -> None:
        self._store: dict[str, list[dict[str, Any]]] = {}
        self._initialized = False

    async def initialize_cross_component_indexes(self) -> None:
        self._initialized = True
        logger.info("NeuralMemoryFederation indexes initialized")

    async def put(self, component: str, item: dict[str, Any]) -> None:
        if not self._initialized:
            raise RuntimeError("NeuralMemoryFederation not initialized")
        self._store.setdefault(component, []).append(item)

    async def search(
        self,
        component: str,
        *,
        key: str,
        value: Any,
    ) -> list[dict[str, Any]]:
        if not self._initialized:
            return []
        results: list[dict[str, Any]] = []
        for record in self._store.get(component, []):
            if record.get(key) == value:
                results.append(record)
        return results

    async def health(self) -> dict[str, Any]:
        return {
            "initialized": self._initialized,
            "components": len(self._store),
            "items": sum(len(items) for items in self._store.values()),
        }


class InfrastructureBootstrap:
    """Prepare core infrastructure required by the unified consciousness runtime."""

    def __init__(self, config: BootstrapConfig):
        self._config = config

    async def initialize(self) -> SystemInfrastructure:
        await self.ensure_redis_running()
        eventbus = ConstitutionalEventBus(self._config.infrastructure)
        await eventbus.initialize_constitutional_channels()

        memory = NeuralMemoryFederation()
        if self._config.infrastructure.memory.neural_federation_enabled:
            await memory.initialize_cross_component_indexes()

        constitutional = ConstitutionalEngine()
        observability = OrchestatorObserver("unified_consciousness")

        advanced = None
        if (
            self._config.advanced_capabilities
            and self._config.advanced_capabilities.any_enabled()
        ):
            advanced = AdvancedCapabilityHarness(
                config=self._config.advanced_capabilities,
                eventbus=eventbus,
                memory=memory,
                constitutional=constitutional,
                observer=observability,
            )
            await advanced.initialize_infrastructure()

        return SystemInfrastructure(
            eventbus=eventbus,
            memory=memory,
            constitutional=constitutional,
            observability=observability,
            config=self._config,
            advanced=advanced,
        )

    async def ensure_redis_running(self) -> None:
        """Best-effort Redis availability check with optional bootstrap."""

        if redis_async is None:
            logger.warning(
                "redis-py not installed; falling back to in-memory event bus"
            )
            return

        cfg = self._config.infrastructure.redis
        client = redis_async.Redis(host=cfg.host, port=cfg.port, db=cfg.db)
        try:
            await asyncio.wait_for(client.ping(), timeout=1.5)
            logger.info("Redis already running at %s", cfg.url)
            await client.close()
            return
        except Exception:  # pragma: no cover - environment dependent
            await client.close()

        executable = shutil.which("redis-server")
        if executable is None:
            logger.warning(
                "redis-server executable not found; using in-memory event bus"
            )
            return

        try:
            from src.core import proc

            await proc.arun([executable, "--daemonize", "yes"])
            await asyncio.sleep(1.0)
            probe = redis_async.Redis(host=cfg.host, port=cfg.port, db=cfg.db)
            await asyncio.wait_for(probe.ping(), timeout=2.0)
            logger.info("Redis server started via %s", executable)
            await probe.close()
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning(
                "Unable to start redis-server automatically: %s", exc
            )


async def record_boot_events(
    eventbus: ConstitutionalEventBus,
    *,
    components: Iterable[str],
) -> None:
    """Emit boot events for traceability."""

    await eventbus.publish_raw(
        event_type="boot",
        source="bootstrap",
        payload={"components": list(components)},
    )
    for name in components:
        await eventbus.publish_raw(
            event_type="component_ready",
            source=name,
            payload={"phase": "bootstrap"},
        )
