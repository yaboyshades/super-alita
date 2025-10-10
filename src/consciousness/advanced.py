from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.contracts import UnifiedEvent

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from src.orchestration.observability import OrchestatorObserver
    from src.unified_intelligence.constitutional_engine import (
        ConstitutionalEngine,
    )

    from .components import ComponentProcess
    from .config import AdvancedCapabilitiesConfig
    from .infrastructure import ConstitutionalEventBus, NeuralMemoryFederation
    from .orchestrator import UnifiedConsciousness


BackgroundTaskFactory = Callable[[], Awaitable[None]]


class AdvancedCapability:
    """Base helper for advanced consciousness capabilities."""

    def __init__(
        self, name: str, logger: logging.Logger | None = None
    ) -> None:
        self.name = name
        self._logger = logger or logging.getLogger(f"advanced.{name}")

    async def on_infrastructure_ready(self) -> None:
        """Hook executed once shared infrastructure is ready."""

    async def on_components_registered(
        self, components: dict[str, ComponentProcess]
    ) -> None:
        """Hook executed once all components have been spawned."""

    async def prepare_consciousness_extensions(
        self, consciousness: UnifiedConsciousness
    ) -> Sequence[BackgroundTaskFactory]:
        """Return background task factories to extend the main runtime."""
        return ()


class TribalKnowledgeExtractorModule(AdvancedCapability):
    def __init__(
        self,
        eventbus: ConstitutionalEventBus,
        config,
    ) -> None:
        super().__init__("tribal_knowledge_extractor")
        self._eventbus = eventbus
        self._config = config
        self._subscriptions_registered = False

    async def on_infrastructure_ready(self) -> None:
        if not self._config.enabled or self._subscriptions_registered:
            return
        handlers: list[
            tuple[str, Callable[[UnifiedEvent], Awaitable[None]]]
        ] = []
        if self._config.integration_points.get("super_alita_genealogy"):
            handlers.append(
                ("genealogy_evolution", self._handle_genealogy_event)
            )
        if self._config.integration_points.get("codex_configuration"):
            handlers.append(
                ("codex_config_change", self._handle_configuration_event)
            )
        if self._config.integration_points.get("sdd_workflow_phases"):
            handlers.append(
                ("sdd_phase_complete", self._handle_sdd_phase_event)
            )

        for topic, handler in handlers:
            await self._eventbus.subscribe(topic, handler)
        if handlers:
            self._subscriptions_registered = True
            self._logger.info(
                "TKE subscriptions active for %s",
                ", ".join(topic for topic, _ in handlers),
            )

    async def _handle_genealogy_event(self, event: UnifiedEvent) -> None:
        self._logger.debug(
            "Captured genealogy evolution event: %s", event.payload
        )

    async def _handle_configuration_event(self, event: UnifiedEvent) -> None:
        self._logger.debug(
            "Captured codex configuration change: %s", event.payload
        )

    async def _handle_sdd_phase_event(self, event: UnifiedEvent) -> None:
        self._logger.debug(
            "Captured SDD workflow completion: %s", event.payload
        )


class LivingArchitectureModule(AdvancedCapability):
    def __init__(
        self,
        memory: NeuralMemoryFederation,
        eventbus: ConstitutionalEventBus,
        config,
    ) -> None:
        super().__init__("living_architecture")
        self._memory = memory
        self._eventbus = eventbus
        self._config = config

    async def prepare_consciousness_extensions(
        self, consciousness: UnifiedConsciousness
    ) -> Sequence[BackgroundTaskFactory]:
        if not self._config.enabled:
            return ()

        async def monitor_reality() -> None:
            interval = max(self._config.reality_gap_threshold, 0.05)
            while consciousness.is_running():
                snapshot = {
                    "score": consciousness.current_score,
                    "timestamp": asyncio.get_event_loop().time(),
                }
                await self._memory.put("living_architecture", snapshot)
                await self._eventbus.publish_raw(
                    event_type="reality_gap_probe",
                    source="living_architecture",
                    payload={"score": consciousness.current_score},
                )
                await asyncio.sleep(interval)

        return (monitor_reality,)


class UniversalPatternMinerModule(AdvancedCapability):
    def __init__(self, memory: NeuralMemoryFederation, config) -> None:
        super().__init__("universal_pattern_miner")
        self._memory = memory
        self._config = config

    async def on_components_registered(
        self, components: dict[str, ComponentProcess]
    ) -> None:
        if not self._config.enabled:
            return
        patterns = {
            "components": list(components.keys()),
            "strategies": list(self._config.mining_strategies),
            "integration": self._config.integration_points,
        }
        await self._memory.put("universal_patterns", patterns)
        self._logger.info(
            "Recorded universal pattern seed with %d components",
            len(components),
        )


class ArchitecturalTimeTravelModule(AdvancedCapability):
    def __init__(self, eventbus: ConstitutionalEventBus, config) -> None:
        super().__init__("architectural_time_travel")
        self._eventbus = eventbus
        self._config = config

    async def prepare_consciousness_extensions(
        self, consciousness: UnifiedConsciousness
    ) -> Sequence[BackgroundTaskFactory]:
        if not self._config.enabled:
            return ()

        async def forecast_loop() -> None:
            horizons = self._config.prediction_horizons or ("3_months",)
            while consciousness.is_running():
                payload = {
                    "horizons": list(horizons),
                    "score": consciousness.current_score,
                    "debt_forecasting": self._config.debt_forecasting,
                }
                await self._eventbus.publish_raw(
                    event_type="architectural_forecast_tick",
                    source="architectural_time_travel",
                    payload=payload,
                )
                await asyncio.sleep(1.0)

        return (forecast_loop,)


class ConstitutionalEvolutionModule(AdvancedCapability):
    def __init__(
        self,
        eventbus: ConstitutionalEventBus,
        observer: OrchestatorObserver,
        config,
    ) -> None:
        super().__init__("constitutional_evolution")
        self._eventbus = eventbus
        self._observer = observer
        self._config = config

    async def on_components_registered(
        self, components: dict[str, ComponentProcess]
    ) -> None:
        if not self._config.enabled:
            return
        await self._eventbus.publish_raw(
            event_type="constitutional_evolution_initialized",
            source="constitutional_evolution",
            payload={
                "components": list(components.keys()),
                "threshold": self._config.evidence_strength_required,
            },
        )
        self._observer.logger.info(
            "Constitutional evolution monitoring enabled"
        )


class ArchitecturalConsciousnessModule(AdvancedCapability):
    def __init__(self, config) -> None:
        super().__init__("architectural_consciousness")
        self._config = config

    async def prepare_consciousness_extensions(
        self, consciousness: UnifiedConsciousness
    ) -> Sequence[BackgroundTaskFactory]:
        if not self._config.enabled:
            return ()

        async def awareness_loop() -> None:
            while consciousness.is_running():
                if (
                    consciousness.current_score
                    < self._config.consciousness_threshold
                ):
                    logging.getLogger(
                        "advanced.architectural_consciousness"
                    ).warning(
                        "Consciousness score %.3f below threshold %.2f",
                        consciousness.current_score,
                        self._config.consciousness_threshold,
                    )
                await asyncio.sleep(1.5)

        return (awareness_loop,)


class ArchitecturalImmuneSystemModule(AdvancedCapability):
    def __init__(self, eventbus: ConstitutionalEventBus, config) -> None:
        super().__init__("architectural_immune_system")
        self._eventbus = eventbus
        self._config = config

    async def prepare_consciousness_extensions(
        self, consciousness: UnifiedConsciousness
    ) -> Sequence[BackgroundTaskFactory]:
        if not self._config.enabled:
            return ()

        async def immune_loop() -> None:
            while consciousness.is_running():
                await self._eventbus.publish_raw(
                    event_type="architectural_health_ping",
                    source="architectural_immune_system",
                    payload={
                        "monitoring": self._config.health_monitoring,
                        "score": consciousness.current_score,
                    },
                )
                await asyncio.sleep(2.0)

        return (immune_loop,)


class PhilosophicalArchitectureModule(AdvancedCapability):
    def __init__(self, eventbus: ConstitutionalEventBus, config) -> None:
        super().__init__("philosophical_architecture")
        self._eventbus = eventbus
        self._config = config

    async def on_infrastructure_ready(self) -> None:
        if not self._config.enabled:
            return
        await self._eventbus.publish_raw(
            event_type="socratic_engine_ready",
            source="philosophical_architecture",
            payload={"depth": self._config.socratic_depth},
        )


@dataclass(slots=True)
class AdvancedCapabilityHarness:
    """Coordinator that wires optional advanced modules into the runtime."""

    config: AdvancedCapabilitiesConfig
    eventbus: ConstitutionalEventBus
    memory: NeuralMemoryFederation
    constitutional: ConstitutionalEngine
    observer: OrchestatorObserver
    _logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("advanced.harness")
    )
    _capabilities: list[AdvancedCapability] = field(
        default_factory=list, init=False
    )

    def __post_init__(self) -> None:  # pragma: no cover - simple wiring
        if not self.config.any_enabled():
            return
        if self.config.tribal_knowledge_extractor.enabled:
            self._capabilities.append(
                TribalKnowledgeExtractorModule(
                    self.eventbus, self.config.tribal_knowledge_extractor
                )
            )
        if self.config.living_architecture_engine.enabled:
            self._capabilities.append(
                LivingArchitectureModule(
                    self.memory,
                    self.eventbus,
                    self.config.living_architecture_engine,
                )
            )
        if self.config.universal_pattern_miner.enabled:
            self._capabilities.append(
                UniversalPatternMinerModule(
                    self.memory, self.config.universal_pattern_miner
                )
            )
        if self.config.architectural_time_travel.enabled:
            self._capabilities.append(
                ArchitecturalTimeTravelModule(
                    self.eventbus, self.config.architectural_time_travel
                )
            )
        if self.config.constitutional_evolution.enabled:
            self._capabilities.append(
                ConstitutionalEvolutionModule(
                    self.eventbus,
                    self.observer,
                    self.config.constitutional_evolution,
                )
            )
        if self.config.architectural_consciousness.enabled:
            self._capabilities.append(
                ArchitecturalConsciousnessModule(
                    self.config.architectural_consciousness
                )
            )
        if self.config.architectural_immune_system.enabled:
            self._capabilities.append(
                ArchitecturalImmuneSystemModule(
                    self.eventbus, self.config.architectural_immune_system
                )
            )
        if self.config.philosophical_architecture.enabled:
            self._capabilities.append(
                PhilosophicalArchitectureModule(
                    self.eventbus, self.config.philosophical_architecture
                )
            )

    def has_capabilities(self) -> bool:
        return bool(self._capabilities)

    async def initialize_infrastructure(self) -> None:
        for capability in self._capabilities:
            await capability.on_infrastructure_ready()
        if self._capabilities:
            enabled = ", ".join(self.config.enabled_capability_names())
            self._logger.info("Advanced capabilities initialized: %s", enabled)

    async def on_components_registered(
        self, components: dict[str, ComponentProcess]
    ) -> None:
        for capability in self._capabilities:
            await capability.on_components_registered(components)

    async def prepare_consciousness_extensions(
        self, consciousness: UnifiedConsciousness
    ) -> list[BackgroundTaskFactory]:
        task_factories: list[BackgroundTaskFactory] = []
        for capability in self._capabilities:
            factories = await capability.prepare_consciousness_extensions(
                consciousness
            )
            task_factories.extend(factories)
        return task_factories
