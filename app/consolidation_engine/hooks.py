"""App-layer helpers for integrating the consolidation engine into REUG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adapters.consolidation_engine import (
    AbilityRegistryAdapterImpl,
    EnvironmentFeatureFlagProvider,
    EventBusPublisherAdapter,
)
from domain.consolidation_engine import ConsolidationEngine, ConsolidationMetrics
from domain.consolidation_engine.service import ConsolidationFeatureFlagProvider


@dataclass(slots=True)
class ConsolidationAppConfig:
    """Container describing runtime dependencies for the consolidation engine."""

    event_bus: Any
    ability_registry: Any
    ace_store: Any
    flag_provider: ConsolidationFeatureFlagProvider | None = None
    metrics: ConsolidationMetrics | None = None


def build_consolidation_engine(config: ConsolidationAppConfig) -> ConsolidationEngine:
    """Instantiate the domain service using application-level dependencies."""

    flag_provider = config.flag_provider or EnvironmentFeatureFlagProvider()
    publisher = EventBusPublisherAdapter(config.event_bus)
    ability_adapter = AbilityRegistryAdapterImpl(config.ability_registry)
    return ConsolidationEngine(
        feature_flags=flag_provider,
        event_publisher=publisher,
        ace_store=config.ace_store,
        ability_registry=ability_adapter,
        metrics=config.metrics,
    )


def configure_post_turn_consolidation(
    *, loop: Any, config: ConsolidationAppConfig
) -> ConsolidationEngine:
    """Wire consolidation into the REUG loop if hooks are available."""

    engine = build_consolidation_engine(config)
    register = getattr(loop, "register_post_turn_hook", None)
    if callable(register):
        register(engine.consolidate_post_turn)
    return engine
