"""Adapters wiring the consolidation engine skeleton to runtime services."""

from .ability_registry import AbilityRegistryAdapterImpl
from .event_publisher import EventBusPublisherAdapter
from .feature_flags import EnvironmentFeatureFlagProvider

__all__ = [
    "AbilityRegistryAdapterImpl",
    "EnvironmentFeatureFlagProvider",
    "EventBusPublisherAdapter",
]
