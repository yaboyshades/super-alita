"""Domain-facing exports for the Intelligence Consolidation Engine skeleton."""

from .models import (
    ACEUpdateReceipt,
    ConsolidationEnvelope,
    ConsolidationEvent,
    ConsolidationEventPayload,
    ConsolidationPatch,
    ConsolidationResult,
    ConsolidationRequestContext,
)
from .service import (
    ACEStoreAdapter,
    AbilityRegistryAdapter,
    ConsolidationEngine,
    ConsolidationFeatureFlagProvider,
    ConsolidationMetrics,
    ConsolidationEventPublisher,
)

__all__ = [
    "ACEStoreAdapter",
    "ACEUpdateReceipt",
    "AbilityRegistryAdapter",
    "ConsolidationEngine",
    "ConsolidationFeatureFlagProvider",
    "ConsolidationMetrics",
    "ConsolidationEnvelope",
    "ConsolidationEvent",
    "ConsolidationEventPayload",
    "ConsolidationPatch",
    "ConsolidationEventPublisher",
    "ConsolidationRequestContext",
    "ConsolidationResult",
]
