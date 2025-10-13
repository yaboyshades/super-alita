"""Application wiring helpers for the consolidation engine."""

from .hooks import ConsolidationAppConfig, build_consolidation_engine, configure_post_turn_consolidation

__all__ = [
    "ConsolidationAppConfig",
    "build_consolidation_engine",
    "configure_post_turn_consolidation",
]
