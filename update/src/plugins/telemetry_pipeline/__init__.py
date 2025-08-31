"""Telemetry Pipeline Plugin for Super Alita"""

from typing import Any, Dict

from src.core.abilities import AbilityRegistry
from src.core.plugin_base import PluginBase

from .abilities import (
    CanonicalizeAbility,
    ClusterAbility,
    CompressorAbility,
    ConflictResolverAbility,
    FinalPromptAssemblerAbility,
    IngestNormalizeAbility,
    PruneAbility,
    RankAbility,
    RelevanceGateAbility,
)


class TelemetryPipelinePlugin(PluginBase):
    """Plugin for processing telemetry into high-signal prompts"""

    def __init__(self):
        super().__init__()
        self.name = "telemetry_pipeline"
        self.version = "1.0.0"

    def register_abilities(self, registry: AbilityRegistry):
        """Register all pipeline abilities"""
        abilities = [
            IngestNormalizeAbility(),
            RelevanceGateAbility(),
            CanonicalizeAbility(),
            RankAbility(),
            ClusterAbility(),
            PruneAbility(),
            ConflictResolverAbility(),
            CompressorAbility(),
            FinalPromptAssemblerAbility(),
        ]

        for ability in abilities:
            registry.register(ability)
