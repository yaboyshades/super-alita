"""Orchestrator for the telemetry pipeline."""

from __future__ import annotations

from typing import Any

from src.plugins.telemetry_pipeline.abilities import (
    ClusterAbility,
    FinalPromptAssemblerAbility,
    IngestNormalizeAbility,
    PruneAbility,
    RankAbility,
    RelevanceGateAbility,
)


class TelemetryPipelineOrchestrator:
    """Orchestrates the multi-stage telemetry pipeline."""

    def __init__(self, llm_provider: Any = None) -> None:
        self.llm = llm_provider

        # Initialize abilities
        self.ingest = IngestNormalizeAbility()
        self.relevance = RelevanceGateAbility()
        self.rank = RankAbility()
        self.cluster = ClusterAbility()
        self.prune = PruneAbility()
        self.assemble = FinalPromptAssemblerAbility()

    async def process_telemetry(
        self,
        task: str,
        telemetry_items: list[dict[str, Any]],
        constraints: list[str] | None = None,
        token_budget: int = 2000,
    ) -> str:
        """Process telemetry through the full pipeline."""

        # Stage 1: Ingest & Normalize
        normalized = await self.ingest.execute(items=telemetry_items)

        # Stage 2: Relevance Gate
        relevant = await self.relevance.execute(
            task=task, items=normalized, llm_provider=self.llm
        )

        # Stage 3: Rank
        ranked = await self.rank.execute(task=task, items=relevant, top_n=200)

        # Stage 4: Cluster
        clusters = await self.cluster.execute(
            items=ranked, llm_provider=self.llm
        )

        # Stage 5: Prune
        pruned = await self.prune.execute(
            clusters=clusters, token_budget=token_budget
        )

        # Stage 6: Assemble Final Prompt
        conflicts: list[dict[str, Any]] = []  # Extract from clusters if needed
        final_prompt = await self.assemble.execute(
            task=task,
            clusters=pruned,
            conflicts=conflicts,
            constraints=constraints or [],
        )

        return final_prompt
