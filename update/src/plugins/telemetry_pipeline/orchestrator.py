"""Orchestrator for the telemetry pipeline"""

from typing import Any

from src.core.abilities import AbilityRegistry


class TelemetryPipelineOrchestrator:
    """Orchestrates the multi-stage telemetry pipeline"""

    def __init__(self, ability_registry: AbilityRegistry, llm_provider):
        self.registry = ability_registry
        self.llm = llm_provider

        # Get abilities
        self.ingest = ability_registry.get("telemetry_ingest_normalize")
        self.relevance = ability_registry.get("telemetry_relevance_gate")
        self.rank = ability_registry.get("telemetry_rank")
        self.cluster = ability_registry.get("telemetry_cluster")
        self.prune = ability_registry.get("telemetry_prune")
        self.assemble = ability_registry.get("telemetry_final_prompt")

    async def process_telemetry(
        self,
        task: str,
        telemetry_items: list[dict[str, Any]],
        constraints: list[str] = None,
        token_budget: int = 2000,
    ) -> str:
        """Process telemetry through the full pipeline"""

        # Stage 1: Ingest & Normalize
        normalized = await self.ingest.execute(telemetry_items)

        # Stage 2: Relevance Gate
        relevant = await self.relevance.execute(task, normalized, self.llm)

        # Stage 3: Rank
        ranked = await self.rank.execute(task, relevant, top_n=200)

        # Stage 4: Cluster
        clusters = await self.cluster.execute(ranked, self.llm)

        # Stage 5: Prune
        pruned = await self.prune.execute(clusters, token_budget)

        # Stage 6: Assemble Final Prompt
        conflicts = []  # Extract from clusters
        final_prompt = await self.assemble.execute(
            task, pruned, conflicts, constraints or []
        )

        return final_prompt
