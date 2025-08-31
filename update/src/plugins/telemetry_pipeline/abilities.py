"""Pipeline abilities for telemetry processing"""

import json
from dataclasses import dataclass
from typing import Any

from src.core.abilities import Ability
from src.core.abilities.decorators import tool

# Shared system prefix for all abilities
SYSTEM_PREFIX = """You are a cautious, loss-aware data reducer. Prefer precision over recall when token-limited.
Never invent facts. Keep numeric values and identifiers. Preserve source ids in square brackets like [E123]."""


@dataclass
class TelemetryItem:
    """Normalized telemetry item"""

    id: str
    ts: str
    source: str
    type: str
    text: str
    facets: dict[str, Any]
    signals: dict[str, float]
    confidence: float
    ttl: int


class IngestNormalizeAbility(Ability):
    """Unifies telemetry shapes into standard format"""

    def __init__(self):
        super().__init__(
            name="telemetry_ingest_normalize",
            description="Normalize telemetry data into unified schema",
            category="pipeline",
        )

    @tool
    async def execute(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize telemetry items to standard schema"""
        normalized = []
        for idx, item in enumerate(items):
            normalized_item = {
                "id": item.get("id", f"E{idx}"),
                "ts": item.get("timestamp", item.get("ts", "")),
                "source": item.get("source", "unknown"),
                "type": item.get("type", "event"),
                "text": item.get("message", item.get("text", str(item))),
                "facets": {
                    "user": item.get("user_id"),
                    "session": item.get("session_id"),
                    "tool": item.get("tool_name"),
                    "span_id": item.get("span_id"),
                },
                "signals": {
                    "latency": item.get("latency_ms", 0),
                    "errors": item.get("error_count", 0),
                    "cost": item.get("cost", 0),
                },
                "confidence": item.get("confidence", 0.8),
                "ttl": item.get("ttl", 3600),
            }
            normalized.append(normalized_item)
        return normalized


class RelevanceGateAbility(Ability):
    """Filter telemetry by task relevance"""

    def __init__(self):
        super().__init__(
            name="telemetry_relevance_gate",
            description="Filter telemetry items by task relevance",
            category="pipeline",
        )
        self.prompt_template = f"""{SYSTEM_PREFIX}

From the telemetry items, keep only those that could change the outcome of the current task.
Discard routine noise. For each kept item, add a 0-1 relevance score.
Output JSON lines."""

    @tool
    async def execute(
        self, task: str, items: list[dict[str, Any]], llm_provider
    ) -> list[dict[str, Any]]:
        """Filter items by relevance to task"""
        # Use LLM to score relevance
        prompt = f"""{self.prompt_template}

Task: {task}

Items to evaluate:
{json.dumps(items, indent=2)}

Output NDJSON with format: {{"id":"E123","keep":true,"relevance":0.86,"reason":"affects model latency"}}
"""

        response = await llm_provider.generate(prompt)
        kept_items = []

        for line in response.strip().split("\n"):
            try:
                result = json.loads(line)
                if result.get("keep", False):
                    # Find original item and add relevance score
                    for item in items:
                        if item["id"] == result["id"]:
                            item["relevance"] = result["relevance"]
                            item["relevance_reason"] = result.get("reason", "")
                            kept_items.append(item)
                            break
            except json.JSONDecodeError:
                continue

        return kept_items


class RankAbility(Ability):
    """Rank items by multiple criteria"""

    def __init__(self):
        super().__init__(
            name="telemetry_rank",
            description="Rank telemetry by relevance, recency, confidence, impact",
            category="pipeline",
        )

    @tool
    async def execute(
        self, task: str, items: list[dict[str, Any]], top_n: int = 200
    ) -> list[dict[str, Any]]:
        """Rank and return top N items"""
        # Simple ranking algorithm (can be enhanced with ML)
        for item in items:
            relevance = item.get("relevance", 0.5)
            recency_score = 1.0  # Calculate based on timestamp
            confidence = item.get("confidence", 0.8)
            impact = (
                item.get("signals", {}).get("errors", 0) * 0.1
                + item.get("signals", {}).get("latency", 0) / 1000.0
            )

            item["rank_score"] = (
                relevance * 0.4
                + recency_score * 0.2
                + confidence * 0.2
                + min(impact, 1.0) * 0.2
            )

        # Sort by rank score
        items.sort(key=lambda x: x.get("rank_score", 0), reverse=True)

        return items[:top_n]


class ClusterAbility(Ability):
    """Group similar telemetry items"""

    def __init__(self):
        super().__init__(
            name="telemetry_cluster",
            description="Cluster similar telemetry items",
            category="pipeline",
        )
        self.prompt_template = f"""{SYSTEM_PREFIX}

Cluster items by semantic similarity and overlapping keys (session/tool/metric).
For each cluster produce: cluster_id, topic, members[], conflicts[] (if contradictory),
and a draft merged summary (≤80 tokens)."""

    @tool
    async def execute(
        self, items: list[dict[str, Any]], llm_provider
    ) -> list[dict[str, Any]]:
        """Cluster similar items"""
        prompt = f"""{self.prompt_template}

Items to cluster:
{json.dumps(items, indent=2)}

Output JSON array of clusters.
"""

        response = await llm_provider.generate(prompt)
        clusters = json.loads(response)

        return clusters


class PruneAbility(Ability):
    """Prune clusters to fit token budget"""

    def __init__(self):
        super().__init__(
            name="telemetry_prune",
            description="Prune content to fit token budget",
            category="pipeline",
        )

    @tool
    async def execute(
        self, clusters: list[dict[str, Any]], token_budget: int = 2000
    ) -> list[dict[str, Any]]:
        """Prune to fit token budget"""
        # Simple token estimation (4 chars = 1 token)
        pruned = []
        total_tokens = 0

        # Sort clusters by total rank of members
        for cluster in clusters:
            cluster["total_rank"] = sum(
                member.get("rank_score", 0) for member in cluster.get("members", [])
            )

        clusters.sort(key=lambda x: x.get("total_rank", 0), reverse=True)

        for cluster in clusters:
            summary_tokens = len(cluster.get("summary", "")) // 4
            if total_tokens + summary_tokens < token_budget:
                pruned.append(cluster)
                total_tokens += summary_tokens
            else:
                break

        return pruned


class FinalPromptAssemblerAbility(Ability):
    """Assemble final prompt from processed telemetry"""

    def __init__(self):
        super().__init__(
            name="telemetry_final_prompt",
            description="Assemble final reasoning prompt",
            category="pipeline",
        )

    @tool
    async def execute(
        self,
        task: str,
        clusters: list[dict[str, Any]],
        conflicts: list[dict[str, Any]],
        constraints: list[str],
    ) -> str:
        """Generate final prompt"""

        # Build critical facts
        facts = []
        for cluster in clusters:
            summary = cluster.get("summary", "")
            source_ids = [m["id"] for m in cluster.get("members", [])]
            facts.append(f"- {summary} [{','.join(source_ids)}]")

        # Build recent changes (filter by timestamp)
        recent = []  # Would filter clusters by recency

        # Build conflicts section
        conflict_notes = []
        for conflict in conflicts:
            note = conflict.get("resolution", "")
            sources = conflict.get("sources", [])
            conflict_notes.append(f"- {note} [{','.join(sources)}]")

        # Assemble final prompt
        prompt = f"""# Task
{task}

# Critical facts
{chr(10).join(facts)}

# Recent changes (24h)
{chr(10).join(recent) if recent else "- No recent changes"}

# Conflicts & stance
{chr(10).join(conflict_notes) if conflict_notes else "- No conflicts detected"}

# Hard constraints
{chr(10).join(f"- {c}" for c in constraints)}

# Your job
Analyze the above information and provide actionable recommendations.
"""

        return prompt
