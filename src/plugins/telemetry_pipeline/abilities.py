"""Telemetry pipeline abilities.

Composable, side‑effect free steps to transform raw telemetry into
high‑signal prompt context. All abilities share a uniform async
``execute(*args, **kwargs)`` signature for flexibility.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar, cast

SYSTEM_PREFIX = (
    "You are a cautious, loss-aware data reducer. Prefer precision over "
    "recall when token-limited. Never invent facts. Keep numeric values "
    "and identifiers. Preserve source ids in square brackets like [E123]."
)


class Ability:
    """Minimal base class for abilities."""

    name: str
    description: str
    category: str

    def __init__(
        self, name: str, description: str, category: str = "pipeline"
    ) -> None:
        self.name = name
        self.description = description
        self.category = category

    async def execute(
        self, *args: Any, **kwargs: Any
    ) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


class LLMProvider(Protocol):
    async def generate(self, prompt: str) -> str: ...  # pragma: no cover


TFunc = TypeVar("TFunc", bound=Callable[..., Coroutine[Any, Any, Any]])


def tool(fn: TFunc) -> TFunc:  # decorator reserved for future metadata
    return fn


@dataclass
class TelemetryItem:
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
    def __init__(self) -> None:
        super().__init__(
            name="telemetry_ingest_normalize",
            description="Normalize telemetry data into unified schema",
        )

    @tool
    async def execute(
        self, *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = kwargs.get("items") or (
            args[0] if args else []
        )
        out: list[dict[str, Any]] = []
        for idx, item in enumerate(items):
            out.append(
                {
                    "id": item.get("id", f"E{idx}"),
                    "ts": item.get("timestamp", item.get("ts", "")),
                    "source": item.get("source", "unknown"),
                    "type": item.get("type", "event"),
                    "text": item.get(
                        "message", item.get("text", str(item))
                    ),
                    "facets": {
                        "user": item.get("user_id"),
                        "session": item.get("session_id"),
                        "tool": item.get("tool_name"),
                        "span_id": item.get("span_id"),
                    },
                    "signals": {
                        "latency": float(item.get("latency_ms", 0) or 0),
                        "errors": float(item.get("error_count", 0) or 0),
                        "cost": float(item.get("cost", 0) or 0),
                    },
                    "confidence": float(item.get("confidence", 0.8) or 0.8),
                    "ttl": int(item.get("ttl", 3600) or 3600),
                }
            )
        return out


class RelevanceGateAbility(Ability):
    def __init__(self) -> None:
        super().__init__(
            name="telemetry_relevance_gate",
            description="Filter telemetry items by task relevance",
        )
        self.template = (
            f"{SYSTEM_PREFIX}\n\nFrom the telemetry items, keep only "
            "those that could change the outcome of the current task. "
            "Discard routine noise. For each kept item, add a 0-1 "
            "relevance score. Output NDJSON."
        )

    @tool
    async def execute(
        self, *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        task: str = kwargs.get("task") or (args[0] if args else "")
        items: list[dict[str, Any]] = kwargs.get("items") or (
            args[1] if len(args) > 1 else []
        )
        llm_provider = kwargs.get("llm_provider") or (
            args[2] if len(args) > 2 else None
        )
        if not items:
            return []
        if llm_provider is None:  # fallback path
            for it in items:
                it.setdefault("relevance", 0.5)
            return items
        prompt = (
            f"{self.template}\n\nTask: {task}\nItems:\n"
            + json.dumps(items, indent=2)
            + "\n\nFormat: {\"id\":...,\"keep\":true,"
            "\"relevance\":0.87,\"reason\":...}"
        )
        try:
            response = await cast(LLMProvider, llm_provider).generate(prompt)
        except (RuntimeError, ValueError, TypeError):
            for it in items:
                it.setdefault("relevance", 0.5)
            return items
        kept: list[dict[str, Any]] = []
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("keep") and obj.get("id"):
                target = next(
                    (i for i in items if i["id"] == obj["id"]), None
                )
                if target:
                    target["relevance"] = float(obj.get("relevance", 0.5))
                    target["relevance_reason"] = obj.get("reason", "")
                    kept.append(target)
        if not kept:  # heuristic fallback
            kept = sorted(
                items,
                key=lambda x: x.get("signals", {}).get("latency", 0),
                reverse=True,
            )[: min(5, len(items))]
            for k in kept:
                k.setdefault("relevance", 0.4)
        return kept


class RankAbility(Ability):
    def __init__(self) -> None:
        super().__init__(
            name="telemetry_rank",
            description="Rank telemetry by relevance, recency, confidence, impact",
        )

    @staticmethod
    def _recency_score(ts_str: str) -> float:
        if not ts_str:
            return 0.5
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            age = (datetime.now(UTC) - dt).total_seconds()
            return max(0.1, min(1.0, 1 - age / 86400))
        except (ValueError, TypeError):
            return 0.5

    @tool
    async def execute(
        self, *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = kwargs.get("items") or (
            args[1] if len(args) > 1 else []
        )
        top_n: int = int(
            kwargs.get("top_n", (args[2] if len(args) > 2 else 200))
        )
        if not items:
            return []
        for it in items:
            relevance = float(it.get("relevance", 0.5))
            recency = self._recency_score(str(it.get("ts", "")))
            confidence = float(it.get("confidence", 0.8))
            sig = it.get("signals", {})
            impact = (sig.get("errors", 0) or 0) * 0.1 + (
                sig.get("latency", 0) or 0
            ) / 1000.0
            it["rank_score"] = (
                relevance * 0.4
                + recency * 0.2
                + confidence * 0.2
                + min(impact, 1.0) * 0.2
            )
        items.sort(key=lambda x: x.get("rank_score", 0), reverse=True)
        return items[:top_n]


class ClusterAbility(Ability):
    def __init__(self) -> None:
        super().__init__(
            name="telemetry_cluster",
            description="Cluster similar telemetry items",
        )
        self.template = (
            f"{SYSTEM_PREFIX}\n\nCluster items by semantic similarity and "
            "overlapping facets. Return JSON array of clusters with: "
            "cluster_id, topic, members, conflicts, summary (<=80 tokens)."
        )

    @tool
    async def execute(
        self, *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = kwargs.get("items") or (
            args[0] if args else []
        )
        llm_provider = kwargs.get("llm_provider") or (
            args[1] if len(args) > 1 else None
        )
        if not items:
            return []
        if llm_provider is None:
            return [
                {
                    "cluster_id": "C1",
                    "topic": "aggregate",
                    "members": items,
                    "conflicts": [],
                    "summary": ", ".join(
                        it.get("text", "") for it in items
                    )[:300],
                }
            ]
        prompt = f"{self.template}\nItems:\n" + json.dumps(items, indent=2)
        try:
            response = await cast(LLMProvider, llm_provider).generate(prompt)
            clusters = json.loads(response)
            if isinstance(clusters, list):
                return clusters
        except (json.JSONDecodeError, ValueError, TypeError, RuntimeError):
            clusters = []
        return [
            {
                "cluster_id": "C1",
                "topic": "aggregate",
                "members": items,
                "conflicts": [],
                "summary": ", ".join(it.get("text", "") for it in items)[:300],
            }
        ]


class PruneAbility(Ability):
    def __init__(self) -> None:
        super().__init__(
            name="telemetry_prune",
            description="Prune content to fit token budget",
        )

    @tool
    async def execute(
        self, *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        clusters: list[dict[str, Any]] = kwargs.get("clusters") or (
            args[0] if args else []
        )
        token_budget: int = int(
            kwargs.get("token_budget", (args[1] if len(args) > 1 else 2000))
        )
        if not clusters:
            return []
        for cl in clusters:
            cl["total_rank"] = sum(
                m.get("rank_score", 0) for m in cl.get("members", [])
            )
        clusters.sort(key=lambda c: c.get("total_rank", 0), reverse=True)
        used = 0
        pruned: list[dict[str, Any]] = []
        for cl in clusters:
            est = len(cl.get("summary", "")) // 4
            if used + est > token_budget:
                break
            pruned.append(cl)
            used += est
        return pruned


class FinalPromptAssemblerAbility(Ability):
    def __init__(self) -> None:
        super().__init__(
            name="telemetry_final_prompt",
            description="Assemble final reasoning prompt",
        )

    @tool
    async def execute(self, *args: Any, **kwargs: Any) -> str:
        task: str = kwargs.get("task") or (args[0] if args else "")
        clusters: list[dict[str, Any]] = kwargs.get("clusters") or (
            args[1] if len(args) > 1 else []
        )
        conflicts: list[dict[str, Any]] = kwargs.get("conflicts") or (
            args[2] if len(args) > 2 else []
        )
        constraints: list[str] = kwargs.get("constraints") or (
            args[3] if len(args) > 3 else []
        )
        facts: list[str] = []
        for cl in clusters:
            summary = cl.get("summary", "")
            ids = [m.get("id", "") for m in cl.get("members", [])]
            facts.append(f"- {summary} [{','.join(ids)}]")
        conflict_lines = [
            f"- {c.get('resolution','')} [{','.join(c.get('sources', []))}]"
            for c in conflicts
        ]
        prompt = (
            f"# Task\n{task}\n\n# Critical facts\n"
            + ("\n".join(facts) if facts else "- None")
            + "\n\n# Recent changes (24h)\n- No recent changes\n"
            + "\n# Conflicts & stance\n"
            + (
                "\n".join(conflict_lines)
                if conflict_lines
                else "- No conflicts detected"
            )
            + "\n\n# Hard constraints\n"
            + (
                "\n".join(f"- {c}" for c in constraints)
                if constraints
                else "- None"
            )
            + "\n\n# Your job\nAnalyze the above information and provide "
            "actionable recommendations.\n"
        )
        return prompt


__all__ = [
    "IngestNormalizeAbility",
    "RelevanceGateAbility",
    "RankAbility",
    "ClusterAbility",
    "PruneAbility",
    "FinalPromptAssemblerAbility",
]
