"""Consolidation pipeline for upgrading episodic memories."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

from src.models import ConsolidationBatch, Memory
from src.stores.episodic import episodic_store
from src.stores.semantic import semantic_store


def run_consolidation() -> Dict[str, any]:
    print("🔄 Starting consolidation job...")
    candidates = _get_consolidation_candidates()
    if not candidates:
        return {"consolidated": 0, "message": "No candidates found"}

    clusters = _cluster_memories(candidates)
    consolidated = 0
    for cluster in clusters:
        if _consolidate_cluster(cluster):
            consolidated += 1
    print(f"✅ Consolidated {consolidated} clusters")
    return {
        "consolidated": consolidated,
        "clusters_processed": len(clusters),
        "timestamp": datetime.utcnow().isoformat(),
    }


def run_ace_informed_consolidation() -> Dict[str, Any]:
    print("🔁 Starting ACE-informed consolidation...")
    candidates = _get_ace_consolidation_candidates()
    if not candidates:
        return {
            "consolidated": 0,
            "clusters_processed": 0,
            "strategies": [],
            "message": "No ACE candidates found",
            "timestamp": datetime.utcnow().isoformat(),
        }

    clusters = _ace_context_clustering(candidates)
    strategies: List[Dict[str, Any]] = []
    consolidated = 0

    for cluster in clusters:
        strategy = _select_consolidation_strategy(cluster)
        success = _consolidate_cluster(cluster)
        strategies.append(
            {
                "strategy": strategy,
                "cluster_size": len(cluster),
                "succeeded": success,
            }
        )
        if success:
            consolidated += 1

    return {
        "consolidated": consolidated,
        "clusters_processed": len(clusters),
        "strategies": strategies,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _get_consolidation_candidates() -> List[Memory]:
    candidates = episodic_store.get_by_importance(min_importance=0.6)
    cutoff = datetime.utcnow() - timedelta(days=1)
    candidates = [memory for memory in candidates if memory.ts < cutoff]
    candidates.sort(
        key=lambda memory: memory.importance * _recency_score(memory.last_access),
        reverse=True,
    )
    return candidates[:200]


def _get_ace_consolidation_candidates() -> List[Memory]:
    baseline = episodic_store.get_by_importance(min_importance=0.4)
    enriched: List[Memory] = []
    for memory in baseline:
        if memory.importance >= 0.7:
            enriched.append(memory)
            continue
        meta = memory.meta if isinstance(memory.meta, dict) else {}
        contradiction_count = int(meta.get("contradiction_count", 0) or 0)
        clarity = float(meta.get("clarity_score", memory.confidence))
        if contradiction_count > 0 or clarity < 0.6:
            enriched.append(memory)
    enriched.sort(
        key=lambda memory: (
            memory.importance,
            -int(memory.meta.get("contradiction_count", 0)) if isinstance(memory.meta, dict) else 0,
            _recency_score(memory.last_access),
        ),
        reverse=True,
    )
    return enriched[:250]


def _recency_score(last_access: datetime) -> float:
    days_since_access = (datetime.utcnow() - last_access).days
    return max(0.1, 1.0 - days_since_access / 30.0)


def _cluster_memories(memories: List[Memory]) -> List[List[Memory]]:
    if not memories:
        return []
    clusters: List[List[Memory]] = [[memories[0]]]
    for memory in memories[1:]:
        placed = False
        for cluster in clusters:
            if _cluster_similarity(memory, cluster) > 0.7:
                cluster.append(memory)
                placed = True
                break
        if not placed:
            clusters.append([memory])
    return [cluster for cluster in clusters if len(cluster) >= 2]


def _ace_context_clustering(memories: List[Memory]) -> List[List[Memory]]:
    if not memories:
        return []
    buckets: Dict[str, List[Memory]] = {}
    for memory in memories:
        meta = memory.meta if isinstance(memory.meta, dict) else {}
        stance = meta.get("stance", "support")
        topic = meta.get("topic") or (memory.tags[0] if memory.tags else "general")
        contradiction_bucket = "contradiction" if meta.get("contradiction_count") else "support"
        key = f"{topic}:{stance}:{contradiction_bucket}"
        buckets.setdefault(key, []).append(memory)
    return [cluster for cluster in buckets.values() if len(cluster) >= 1]


def _cluster_similarity(memory: Memory, cluster: List[Memory]) -> float:
    if not cluster:
        return 0.0
    if memory.embeddings and all(member.embeddings for member in cluster):
        centroid = _average_embedding([member.embeddings for member in cluster if member.embeddings])
        return _cosine_similarity(memory.embeddings, centroid)
    cluster_text = " ".join(member.text for member in cluster)
    return _text_overlap(memory.text, cluster_text)


def _consolidate_cluster(cluster: List[Memory]) -> bool:
    try:
        summary = _generate_summary(cluster)
        if not summary:
            return False
        confidence = _calculate_confidence(cluster)
        if confidence < 0.3:
            return False

        semantic_memory = Memory(
            text=summary,
            kind="semantic",
            importance=min(0.9, max(memory.importance for memory in cluster) + 0.1),
            confidence=confidence,
            tags=list({tag for memory in cluster for tag in memory.tags}),
            source="consolidation",
            meta={
                "consolidated_from": [memory.id for memory in cluster],
                "cluster_size": len(cluster),
                "first_seen": min(memory.ts for memory in cluster).isoformat(),
                "last_seen": max(memory.last_access for memory in cluster).isoformat(),
            },
        )

        batch = ConsolidationBatch(
            episodic_ids=[memory.id for memory in cluster],
            semantic_id=semantic_memory.id,
            summary=summary,
            confidence=confidence,
            evidence_count=len(cluster),
        )

        semantic_store.add(semantic_memory)
        semantic_store.record_consolidation(batch)

        for memory in cluster:
            memory.importance *= 0.5
        return True
    except Exception as exc:  # pragma: no cover - defensive
        print(f"❌ Consolidation failed: {exc}")
        return False


def _select_consolidation_strategy(cluster: List[Memory]) -> str:
    meta_values = [memory.meta if isinstance(memory.meta, dict) else {} for memory in cluster]
    contradiction_total = sum(int(meta.get("contradiction_count", 0) or 0) for meta in meta_values)
    avg_confidence = sum(memory.confidence for memory in cluster) / len(cluster)
    if contradiction_total:
        return "counterexample-first"
    if avg_confidence < 0.5:
        return "evidence-expansion"
    return "standard"


def _generate_summary(cluster: List[Memory]) -> str:
    if len(cluster) == 1:
        return f"Remembered: {cluster[0].text}"
    primary = cluster[0].text
    if len(primary) > 100:
        primary = primary[:100] + "..."
    return f"Consolidated from {len(cluster)} memories: {primary}"


def _calculate_confidence(cluster: List[Memory]) -> float:
    if len(cluster) < 2:
        return 0.3
    base = min(0.8, 0.3 + len(cluster) * 0.1)
    avg_importance = sum(memory.importance for memory in cluster) / len(cluster)
    importance_boost = avg_importance * 0.2
    tag_consistency = _tag_consistency(cluster)
    return min(0.95, base + importance_boost + tag_consistency)


def _tag_consistency(cluster: List[Memory]) -> float:
    if not cluster or not cluster[0].tags:
        return 0.0
    all_tags = set(cluster[0].tags)
    for memory in cluster[1:]:
        all_tags |= set(memory.tags)
    common = set(cluster[0].tags)
    for memory in cluster[1:]:
        common &= set(memory.tags)
    return len(common) / len(all_tags) if all_tags else 0.0


def _average_embedding(embeddings: List[List[float]]) -> List[float]:
    if not embeddings:
        return []
    dimension = len(embeddings[0])
    avg = [0.0] * dimension
    for embedding in embeddings:
        for idx, value in enumerate(embedding):
            avg[idx] += value
    return [value / len(embeddings) for value in avg]


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if not norm1 or not norm2:
        return 0.0
    return dot / (norm1 * norm2)


def _text_overlap(text1: str, text2: str) -> float:
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union else 0.0
