"""Evaluation harness for synthetic curricula."""
from __future__ import annotations

import statistics
import time
from datetime import datetime
from typing import Any, Dict, List

from src.app import capture_messages, get_context
from src.controller.consolidate import run_consolidation
from src.models import Memory, Message, Role
from src.stores.episodic import episodic_store
from src.stores.semantic import semantic_store


class EvaluationHarness:
    def run_curriculum(self, curriculum: Dict[str, Any]) -> Dict[str, Any]:
        print(f"🏃 Running curriculum: {curriculum['name']}")
        self._seed_memories(curriculum["seed_facts"])
        baseline_accuracy = self._measure_retention(curriculum["seed_facts"])
        self._seed_memories(curriculum["new_facts"])
        immediate_accuracy = self._measure_retention(
            curriculum["seed_facts"] + curriculum["new_facts"]
        )
        consolidation_result = run_consolidation()
        post_accuracy = self._measure_retention(curriculum["seed_facts"] + curriculum["new_facts"])
        stability = baseline_accuracy["seed"] - post_accuracy["seed"]
        plasticity = post_accuracy["new"]
        return {
            "curriculum": curriculum["name"],
            "stability": max(0, stability),
            "plasticity": plasticity,
            "consolidation_effectiveness": consolidation_result.get("consolidated", 0),
            "memory_counts": {
                "episodic": episodic_store.count(),
                "semantic": semantic_store.count(),
            },
            "timestamps": {"end": datetime.utcnow().isoformat()},
        }

    def _seed_memories(self, facts: List[Dict[str, str]]) -> None:
        messages = [
            Message(role=Role.USER, content=fact["content"], meta={"category": fact.get("category", "general")})
            for fact in facts
        ]
        capture_messages(messages)

    def _measure_retention(self, facts: List[Dict[str, str]]) -> Dict[str, float]:
        seed_correct = 0
        new_correct = 0
        total_seed = 0
        total_new = 0
        for fact in facts:
            context = get_context(fact["content"], k=5, budget=200)
            if fact["content"].lower() in context.text.lower():
                if fact.get("type") == "seed":
                    seed_correct += 1
                else:
                    new_correct += 1
            if fact.get("type") == "seed":
                total_seed += 1
            else:
                total_new += 1
        return {
            "seed": seed_correct / total_seed if total_seed else 0.0,
            "new": new_correct / total_new if total_new else 0.0,
            "overall": (seed_correct + new_correct)
            / (total_seed + total_new)
            if (total_seed + total_new)
            else 0.0,
        }

    def run_performance_benchmark(self) -> Dict[str, Any]:
        print("⏱️  Running performance benchmarks...")
        ingest_times: List[float] = []
        for i in range(50):
            start = time.time()
            episodic_store.add(
                Memory(text=f"Test memory {i} for benchmarking", importance=0.5)
            )
            ingest_times.append(time.time() - start)
        retrieval_times: List[float] = []
        for i in range(25):
            start = time.time()
            episodic_store.search(f"test memory {i}", k=10)
            retrieval_times.append(time.time() - start)
        consolidation_start = time.time()
        run_consolidation()
        consolidation_time = time.time() - consolidation_start
        return {
            "ingest_performance": {
                "p50": statistics.median(ingest_times),
                "p95": sorted(ingest_times)[int(len(ingest_times) * 0.95)],
                "ops_per_second": 1 / statistics.mean(ingest_times),
            },
            "retrieval_performance": {
                "p50": statistics.median(retrieval_times),
                "p95": sorted(retrieval_times)[int(len(retrieval_times) * 0.95)],
                "queries_per_second": 1 / statistics.mean(retrieval_times),
            },
            "consolidation_performance": {
                "total_time": consolidation_time,
                "memories_per_second": episodic_store.count() / consolidation_time
                if consolidation_time
                else 0.0,
            },
            "memory_usage": {
                "episodic_size": episodic_store.count(),
                "semantic_size": semantic_store.count(),
            },
        }


PREFERENCES_CURRICULUM = {
    "name": "user_preferences_tracking",
    "seed_facts": [
        {"content": "I prefer dogs over cats", "type": "seed", "category": "pets"},
        {"content": "My favorite color is blue", "type": "seed", "category": "colors"},
        {"content": "I don't like spicy food", "type": "seed", "category": "food"},
    ],
    "new_facts": [
        {"content": "I enjoy hiking on weekends", "type": "new", "category": "hobbies"},
        {"content": "I prefer coffee in the morning", "type": "new", "category": "food"},
        {"content": "My favorite season is autumn", "type": "new", "category": "seasons"},
    ],
}
