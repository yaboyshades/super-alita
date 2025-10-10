"""Evaluation framework for telemetry pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PipelineMetrics:
    precision: float
    recall: float
    redundancy_rate: float
    token_efficiency: float
    conflict_coverage: float


class PipelineEvaluator:
    """Evaluate pipeline performance."""

    def __init__(self, golden_sets_path: str | Path | None = None) -> None:
        self.golden_sets: dict[str, Any] = {}
        if golden_sets_path:
            golden_path = Path(golden_sets_path)
            if golden_path.exists():
                with golden_path.open(encoding="utf-8") as f:
                    self.golden_sets = json.load(f)

    def evaluate(
        self, pipeline_output: str, golden_facts: list[str]
    ) -> PipelineMetrics:
        """Compare pipeline output against golden facts."""

        # Extract facts from output
        extracted_facts = self._extract_facts(pipeline_output)

        # Calculate metrics
        precision = self._calculate_precision(extracted_facts, golden_facts)
        recall = self._calculate_recall(extracted_facts, golden_facts)
        redundancy = self._calculate_redundancy(extracted_facts)
        efficiency = self._calculate_efficiency(
            extracted_facts, pipeline_output
        )
        conflicts = self._calculate_conflict_coverage(pipeline_output)

        return PipelineMetrics(
            precision=precision,
            recall=recall,
            redundancy_rate=redundancy,
            token_efficiency=efficiency,
            conflict_coverage=conflicts,
        )

    def _extract_facts(self, output: str) -> list[str]:
        """Extract facts from pipeline output."""
        facts = []
        for line in output.split("\n"):
            if line.strip().startswith("- "):
                fact = line.strip()[2:]  # Remove "- "
                if "[" in fact:  # Has source IDs
                    fact = fact.split("[")[0].strip()
                if fact:
                    facts.append(fact)
        return facts

    def _calculate_precision(
        self, extracted: list[str], golden: list[str]
    ) -> float:
        """Calculate precision: correct facts / total extracted facts."""
        if not extracted:
            return 0.0
        correct = sum(
            1 for fact in extracted if any(g in fact for g in golden)
        )
        return correct / len(extracted)

    def _calculate_recall(
        self, extracted: list[str], golden: list[str]
    ) -> float:
        """Calculate recall: correct facts / total golden facts."""
        if not golden:
            return 1.0
        covered = sum(
            1 for g in golden if any(g in fact for fact in extracted)
        )
        return covered / len(golden)

    def _calculate_redundancy(self, extracted: list[str]) -> float:
        """Calculate redundancy rate."""
        if len(extracted) <= 1:
            return 0.0
        unique_facts = set(extracted)
        return 1.0 - (len(unique_facts) / len(extracted))

    def _calculate_efficiency(
        self, extracted: list[str], full_output: str
    ) -> float:
        """Calculate token efficiency."""
        fact_tokens = sum(len(fact.split()) for fact in extracted)
        total_tokens = len(full_output.split())
        return fact_tokens / total_tokens if total_tokens > 0 else 0.0

    def _calculate_conflict_coverage(self, output: str) -> float:
        """Calculate conflict detection coverage."""
        conflict_section = "# Conflicts & stance"
        if conflict_section in output:
            conflict_content = output.split(conflict_section)[1].split("#")[0]
            return (
                1.0 if "No conflicts detected" not in conflict_content else 0.0
            )
        return 0.0


__all__ = ["PipelineEvaluator", "PipelineMetrics"]
