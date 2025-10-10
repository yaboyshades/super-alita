"""Energy calculation engine for task prioritization."""

import math
import time
from dataclasses import dataclass, field
from typing import Any

from src.knowledge_graph import KnowledgeGraphInterface
from src.ladder.graph.task_graph import Task


@dataclass
class EnergyMetrics:
    """Metrics used in energy calculation."""

    effort_score: float = 0.0  # Estimated effort (0-1, higher = more effort)
    success_probability: float = 0.5  # Historical success rate (0-1)
    pattern_confidence: float = 0.0  # Confidence in pattern match (0-1)
    complexity_score: float = 0.0  # Task complexity (0-1)
    dependency_score: float = 0.0  # Dependency complexity (0-1)
    recency_bonus: float = 0.0  # Bonus for recent successful patterns (0-1)
    context_relevance: float = 0.0  # Relevance to current context (0-1)


@dataclass
class TaskEnergy:
    """Energy calculation result for a task."""

    task_id: str
    energy_score: float  # Final energy score (lower = higher priority)
    confidence: float  # Confidence in the energy calculation
    metrics: EnergyMetrics = field(default_factory=EnergyMetrics)
    reasoning: list[str] = field(default_factory=list)
    calculated_at: float = field(default_factory=time.time)


class EnergyCalculator:
    """
    Calculates task energy for prioritization.

    Energy represents the "cost" of a task considering:
    - Historical success rates from KG patterns
    - Task complexity and dependencies
    - Context relevance and recency
    - Estimated effort vs. success probability

    Lower energy = Higher priority
    """

    def __init__(
        self,
        kg_interface: KnowledgeGraphInterface,
        effort_weight: float = 0.3,
        success_weight: float = 0.4,
        complexity_weight: float = 0.2,
        context_weight: float = 0.1,
    ):
        """Initialize the energy calculator.

        Args:
            kg_interface: Knowledge graph for historical data
            effort_weight: Weight for effort in energy calculation
            success_weight: Weight for success probability
            complexity_weight: Weight for complexity
            context_weight: Weight for context relevance
        """
        self.kg_interface = kg_interface
        self.effort_weight = effort_weight
        self.success_weight = success_weight
        self.complexity_weight = complexity_weight
        self.context_weight = context_weight

        # Ensure weights sum to 1.0
        total_weight = (
            effort_weight + success_weight + complexity_weight + context_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    def calculate_task_energy(
        self, task: Task, context: dict[str, Any] | None = None
    ) -> TaskEnergy:
        """Calculate energy for a single task.

        Args:
            task: Task to calculate energy for
            context: Planning context

        Returns:
            TaskEnergy with score and metrics
        """
        context = context or {}
        metrics = EnergyMetrics()
        reasoning = []

        # 1. Calculate effort score based on task description and complexity
        metrics.effort_score = self._calculate_effort_score(task, reasoning)

        # 2. Calculate success probability from KG patterns
        metrics.success_probability = self._calculate_success_probability(
            task, context, reasoning
        )

        # 3. Calculate pattern confidence
        metrics.pattern_confidence = self._calculate_pattern_confidence(
            task, context, reasoning
        )

        # 4. Calculate complexity score
        metrics.complexity_score = self._calculate_complexity_score(
            task, reasoning
        )

        # 5. Calculate dependency score
        metrics.dependency_score = self._calculate_dependency_score(
            task, reasoning
        )

        # 6. Calculate recency bonus
        metrics.recency_bonus = self._calculate_recency_bonus(
            task, context, reasoning
        )

        # 7. Calculate context relevance
        metrics.context_relevance = self._calculate_context_relevance(
            task, context, reasoning
        )

        # 8. Combine into final energy score
        energy_score = self._combine_energy_metrics(metrics, reasoning)

        # 9. Calculate confidence in the energy calculation
        confidence = self._calculate_confidence(metrics)

        return TaskEnergy(
            task_id=task.id,
            energy_score=energy_score,
            confidence=confidence,
            metrics=metrics,
            reasoning=reasoning,
        )

    def _calculate_effort_score(
        self, task: Task, reasoning: list[str]
    ) -> float:
        """Calculate effort score based on task characteristics."""
        effort = 0.5  # Default medium effort

        # Check task description for effort indicators
        description = task.description.lower()

        # High effort keywords
        high_effort_keywords = [
            "implement",
            "create",
            "build",
            "develop",
            "design",
            "complex",
            "comprehensive",
            "full",
            "complete",
            "entire",
        ]

        # Low effort keywords
        low_effort_keywords = [
            "check",
            "verify",
            "test",
            "validate",
            "review",
            "simple",
            "quick",
            "basic",
            "minimal",
            "small",
        ]

        high_count = sum(1 for kw in high_effort_keywords if kw in description)
        low_count = sum(1 for kw in low_effort_keywords if kw in description)

        if high_count > low_count:
            effort = min(0.9, 0.5 + (high_count - low_count) * 0.1)
            reasoning.append(
                f"High effort task detected ({high_count} indicators)"
            )
        elif low_count > high_count:
            effort = max(0.1, 0.5 - (low_count - high_count) * 0.1)
            reasoning.append(
                f"Low effort task detected ({low_count} indicators)"
            )

        return effort

    def _calculate_success_probability(
        self, task: Task, context: dict[str, Any], reasoning: list[str]
    ) -> float:
        """Calculate success probability from KG patterns."""
        try:
            # Import KnowledgeQuery here to avoid circular imports
            from src.knowledge_graph.models import KnowledgeQuery

            # Create query for relevant patterns
            query = KnowledgeQuery(
                goal=task.description,
                domain=context.get("domain", "general"),
                context=context,
                include_patterns=True,
                max_results=5,
                min_confidence=0.3,
            )

            result = self.kg_interface.query(query)

            if not result.patterns:
                reasoning.append(
                    "No historical patterns found, using default probability"
                )
                return 0.5

            # Calculate weighted success rate
            success_rates = [
                pattern.success_rate for pattern in result.patterns
            ]
            avg_success = sum(success_rates) / len(success_rates)

            reasoning.append(
                f"Success probability {avg_success:.2f} from {len(result.patterns)} patterns"
            )
            return max(0.1, min(0.95, avg_success))

        except Exception as e:
            reasoning.append(
                f"Error calculating success probability: {str(e)}"
            )
            return 0.5  # Safe default

    def _calculate_pattern_confidence(
        self, task: Task, context: dict[str, Any], reasoning: list[str]
    ) -> float:
        """Calculate confidence in pattern matching."""
        try:
            # Import KnowledgeQuery here to avoid circular imports
            from src.knowledge_graph.models import KnowledgeQuery

            # Create query for relevant patterns
            query = KnowledgeQuery(
                goal=task.description,
                domain=context.get("domain", "general"),
                context=context,
                include_patterns=True,
                max_results=10,
                min_confidence=0.2,
            )

            result = self.kg_interface.query(query)
            patterns = result.patterns

            if not patterns:
                return 0.0

            # Calculate confidence based on pattern relevance and count
            max_relevance = max(
                result.relevance_scores.get(p.pattern_name, 0.0)
                for p in patterns
            )
            # More patterns = higher confidence, diminishing returns
            pattern_count_factor = min(1.0, len(patterns) / 3.0)

            confidence = (max_relevance + pattern_count_factor) / 2.0
            reasoning.append(
                f"Pattern confidence {confidence:.2f} from "
                f"{len(patterns)} patterns"
            )
            return confidence

        except Exception:
            return 0.0

    def _calculate_complexity_score(
        self, task: Task, reasoning: list[str]
    ) -> float:
        """Calculate task complexity score."""
        complexity = 0.5  # Default medium complexity

        # Factors that increase complexity
        description_length = len(task.description)
        if description_length > 200:
            complexity += 0.2
            reasoning.append("Complex task (long description)")

        # Check for complexity keywords
        complex_keywords = [
            "multiple",
            "several",
            "various",
            "many",
            "complex",
            "sophisticated",
            "advanced",
            "integrate",
            "coordinate",
        ]

        desc_lower = task.description.lower()
        complex_count = sum(1 for kw in complex_keywords if kw in desc_lower)
        if complex_count > 0:
            complexity += min(0.3, complex_count * 0.1)
            reasoning.append(
                f"Complexity increased by {complex_count} indicators"
            )

        return min(1.0, complexity)

    def _calculate_dependency_score(
        self, task: Task, reasoning: list[str]
    ) -> float:
        """Calculate dependency complexity score."""
        # For now, simple dependency count
        dependency_count = (
            len(task.dependencies) if hasattr(task, "dependencies") else 0
        )

        if dependency_count == 0:
            score = 0.0
        elif dependency_count <= 2:
            score = 0.3
        elif dependency_count <= 5:
            score = 0.6
        else:
            score = 0.9

        if dependency_count > 0:
            reasoning.append(
                f"Dependency score {score:.1f} from {dependency_count} dependencies"
            )

        return score

    def _calculate_recency_bonus(
        self, task: Task, context: dict[str, Any], reasoning: list[str]
    ) -> float:
        """Calculate bonus for recently successful patterns."""
        try:
            # Import KnowledgeQuery here to avoid circular imports
            from src.knowledge_graph.models import KnowledgeQuery

            # Create query for relevant patterns
            query = KnowledgeQuery(
                goal=task.description,
                domain=context.get("domain", "general"),
                context=context,
                include_patterns=True,
                max_results=5,
                min_confidence=0.3,
            )

            result = self.kg_interface.query(query)
            patterns = result.patterns

            if not patterns:
                return 0.0

            current_time = time.time()
            max_bonus = 0.0

            for pattern in patterns:
                # For now, use creation time as success time
                last_success = getattr(pattern, "last_success_time", 0)
                if last_success > 0:
                    # Calculate time decay (bonus decreases over time)
                    time_diff = current_time - last_success
                    days_ago = time_diff / (24 * 3600)

                    # Exponential decay: full if used today, half after 7 days
                    bonus = math.exp(-days_ago / 7.0) * pattern.success_rate
                    max_bonus = max(max_bonus, bonus)

            if max_bonus > 0.1:
                reasoning.append(
                    f"Recency bonus {max_bonus:.2f} from recent successes"
                )

            return max_bonus

        except Exception:
            return 0.0

    def _calculate_context_relevance(
        self, task: Task, context: dict[str, Any], reasoning: list[str]
    ) -> float:
        """Calculate relevance to current context."""
        relevance = 0.5  # Default medium relevance

        # Check domain match
        current_domain = context.get("domain", "general")
        task_domain = self._extract_task_domain(task)

        if task_domain == current_domain:
            relevance += 0.3
            reasoning.append(f"Domain match bonus: {current_domain}")

        # Check for context keywords in task
        context_keywords = context.get("keywords", [])
        if context_keywords:
            desc_lower = task.description.lower()
            matches = sum(
                1 for kw in context_keywords if kw.lower() in desc_lower
            )
            if matches > 0:
                keyword_bonus = min(0.2, matches * 0.05)
                relevance += keyword_bonus
                reasoning.append(
                    f"Keyword relevance bonus: {keyword_bonus:.2f}"
                )

        return min(1.0, relevance)

    def _extract_task_domain(self, task: Task) -> str:
        """Extract domain from task description."""
        desc_lower = task.description.lower()

        # Simple domain detection
        if any(
            kw in desc_lower
            for kw in ["code", "program", "function", "class", "method"]
        ):
            return "software_development"
        elif any(
            kw in desc_lower
            for kw in ["research", "analyze", "study", "paper"]
        ):
            return "research"
        else:
            return "general"

    def _combine_energy_metrics(
        self, metrics: EnergyMetrics, reasoning: list[str]
    ) -> float:
        """Combine metrics into final energy score."""
        # Energy formula: lower is better (higher priority)
        # effort and complexity increase energy
        # success_probability and bonuses decrease energy

        base_energy = (
            self.effort_weight * metrics.effort_score
            + self.complexity_weight * metrics.complexity_score
        )

        # Success factors reduce energy (higher success = lower energy)
        success_reduction = (
            self.success_weight * metrics.success_probability
            + self.context_weight * metrics.context_relevance
        )

        # Apply recency bonus (reduces energy)
        recency_reduction = metrics.recency_bonus * 0.1

        # Dependency penalty (increases energy)
        dependency_penalty = metrics.dependency_score * 0.1

        energy = (
            base_energy
            - success_reduction
            - recency_reduction
            + dependency_penalty
        )

        # Normalize to 0-1 range
        energy = max(0.0, min(1.0, energy))

        reasoning.append(
            f"Final energy: {energy:.3f} (lower = higher priority)"
        )
        return energy

    def _calculate_confidence(self, metrics: EnergyMetrics) -> float:
        """Calculate confidence in the energy calculation."""
        # Confidence based on available data quality
        factors = [
            metrics.pattern_confidence * 0.4,  # Pattern quality most important
            (1.0 if metrics.success_probability != 0.5 else 0.0)
            * 0.3,  # Historical data
            (1.0 if metrics.recency_bonus > 0 else 0.0) * 0.2,  # Recent data
            0.1,  # Base confidence for calculated metrics
        ]

        return sum(factors)
