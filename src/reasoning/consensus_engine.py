"""Multi-Model Consensus Engine with parallel reasoning and uncertainty.

Coordinates multiple LLMs (GPT-5, Claude, local models) for:
- Parallel reasoning with uncertainty quantification
- Consensus merging via voting/weighted averaging
- Constitutional compliance scoring across models
- Uncertainty-driven tie-breaking
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


class ModelProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT5 = "openai_gpt5"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    LOCAL_LLAMA = "local_llama"
    LOCAL_MIXTRAL = "local_mixtral"


class ConsensusMethod(str, Enum):
    """Methods for combining model outputs."""

    SIMPLE_VOTE = "simple_vote"  # Majority vote
    WEIGHTED_VOTE = "weighted_vote"  # Weight by confidence
    UNCERTAINTY_WEIGHTED = (
        "uncertainty_weighted"  # Weight by inverse uncertainty
    )
    CONSTITUTIONAL_WEIGHTED = "constitutional_weighted"  # Weight by compliance
    ENSEMBLE_RANKING = "ensemble_ranking"  # Rank aggregation


@dataclass
class ModelResponse:
    """Response from a single model."""

    model: ModelProvider
    response: str
    confidence: float = 0.0  # 0.0-1.0
    uncertainty: float = 1.0  # 0.0-1.0 (lower is better)
    constitutional_score: float = 0.0  # 0.0-1.0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ConsensusResult:
    """Result of consensus across multiple models."""

    consensus: str  # Final consensus response
    method: ConsensusMethod
    confidence: float  # Overall confidence in consensus
    uncertainty: float  # Overall uncertainty
    constitutional_score: float  # Overall constitutional compliance
    responses: list[ModelResponse] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiModelConsensusEngine:
    """Parallel reasoning across multiple LLMs with consensus merging.

    Coordinates parallel queries to multiple models and combines results
    using constitutional compliance, confidence, and uncertainty metrics.
    """

    def __init__(
        self,
        model_clients: dict[ModelProvider, Any] | None = None,
        constitutional_validator: Callable[[str], float] | None = None,
        default_method: ConsensusMethod = ConsensusMethod.WEIGHTED_VOTE,
    ):
        """Initialize the consensus engine.

        Args:
            model_clients: Dictionary of provider -> client instances
            constitutional_validator: Function to score constitutional compliance
            default_method: Default consensus method
        """
        self.model_clients = model_clients or {}
        self.constitutional_validator = constitutional_validator
        self.default_method = default_method

        # Metrics
        self.query_count = 0
        self.consensus_history: list[ConsensusResult] = []

    async def query_model(
        self,
        model: ModelProvider,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> ModelResponse:
        """Query a single model.

        Args:
            model: Model provider to query
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            ModelResponse with result and metadata
        """
        client = self.model_clients.get(model)
        if not client:
            # Stub response for missing client
            return ModelResponse(
                model=model,
                response=f"[Stub response from {model}]",
                confidence=0.5,
                uncertainty=0.5,
            )

        # Simulate query (replace with real client calls)
        start = asyncio.get_event_loop().time()
        response_text = f"Response from {model}: {prompt[:50]}..."
        latency_ms = (asyncio.get_event_loop().time() - start) * 1000

        # Estimate confidence/uncertainty (real impl uses model logprobs)
        confidence = 0.75 + (hash(prompt) % 100) / 400  # 0.75-1.0
        uncertainty = 1.0 - confidence

        # Constitutional scoring
        constitutional_score = 0.0
        if self.constitutional_validator:
            constitutional_score = await asyncio.to_thread(
                self.constitutional_validator, response_text
            )

        return ModelResponse(
            model=model,
            response=response_text,
            confidence=confidence,
            uncertainty=uncertainty,
            constitutional_score=constitutional_score,
            latency_ms=latency_ms,
        )

    async def query_all(
        self,
        prompt: str,
        models: list[ModelProvider] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> list[ModelResponse]:
        """Query all models in parallel.

        Args:
            prompt: Input prompt
            models: List of models to query (all if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            List of ModelResponse objects
        """
        models = models or list(self.model_clients.keys())
        if not models:
            models = [ModelProvider.OPENAI_GPT4]  # Fallback

        # Parallel queries
        tasks = [
            self.query_model(model, prompt, temperature, max_tokens)
            for model in models
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=False)

        self.query_count += 1
        return responses

    async def compute_consensus(
        self,
        responses: list[ModelResponse],
        method: ConsensusMethod | None = None,
    ) -> ConsensusResult:
        """Compute consensus from multiple model responses.

        Args:
            responses: List of model responses
            method: Consensus method (uses default if None)

        Returns:
            ConsensusResult with merged output
        """
        method = method or self.default_method

        if not responses:
            return ConsensusResult(
                consensus="",
                method=method,
                confidence=0.0,
                uncertainty=1.0,
                constitutional_score=0.0,
            )

        if method == ConsensusMethod.SIMPLE_VOTE:
            result = self._simple_vote(responses)
        elif method == ConsensusMethod.WEIGHTED_VOTE:
            result = self._weighted_vote(responses)
        elif method == ConsensusMethod.UNCERTAINTY_WEIGHTED:
            result = self._uncertainty_weighted(responses)
        elif method == ConsensusMethod.CONSTITUTIONAL_WEIGHTED:
            result = self._constitutional_weighted(responses)
        elif method == ConsensusMethod.ENSEMBLE_RANKING:
            result = self._ensemble_ranking(responses)
        else:
            result = self._weighted_vote(responses)

        self.consensus_history.append(result)
        return result

    def _simple_vote(self, responses: list[ModelResponse]) -> ConsensusResult:
        """Majority vote consensus."""
        # Group by response hash
        vote_map: dict[str, list[ModelResponse]] = defaultdict(list)
        for resp in responses:
            key = hashlib.sha256(resp.response.encode()).hexdigest()[:8]
            vote_map[key].append(resp)

        # Winner by count
        winner = max(vote_map.values(), key=len)
        consensus_resp = winner[0].response

        avg_confidence = sum(r.confidence for r in winner) / len(winner)
        avg_uncertainty = sum(r.uncertainty for r in winner) / len(winner)
        avg_constitutional = sum(r.constitutional_score for r in winner) / len(
            winner
        )

        return ConsensusResult(
            consensus=consensus_resp,
            method=ConsensusMethod.SIMPLE_VOTE,
            confidence=avg_confidence,
            uncertainty=avg_uncertainty,
            constitutional_score=avg_constitutional,
            responses=responses,
            metadata={"votes": len(winner), "total": len(responses)},
        )

    def _weighted_vote(
        self, responses: list[ModelResponse]
    ) -> ConsensusResult:
        """Weighted vote by confidence."""
        if not responses:
            return ConsensusResult(
                consensus="",
                method=ConsensusMethod.WEIGHTED_VOTE,
                confidence=0.0,
                uncertainty=1.0,
                constitutional_score=0.0,
            )

        # Weight by confidence
        total_weight = sum(r.confidence for r in responses)
        if total_weight == 0:
            return self._simple_vote(responses)

        # Choose highest weighted response
        best = max(responses, key=lambda r: r.confidence)

        # Weighted averages
        avg_confidence = (
            sum(r.confidence * r.confidence for r in responses) / total_weight
        )
        avg_uncertainty = (
            sum(r.confidence * r.uncertainty for r in responses) / total_weight
        )
        avg_constitutional = (
            sum(r.confidence * r.constitutional_score for r in responses)
            / total_weight
        )

        return ConsensusResult(
            consensus=best.response,
            method=ConsensusMethod.WEIGHTED_VOTE,
            confidence=avg_confidence,
            uncertainty=avg_uncertainty,
            constitutional_score=avg_constitutional,
            responses=responses,
        )

    def _uncertainty_weighted(
        self, responses: list[ModelResponse]
    ) -> ConsensusResult:
        """Weight by inverse uncertainty (prefer confident models)."""
        if not responses:
            return ConsensusResult(
                consensus="",
                method=ConsensusMethod.UNCERTAINTY_WEIGHTED,
                confidence=0.0,
                uncertainty=1.0,
                constitutional_score=0.0,
            )

        # Weight = 1 / (uncertainty + epsilon)
        epsilon = 0.01
        weights = [1.0 / (r.uncertainty + epsilon) for r in responses]
        total_weight = sum(weights)

        if total_weight == 0:
            return self._simple_vote(responses)

        # Choose lowest uncertainty response
        best = min(responses, key=lambda r: r.uncertainty)

        # Weighted averages
        avg_confidence = (
            sum(
                w * r.confidence
                for w, r in zip(weights, responses, strict=False)
            )
            / total_weight
        )
        avg_uncertainty = (
            sum(
                w * r.uncertainty
                for w, r in zip(weights, responses, strict=False)
            )
            / total_weight
        )
        avg_constitutional = (
            sum(
                w * r.constitutional_score
                for w, r in zip(weights, responses, strict=False)
            )
            / total_weight
        )

        return ConsensusResult(
            consensus=best.response,
            method=ConsensusMethod.UNCERTAINTY_WEIGHTED,
            confidence=avg_confidence,
            uncertainty=avg_uncertainty,
            constitutional_score=avg_constitutional,
            responses=responses,
        )

    def _constitutional_weighted(
        self, responses: list[ModelResponse]
    ) -> ConsensusResult:
        """Weight by constitutional compliance."""
        if not responses:
            return ConsensusResult(
                consensus="",
                method=ConsensusMethod.CONSTITUTIONAL_WEIGHTED,
                confidence=0.0,
                uncertainty=1.0,
                constitutional_score=0.0,
            )

        # Weight by constitutional score
        total_weight = sum(r.constitutional_score for r in responses)
        if total_weight == 0:
            return self._weighted_vote(responses)

        # Choose highest constitutional compliance
        best = max(responses, key=lambda r: r.constitutional_score)

        # Weighted averages
        avg_confidence = (
            sum(r.constitutional_score * r.confidence for r in responses)
            / total_weight
        )
        avg_uncertainty = (
            sum(r.constitutional_score * r.uncertainty for r in responses)
            / total_weight
        )
        avg_constitutional = (
            sum(
                r.constitutional_score * r.constitutional_score
                for r in responses
            )
            / total_weight
        )

        return ConsensusResult(
            consensus=best.response,
            method=ConsensusMethod.CONSTITUTIONAL_WEIGHTED,
            confidence=avg_confidence,
            uncertainty=avg_uncertainty,
            constitutional_score=avg_constitutional,
            responses=responses,
        )

    def _ensemble_ranking(
        self, responses: list[ModelResponse]
    ) -> ConsensusResult:
        """Ensemble ranking: combine multiple scoring dimensions."""
        if not responses:
            return ConsensusResult(
                consensus="",
                method=ConsensusMethod.ENSEMBLE_RANKING,
                confidence=0.0,
                uncertainty=1.0,
                constitutional_score=0.0,
            )

        # Composite score: confidence + (1 - uncertainty) + constitutional
        scores = [
            r.confidence + (1.0 - r.uncertainty) + r.constitutional_score
            for r in responses
        ]

        best_idx = scores.index(max(scores))
        best = responses[best_idx]

        # Global averages
        avg_confidence = sum(r.confidence for r in responses) / len(responses)
        avg_uncertainty = sum(r.uncertainty for r in responses) / len(
            responses
        )
        avg_constitutional = sum(
            r.constitutional_score for r in responses
        ) / len(responses)

        return ConsensusResult(
            consensus=best.response,
            method=ConsensusMethod.ENSEMBLE_RANKING,
            confidence=avg_confidence,
            uncertainty=avg_uncertainty,
            constitutional_score=avg_constitutional,
            responses=responses,
            metadata={"composite_scores": scores},
        )

    def get_stats(self) -> dict[str, Any]:
        """Get consensus engine statistics."""
        if not self.consensus_history:
            return {
                "query_count": self.query_count,
                "consensus_count": 0,
            }

        avg_confidence = sum(
            r.confidence for r in self.consensus_history
        ) / len(self.consensus_history)
        avg_uncertainty = sum(
            r.uncertainty for r in self.consensus_history
        ) / len(self.consensus_history)
        avg_constitutional = sum(
            r.constitutional_score for r in self.consensus_history
        ) / len(self.consensus_history)

        method_counts = defaultdict(int)
        for result in self.consensus_history:
            method_counts[result.method] += 1

        return {
            "query_count": self.query_count,
            "consensus_count": len(self.consensus_history),
            "avg_confidence": avg_confidence,
            "avg_uncertainty": avg_uncertainty,
            "avg_constitutional_score": avg_constitutional,
            "methods_used": dict(method_counts),
        }


# Example usage
async def example_consensus() -> None:
    """Example demonstrating MultiModelConsensusEngine."""

    def stub_validator(text: str) -> float:
        """Stub constitutional validator."""
        return 0.8 + (hash(text) % 20) / 100  # 0.8-1.0

    engine = MultiModelConsensusEngine(
        constitutional_validator=stub_validator,
        default_method=ConsensusMethod.CONSTITUTIONAL_WEIGHTED,
    )

    # Query all models
    prompt = "Design a secure authentication system with JWT tokens."
    responses = await engine.query_all(prompt)

    print(f"Queried {len(responses)} models")
    for resp in responses:
        print(
            f"  {resp.model}: confidence={resp.confidence:.2f}, "
            f"uncertainty={resp.uncertainty:.2f}, "
            f"constitutional={resp.constitutional_score:.2f}"
        )

    # Compute consensus with different methods
    for method in ConsensusMethod:
        result = await engine.compute_consensus(responses, method)
        print(f"\n{method}:")
        print(f"  Consensus: {result.consensus[:100]}...")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Uncertainty: {result.uncertainty:.2f}")
        print(f"  Constitutional: {result.constitutional_score:.2f}")

    # Stats
    print("\nEngine Stats:")
    print(json.dumps(engine.get_stats(), indent=2))


if __name__ == "__main__":
    asyncio.run(example_consensus())
