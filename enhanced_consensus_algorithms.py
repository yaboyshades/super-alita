#!/usr/bin/env python3
"""Enhanced consensus algorithms for Super Alita."""

import asyncio
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx


class ConsensusMethod(Enum):
    """Available consensus methods."""

    SIMPLE_VOTE = "simple_vote"
    WEIGHTED_VOTE = "weighted_vote"
    CONFIDENCE_BASED = "confidence_based"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    ENSEMBLE_RANKING = "ensemble_ranking"


@dataclass
class ConsensusResponse:
    """Structured consensus response."""

    consensus_text: str
    consensus_confidence: float
    aggregation_method: str
    individual_responses: list[str]
    confidence_scores: list[float]
    metadata: dict[str, Any]


class EnhancedConsensusProvider:
    """Enhanced consensus sampling with multiple aggregation methods."""

    def __init__(self, base_url: str, model_name: str, timeout: float = 60.0):
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout

    async def consensus_sampling(
        self,
        prompt: str,
        num_samples: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 512,
        method: ConsensusMethod = ConsensusMethod.SIMPLE_VOTE,
        confidence_threshold: float = 0.7,
        temperature_range: float = 0.2,
    ) -> dict[str, Any]:
        """Enhanced consensus sampling with multiple methods."""

        # Generate diverse responses
        responses_with_confidence = await self._generate_responses(
            prompt, num_samples, temperature, max_tokens, temperature_range
        )

        if not responses_with_confidence:
            return {
                "consensus_text": "Error: No valid responses generated",
                "consensus_confidence": 0.0,
                "aggregation_method": method.value,
                "individual_responses": [],
                "confidence_scores": [],
                "metadata": {"error": "No valid responses"},
            }

        responses, confidence_scores = zip(*responses_with_confidence, strict=False)

        # Apply consensus method
        if method == ConsensusMethod.SIMPLE_VOTE:
            consensus = self._simple_vote_consensus(responses)
        elif method == ConsensusMethod.WEIGHTED_VOTE:
            consensus = self._weighted_vote_consensus(responses, confidence_scores)
        elif method == ConsensusMethod.CONFIDENCE_BASED:
            consensus = self._confidence_based_consensus(
                responses, confidence_scores, confidence_threshold
            )
        elif method == ConsensusMethod.SEMANTIC_SIMILARITY:
            consensus = await self._semantic_similarity_consensus(responses)
        elif method == ConsensusMethod.ENSEMBLE_RANKING:
            consensus = self._ensemble_ranking_consensus(responses, confidence_scores)
        else:
            consensus = self._simple_vote_consensus(responses)

        return {
            "consensus_text": consensus.consensus_text,
            "consensus_confidence": consensus.consensus_confidence,
            "aggregation_method": consensus.aggregation_method,
            "individual_responses": list(responses),
            "confidence_scores": list(confidence_scores),
            "metadata": consensus.metadata,
        }

    async def _generate_responses(
        self,
        prompt: str,
        num_samples: int,
        base_temp: float,
        max_tokens: int,
        temp_range: float,
    ) -> list[tuple[str, float]]:
        """Generate diverse responses with confidence estimation."""

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = []

            for i in range(num_samples):
                # Create temperature diversity
                temp_offset = (i / max(1, num_samples - 1) - 0.5) * temp_range
                temperature = max(0.1, min(1.0, base_temp + temp_offset))

                task = self._single_request(client, prompt, temperature, max_tokens)
                tasks.append(task)

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            valid_responses = []
            for i, resp in enumerate(responses):
                if isinstance(resp, Exception):
                    continue

                try:
                    content, confidence = resp
                    valid_responses.append((content, confidence))
                except Exception:
                    continue

            return valid_responses

    async def _single_request(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, float]:
        """Make a single request and estimate confidence."""

        response = await client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )

        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()

        # Estimate confidence based on response characteristics
        confidence = self._estimate_confidence(content, temperature)

        return content, confidence

    def _estimate_confidence(self, content: str, temperature: float) -> float:
        """Estimate response confidence based on content characteristics."""
        confidence = 0.5  # Base confidence

        # Length factor - neither too short nor too long
        length_factor = min(1.0, len(content) / 100) * max(
            0.5, 1.0 - len(content) / 1000
        )
        confidence += length_factor * 0.2

        # Temperature factor - lower temperature = higher confidence
        temp_factor = 1.0 - temperature
        confidence += temp_factor * 0.2

        # Specificity indicators
        if any(
            word in content.lower() for word in ["specific", "exactly", "precisely"]
        ):
            confidence += 0.1

        # Uncertainty indicators
        if any(
            word in content.lower()
            for word in ["maybe", "perhaps", "might", "possibly"]
        ):
            confidence -= 0.1

        # Numeric answers tend to be more confident
        if re.search(r"\d+", content):
            confidence += 0.1

        return max(0.1, min(1.0, confidence))

    def _simple_vote_consensus(self, responses: tuple[str, ...]) -> ConsensusResponse:
        """Simple majority voting consensus."""
        response_counts = {}
        for resp in responses:
            response_counts[resp] = response_counts.get(resp, 0) + 1

        consensus_text = max(response_counts.keys(), key=lambda x: response_counts[x])
        consensus_confidence = response_counts[consensus_text] / len(responses)

        return ConsensusResponse(
            consensus_text=consensus_text,
            consensus_confidence=consensus_confidence,
            aggregation_method="simple_vote",
            individual_responses=list(responses),
            confidence_scores=[0.5] * len(responses),  # Default scores
            metadata={
                "num_responses": len(responses),
                "num_unique": len(response_counts),
                "vote_distribution": response_counts,
            },
        )

    def _weighted_vote_consensus(
        self, responses: tuple[str, ...], confidence_scores: tuple[float, ...]
    ) -> ConsensusResponse:
        """Weighted voting based on confidence scores."""
        weighted_counts = {}

        for resp, conf in zip(responses, confidence_scores, strict=False):
            if resp not in weighted_counts:
                weighted_counts[resp] = 0.0
            weighted_counts[resp] += conf

        consensus_text = max(weighted_counts.keys(), key=lambda x: weighted_counts[x])
        total_weight = sum(confidence_scores)
        consensus_confidence = (
            weighted_counts[consensus_text] / total_weight if total_weight > 0 else 0.0
        )

        return ConsensusResponse(
            consensus_text=consensus_text,
            consensus_confidence=consensus_confidence,
            aggregation_method="weighted_vote",
            individual_responses=list(responses),
            confidence_scores=list(confidence_scores),
            metadata={
                "num_responses": len(responses),
                "num_unique": len(weighted_counts),
                "weighted_distribution": weighted_counts,
                "total_weight": total_weight,
            },
        )

    def _confidence_based_consensus(
        self,
        responses: tuple[str, ...],
        confidence_scores: tuple[float, ...],
        threshold: float,
    ) -> ConsensusResponse:
        """Select highest confidence response above threshold."""

        # Filter responses by confidence threshold
        high_conf_responses = [
            (resp, conf)
            for resp, conf in zip(responses, confidence_scores, strict=False)
            if conf >= threshold
        ]

        if not high_conf_responses:
            # Fall back to highest confidence if none meet threshold
            max_idx = max(
                range(len(confidence_scores)), key=lambda i: confidence_scores[i]
            )
            consensus_text = responses[max_idx]
            consensus_confidence = confidence_scores[max_idx]
            fallback = True
        else:
            # Select highest confidence from qualified responses
            consensus_text, consensus_confidence = max(
                high_conf_responses, key=lambda x: x[1]
            )
            fallback = False

        return ConsensusResponse(
            consensus_text=consensus_text,
            consensus_confidence=consensus_confidence,
            aggregation_method="confidence_based",
            individual_responses=list(responses),
            confidence_scores=list(confidence_scores),
            metadata={
                "threshold": threshold,
                "qualified_responses": len(high_conf_responses),
                "fallback_used": fallback,
                "avg_confidence": sum(confidence_scores) / len(confidence_scores),
            },
        )

    async def _semantic_similarity_consensus(
        self, responses: tuple[str, ...]
    ) -> ConsensusResponse:
        """Consensus based on semantic similarity clustering."""
        # Simplified semantic similarity using word overlap
        # In production, would use embeddings/transformers

        similarity_matrix = []
        for i, resp1 in enumerate(responses):
            row = []
            words1 = set(resp1.lower().split())
            for j, resp2 in enumerate(responses):
                words2 = set(resp2.lower().split())
                # Jaccard similarity
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                similarity = intersection / union if union > 0 else 0.0
                row.append(similarity)
            similarity_matrix.append(row)

        # Find most similar to all others
        avg_similarities = [sum(row) / len(row) for row in similarity_matrix]

        consensus_idx = max(
            range(len(avg_similarities)), key=lambda i: avg_similarities[i]
        )
        consensus_text = responses[consensus_idx]
        consensus_confidence = avg_similarities[consensus_idx]

        return ConsensusResponse(
            consensus_text=consensus_text,
            consensus_confidence=consensus_confidence,
            aggregation_method="semantic_similarity",
            individual_responses=list(responses),
            confidence_scores=[0.5] * len(responses),
            metadata={
                "similarity_scores": avg_similarities,
                "selected_index": consensus_idx,
            },
        )

    def _ensemble_ranking_consensus(
        self, responses: tuple[str, ...], confidence_scores: tuple[float, ...]
    ) -> ConsensusResponse:
        """Ensemble ranking combining multiple factors."""

        # Score based on multiple factors
        ensemble_scores = []

        for i, (resp, conf) in enumerate(
            zip(responses, confidence_scores, strict=False)
        ):
            score = 0.0

            # Confidence component (40%)
            score += conf * 0.4

            # Length appropriateness (20%)
            length_score = min(1.0, len(resp) / 100) * max(0.5, 1.0 - len(resp) / 1000)
            score += length_score * 0.2

            # Specificity (20%)
            specificity = len(re.findall(r"\d+", resp)) * 0.1 + len(resp.split()) * 0.01
            score += min(1.0, specificity) * 0.2

            # Uniqueness bonus (20%)
            uniqueness = 1.0 - (list(responses).count(resp) - 1) * 0.2
            score += max(0.0, uniqueness) * 0.2

            ensemble_scores.append(score)

        consensus_idx = max(
            range(len(ensemble_scores)), key=lambda i: ensemble_scores[i]
        )
        consensus_text = responses[consensus_idx]
        consensus_confidence = ensemble_scores[consensus_idx]

        return ConsensusResponse(
            consensus_text=consensus_text,
            consensus_confidence=consensus_confidence,
            aggregation_method="ensemble_ranking",
            individual_responses=list(responses),
            confidence_scores=list(confidence_scores),
            metadata={
                "ensemble_scores": ensemble_scores,
                "selected_index": consensus_idx,
                "scoring_components": [
                    "confidence",
                    "length",
                    "specificity",
                    "uniqueness",
                ],
            },
        )


# Test function
async def test_enhanced_consensus():
    """Test the enhanced consensus algorithms."""
    print("🧪 Testing Enhanced Consensus Algorithms...")

    provider = EnhancedConsensusProvider(
        base_url="http://localhost:11434/v1", model_name="gpt-oss:20b", timeout=30.0
    )

    test_prompt = "What is the capital of Japan?"

    methods = [
        ConsensusMethod.SIMPLE_VOTE,
        ConsensusMethod.WEIGHTED_VOTE,
        ConsensusMethod.CONFIDENCE_BASED,
        ConsensusMethod.SEMANTIC_SIMILARITY,
        ConsensusMethod.ENSEMBLE_RANKING,
    ]

    for method in methods:
        print(f"\n🔍 Testing {method.value}...")
        try:
            result = await provider.consensus_sampling(
                prompt=test_prompt,
                num_samples=3,
                temperature=0.6,
                max_tokens=100,
                method=method,
            )

            print(f"✅ {method.value}:")
            print(f"   Consensus: {result['consensus_text']}")
            print(f"   Confidence: {result['consensus_confidence']:.3f}")
            print(f"   Responses: {len(result['individual_responses'])}")

        except Exception as e:
            print(f"❌ {method.value}: {e}")


if __name__ == "__main__":
    asyncio.run(test_enhanced_consensus())
