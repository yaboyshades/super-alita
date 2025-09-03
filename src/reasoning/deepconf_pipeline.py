"""
Enhanced DeepConf Pipeline - Advanced consensus mechanisms
Implements offline/online consensus with confidence calibration
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from src.reasoning.deepconf_conf import MultiDomainConfidenceCalibrator


@dataclass
class ConsensusResult:
    """Result structure for consensus aggregation"""

    consensus_text: str
    confidence: float
    method_used: str
    individual_scores: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "consensus_text": self.consensus_text,
            "confidence": self.confidence,
            "method_used": self.method_used,
            "individual_scores": self.individual_scores,
            "metadata": self.metadata,
            "processing_time": self.processing_time,
        }


class AdvancedConsensusAggregator:
    """
    Advanced consensus mechanisms for response aggregation
    Implements multiple voting strategies with confidence weighting
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.method_stats = {}

    async def aggregate_responses(
        self,
        responses: list[dict[str, Any]],
        method: str = "weighted_vote",
        confidence_threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Aggregate multiple responses using specified consensus method
        """
        start_time = time.time()
        threshold = confidence_threshold or self.confidence_threshold

        if not responses:
            return {
                "consensus_text": "",
                "confidence": 0.0,
                "individual_scores": [],
                "method_used": method,
                "error": "No responses provided",
            }

        # Extract texts and confidence scores
        texts = [r.get("text", "") for r in responses]
        confidences = [r.get("confidence_score", 0.5) for r in responses]

        # Apply consensus method
        if method == "weighted_vote":
            result = await self._weighted_vote_consensus(texts, confidences, threshold)
        elif method == "borda_count":
            result = await self._borda_count_consensus(texts, confidences, threshold)
        elif method == "condorcet":
            result = await self._condorcet_consensus(texts, confidences, threshold)
        elif method == "confidence_ranking":
            result = await self._confidence_ranking_consensus(
                texts, confidences, threshold
            )
        elif method == "similarity_clustering":
            result = await self._similarity_clustering_consensus(
                texts, confidences, threshold
            )
        else:
            # Fallback to simple majority
            result = await self._simple_majority_consensus(
                texts, confidences, threshold
            )

        # Add metadata
        processing_time = time.time() - start_time
        result.update(
            {
                "method_used": method,
                "processing_time": processing_time,
                "individual_scores": confidences,
                "num_responses": len(responses),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        # Update method statistics
        if method not in self.method_stats:
            self.method_stats[method] = {
                "usage_count": 0,
                "avg_confidence": 0.0,
                "total_processing_time": 0.0,
            }

        stats = self.method_stats[method]
        stats["usage_count"] += 1
        stats["avg_confidence"] = (
            stats["avg_confidence"] * (stats["usage_count"] - 1) + result["confidence"]
        ) / stats["usage_count"]
        stats["total_processing_time"] += processing_time

        return result

    async def _weighted_vote_consensus(
        self, texts: list[str], confidences: list[float], threshold: float
    ) -> dict[str, Any]:
        """Weighted voting based on confidence scores"""
        if not texts:
            return {"consensus_text": "", "confidence": 0.0}

        # Create weighted frequency map
        text_weights = {}
        for text, conf in zip(texts, confidences, strict=False):
            text = text.strip()
            if text:
                text_weights[text] = text_weights.get(text, 0) + conf

        if not text_weights:
            return {"consensus_text": "", "confidence": 0.0}

        # Find text with highest weighted score
        best_text = max(text_weights, key=text_weights.get)
        total_weight = sum(text_weights.values())
        consensus_confidence = text_weights[best_text] / total_weight

        return {
            "consensus_text": best_text,
            "confidence": min(1.0, consensus_confidence),
            "metadata": {
                "text_weights": text_weights,
                "total_weight": total_weight,
                "unique_responses": len(text_weights),
            },
        }

    async def _borda_count_consensus(
        self, texts: list[str], confidences: list[float], threshold: float
    ) -> dict[str, Any]:
        """Borda count voting with confidence weighting"""
        if not texts:
            return {"consensus_text": "", "confidence": 0.0}

        # Create unique text list
        unique_texts = list(set(t.strip() for t in texts if t.strip()))
        if not unique_texts:
            return {"consensus_text": "", "confidence": 0.0}

        # Calculate Borda scores
        borda_scores = dict.fromkeys(unique_texts, 0.0)
        n = len(unique_texts)

        for text, conf in zip(texts, confidences, strict=False):
            text = text.strip()
            if text in unique_texts:
                # Assign Borda points weighted by confidence
                rank_points = n - 1  # Highest rank for exact match
                borda_scores[text] += rank_points * conf

                # Give partial points to similar texts (simplified)
                for other_text in unique_texts:
                    if (
                        other_text != text
                        and self._text_similarity(text, other_text) > 0.7
                    ):
                        similarity = self._text_similarity(text, other_text)
                        partial_points = (n - 2) * conf * similarity
                        borda_scores[other_text] += partial_points

        # Find winner
        if not any(borda_scores.values()):
            return {"consensus_text": "", "confidence": 0.0}

        best_text = max(borda_scores, key=borda_scores.get)
        max_possible_score = n * sum(confidences)
        consensus_confidence = borda_scores[best_text] / max(1.0, max_possible_score)

        return {
            "consensus_text": best_text,
            "confidence": min(1.0, consensus_confidence),
            "metadata": {
                "borda_scores": borda_scores,
                "max_possible_score": max_possible_score,
            },
        }

    async def _condorcet_consensus(
        self, texts: list[str], confidences: list[float], threshold: float
    ) -> dict[str, Any]:
        """Condorcet method for pairwise comparisons"""
        if not texts:
            return {"consensus_text": "", "confidence": 0.0}

        unique_texts = list(set(t.strip() for t in texts if t.strip()))
        if len(unique_texts) <= 1:
            return {
                "consensus_text": unique_texts[0] if unique_texts else "",
                "confidence": (
                    sum(confidences) / len(confidences) if confidences else 0.0
                ),
            }

        # Create pairwise comparison matrix
        n = len(unique_texts)
        comparison_matrix = np.zeros((n, n))

        for i, text_a in enumerate(unique_texts):
            for j, text_b in enumerate(unique_texts):
                if i != j:
                    # Count weighted preferences
                    score_a = 0.0
                    score_b = 0.0

                    for text, conf in zip(texts, confidences, strict=False):
                        text = text.strip()
                        sim_a = self._text_similarity(text, text_a)
                        sim_b = self._text_similarity(text, text_b)

                        if sim_a > sim_b:
                            score_a += conf
                        elif sim_b > sim_a:
                            score_b += conf

                    comparison_matrix[i][j] = score_a / (score_a + score_b + 1e-8)

        # Find Condorcet winner (beats all others in pairwise comparisons)
        condorcet_scores = []
        for i in range(n):
            wins = sum(1 for j in range(n) if i != j and comparison_matrix[i][j] > 0.5)
            condorcet_scores.append(wins)

        # Select winner
        best_idx = np.argmax(condorcet_scores)
        best_text = unique_texts[best_idx]

        # Calculate confidence based on dominance
        max_wins = max(condorcet_scores)
        consensus_confidence = max_wins / (n - 1) if n > 1 else 1.0

        return {
            "consensus_text": best_text,
            "confidence": consensus_confidence,
            "metadata": {
                "condorcet_scores": condorcet_scores,
                "comparison_matrix": comparison_matrix.tolist(),
                "total_comparisons": n * (n - 1),
            },
        }

    async def _confidence_ranking_consensus(
        self, texts: list[str], confidences: list[float], threshold: float
    ) -> dict[str, Any]:
        """Simple confidence-based ranking"""
        if not texts:
            return {"consensus_text": "", "confidence": 0.0}

        # Find text with highest individual confidence
        max_conf_idx = np.argmax(confidences)
        best_text = texts[max_conf_idx].strip()
        best_confidence = confidences[max_conf_idx]

        # Calculate consensus confidence (average of top texts)
        sorted_pairs = sorted(
            zip(texts, confidences, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )
        top_n = min(3, len(sorted_pairs))
        top_confidences = [pair[1] for pair in sorted_pairs[:top_n]]
        consensus_confidence = sum(top_confidences) / len(top_confidences)

        return {
            "consensus_text": best_text,
            "confidence": consensus_confidence,
            "metadata": {
                "individual_confidences": confidences,
                "best_individual_confidence": best_confidence,
                "top_n_average": consensus_confidence,
            },
        }

    async def _similarity_clustering_consensus(
        self, texts: list[str], confidences: list[float], threshold: float
    ) -> dict[str, Any]:
        """Cluster similar responses and select from largest cluster"""
        if not texts:
            return {"consensus_text": "", "confidence": 0.0}

        # Simple clustering based on text similarity
        clusters = []
        used_indices = set()

        for i, text_a in enumerate(texts):
            if i in used_indices:
                continue

            cluster = [(i, text_a, confidences[i])]
            used_indices.add(i)

            for j, text_b in enumerate(texts):
                if j in used_indices:
                    continue

                if self._text_similarity(text_a, text_b) > 0.6:
                    cluster.append((j, text_b, confidences[j]))
                    used_indices.add(j)

            clusters.append(cluster)

        if not clusters:
            return {"consensus_text": "", "confidence": 0.0}

        # Select cluster with highest total confidence
        cluster_scores = []
        for cluster in clusters:
            total_confidence = sum(item[2] for item in cluster)
            cluster_scores.append(total_confidence)

        best_cluster_idx = np.argmax(cluster_scores)
        best_cluster = clusters[best_cluster_idx]

        # Select representative text from best cluster (highest confidence)
        best_item = max(best_cluster, key=lambda x: x[2])
        consensus_text = best_item[1]

        # Calculate cluster confidence
        cluster_confidence = cluster_scores[best_cluster_idx] / len(best_cluster)

        return {
            "consensus_text": consensus_text,
            "confidence": min(1.0, cluster_confidence / len(texts)),
            "metadata": {
                "num_clusters": len(clusters),
                "cluster_sizes": [len(c) for c in clusters],
                "best_cluster_size": len(best_cluster),
                "cluster_confidences": cluster_scores,
            },
        }

    async def _simple_majority_consensus(
        self, texts: list[str], confidences: list[float], threshold: float
    ) -> dict[str, Any]:
        """Simple majority voting fallback"""
        if not texts:
            return {"consensus_text": "", "confidence": 0.0}

        # Count occurrences
        text_counts = {}
        text_confidences = {}

        for text, conf in zip(texts, confidences, strict=False):
            text = text.strip()
            if text:
                text_counts[text] = text_counts.get(text, 0) + 1
                if text not in text_confidences:
                    text_confidences[text] = []
                text_confidences[text].append(conf)

        if not text_counts:
            return {"consensus_text": "", "confidence": 0.0}

        # Find most frequent text
        best_text = max(text_counts, key=text_counts.get)
        occurrence_ratio = text_counts[best_text] / len(texts)
        avg_confidence = sum(text_confidences[best_text]) / len(
            text_confidences[best_text]
        )

        consensus_confidence = occurrence_ratio * avg_confidence

        return {
            "consensus_text": best_text,
            "confidence": consensus_confidence,
            "metadata": {
                "text_counts": text_counts,
                "occurrence_ratio": occurrence_ratio,
                "average_confidence": avg_confidence,
            },
        }

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity using character overlap"""
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        if t1 == t2:
            return 1.0

        # Simple character-based similarity (Jaccard)
        set1 = set(t1)
        set2 = set(t2)

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0


class EnhancedDeepConfPipeline:
    """
    Enhanced DeepConf pipeline with offline/online modes and caching
    Orchestrates consensus sampling with confidence calibration
    """

    def __init__(
        self,
        model_api,
        cache_size: int = 1000,
        enable_adaptive_sampling: bool = True,
        confidence_threshold: float = 0.7,
    ):
        self.model_api = model_api
        self.cache_size = cache_size
        self.enable_adaptive_sampling = enable_adaptive_sampling
        self.confidence_threshold = confidence_threshold

        # Initialize components
        self.consensus_aggregator = AdvancedConsensusAggregator(confidence_threshold)
        self.confidence_calibrator = MultiDomainConfidenceCalibrator()

        # Caching and performance tracking
        self.response_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.pipeline_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "cache_hit_rate": 0.0,
        }

    async def process_consensus_request(
        self,
        prompt: str,
        num_samples: int = 3,
        consensus_method: str = "weighted_vote",
        temperature: float = 0.7,
        max_tokens: int = 512,
        confidence_threshold: float | None = None,
        domain: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """
        Process a consensus sampling request with full pipeline
        """
        start_time = time.time()
        self.pipeline_stats["total_requests"] += 1

        try:
            # Check cache first
            cache_key = self._generate_cache_key(
                prompt, num_samples, consensus_method, temperature, max_tokens
            )

            if use_cache and cache_key in self.response_cache:
                self.cache_hits += 1
                cached_result = self.response_cache[cache_key]
                cached_result["metadata"]["from_cache"] = True
                cached_result["metadata"]["cache_timestamp"] = datetime.now(
                    UTC
                ).isoformat()
                return cached_result

            self.cache_misses += 1

            # Generate multiple samples
            samples = []
            generation_tasks = []

            for i in range(num_samples):
                # Vary temperature for diversity if adaptive sampling is enabled
                if self.enable_adaptive_sampling and num_samples > 1:
                    temp_variation = 0.1 * (i - num_samples // 2)
                    adjusted_temp = max(0.1, min(1.0, temperature + temp_variation))
                else:
                    adjusted_temp = temperature

                task = self.model_api.generate_with_logprobs(
                    prompt=prompt,
                    temperature=adjusted_temp,
                    max_tokens=max_tokens,
                )
                generation_tasks.append(task)

            # Execute generations concurrently
            generation_results = await asyncio.gather(
                *generation_tasks, return_exceptions=True
            )

            # Process results
            for i, result in enumerate(generation_results):
                if isinstance(result, Exception):
                    print(f"Generation {i} failed: {result}")
                    continue

                samples.append(
                    {
                        "text": result.text,
                        "confidence_score": result.confidence_score,
                        "logprobs": result.logprobs,
                        "generation_time": result.generation_time,
                        "metadata": result.metadata,
                    }
                )

            if not samples:
                raise Exception("All generation attempts failed")

            # Apply consensus mechanism
            consensus_result = await self.consensus_aggregator.aggregate_responses(
                samples,
                method=consensus_method,
                confidence_threshold=confidence_threshold or self.confidence_threshold,
            )

            # Apply confidence calibration if domain is specified
            if domain and self.confidence_calibrator:
                calibrated_confidence = (
                    await self.confidence_calibrator.calibrate_confidence(
                        [consensus_result["confidence"]], domain=domain
                    )
                )
                consensus_result["calibrated_confidence"] = calibrated_confidence[0]

            # Add pipeline metadata
            processing_time = time.time() - start_time
            consensus_result["metadata"].update(
                {
                    "pipeline_processing_time": processing_time,
                    "num_samples_generated": len(samples),
                    "num_samples_requested": num_samples,
                    "adaptive_sampling": self.enable_adaptive_sampling,
                    "domain": domain,
                    "pipeline_version": "enhanced_v1.0",
                }
            )

            # Cache result
            if use_cache and len(self.response_cache) < self.cache_size:
                self.response_cache[cache_key] = consensus_result.copy()

            # Update stats
            self.pipeline_stats["successful_requests"] += 1
            self.pipeline_stats["total_processing_time"] += processing_time
            self._update_cache_hit_rate()

            return consensus_result

        except Exception as e:
            self.pipeline_stats["failed_requests"] += 1
            processing_time = time.time() - start_time
            self.pipeline_stats["total_processing_time"] += processing_time

            return {
                "consensus_text": "",
                "confidence": 0.0,
                "error": str(e),
                "metadata": {
                    "pipeline_processing_time": processing_time,
                    "error_type": type(e).__name__,
                    "num_samples_requested": num_samples,
                },
            }

    def _generate_cache_key(
        self,
        prompt: str,
        num_samples: int,
        method: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate cache key for request parameters"""
        # Simple hash-based cache key
        import hashlib

        key_data = f"{prompt}:{num_samples}:{method}:{temperature}:{max_tokens}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _update_cache_hit_rate(self):
        """Update cache hit rate statistics"""
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests > 0:
            self.pipeline_stats["cache_hit_rate"] = (
                self.cache_hits / total_cache_requests
            )

    def get_pipeline_stats(self) -> dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        return {
            **self.pipeline_stats,
            "cache_stats": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_size": len(self.response_cache),
                "cache_capacity": self.cache_size,
            },
            "consensus_stats": self.consensus_aggregator.method_stats,
            "avg_processing_time": (
                self.pipeline_stats["total_processing_time"]
                / max(1, self.pipeline_stats["total_requests"])
            ),
            "success_rate": (
                self.pipeline_stats["successful_requests"]
                / max(1, self.pipeline_stats["total_requests"])
            ),
        }

    def clear_cache(self):
        """Clear the response cache"""
        self.response_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self._update_cache_hit_rate()


# Export main classes
__all__ = [
    "EnhancedDeepConfPipeline",
    "AdvancedConsensusAggregator",
    "ConsensusResult",
]
