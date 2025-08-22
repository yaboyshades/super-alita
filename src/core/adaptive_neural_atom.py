#!/usr/bin/env python3
"""
🧠 ADAPTIVE NEURAL ATOM - SELF-LEARNING INTELLIGENCE UNITS
Core adaptive learning capabilities for Super Alita's neural-symbolic architecture

AGENT DEV MODE (Copilot read this):
- Event-driven only; define Pydantic events (Literal event_type) and add to EVENT_TYPE_MAP
- Neural Atoms are concrete subclasses with UUIDv5 deterministic IDs
- Use logging.getLogger(__name__), never print. Clamp sizes/ranges
- Write tests: fixed inputs ⇒ fixed outputs; handler validation
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from src.core.neural_atom import NeuralAtom, NeuralAtomMetadata

logger = logging.getLogger(__name__)


@dataclass
class LearningMetrics:
    """Comprehensive learning metrics for adaptive behavior"""

    execution_count: int = 0
    success_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    confidence_history: list[float] = field(default_factory=list)
    performance_trend: float = 0.0
    adaptation_rate: float = 0.1
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionPattern:
    """Pattern recognition for similar execution contexts"""

    pattern_id: str
    input_signature: str
    execution_times: list[float] = field(default_factory=list)
    success_rates: list[bool] = field(default_factory=list)
    confidence_scores: list[float] = field(default_factory=list)
    frequency: int = 0
    last_seen: datetime = field(default_factory=datetime.now)


class AdaptiveNeuralAtom(NeuralAtom):
    """
    🧠 Base class for adaptive neural atoms with learning capabilities

    Extends NeuralAtom with:
    - Real-time performance learning
    - Pattern recognition and prediction
    - Adaptive confidence scoring
    - Self-optimization capabilities
    """

    def __init__(self, metadata: NeuralAtomMetadata):
        super().__init__(metadata)
        self.key = metadata.name  # Required for NeuralStore compatibility

        # Adaptive learning components
        self.learning_metrics = LearningMetrics()
        self.execution_patterns: dict[str, ExecutionPattern] = {}
        self.performance_history: deque = deque(maxlen=1000)

        # Learning parameters
        self.learning_rate = 0.1
        self.pattern_similarity_threshold = 0.8
        self.confidence_decay_rate = 0.95
        self.performance_window = 50

        logger.info(f"🧠 AdaptiveNeuralAtom initialized: {metadata.name}")

    async def adaptive_execute(self, input_data: Any) -> dict[str, Any]:
        """
        Execute with adaptive learning and performance tracking
        """
        start_time = time.time()

        # Pre-execution: Predict performance based on patterns
        prediction = await self._predict_performance(input_data)

        try:
            # Execute the core functionality
            result = await self.execute(input_data)

            execution_time = time.time() - start_time
            success = True

            # Post-execution: Learn from this execution
            await self._learn_from_execution(
                input_data, result, execution_time, success, prediction
            )

            return {
                "result": result,
                "success": True,
                "execution_time": execution_time,
                "predicted_time": prediction["expected_duration"],
                "confidence": prediction["confidence"],
                "performance": {
                    "efficiency": self._calculate_efficiency(
                        execution_time, prediction["expected_duration"]
                    ),
                    "accuracy": 1.0,  # Successful execution
                    "adaptation_score": self._calculate_adaptation_score(),
                    "execution_time": execution_time,
                    "confidence": prediction["confidence"],
                },
                "metadata": self.metadata,
                "learning_summary": self._get_learning_summary(),
            }

        except Exception as e:
            execution_time = time.time() - start_time

            # Learn from failure
            await self._learn_from_execution(
                input_data, None, execution_time, False, prediction
            )

            logger.error(f"🧠 Adaptive execution failed: {self.metadata.name} - {e}")

            return {
                "result": None,
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "predicted_time": prediction["expected_duration"],
                "confidence": prediction["confidence"],
                "performance": {
                    "efficiency": 0.0,
                    "accuracy": 0.0,
                    "adaptation_score": self._calculate_adaptation_score(),
                    "execution_time": execution_time,
                    "confidence": prediction["confidence"],
                },
                "metadata": self.metadata,
                "learning_summary": self._get_learning_summary(),
            }

    async def _predict_performance(self, input_data: Any) -> dict[str, Any]:
        """
        Predict execution performance based on learned patterns
        """
        input_signature = self._create_input_signature(input_data)

        # Find similar patterns
        similar_patterns = self._find_similar_patterns(input_signature)

        if similar_patterns:
            # Use historical data for prediction
            execution_times = []
            success_rates = []
            confidence_scores = []

            for pattern in similar_patterns:
                execution_times.extend(pattern.execution_times)
                success_rates.extend([1.0 if s else 0.0 for s in pattern.success_rates])
                confidence_scores.extend(pattern.confidence_scores)

            predicted_duration = np.mean(execution_times) if execution_times else 1.0
            predicted_success_rate = np.mean(success_rates) if success_rates else 0.5
            pattern_confidence = (
                np.mean(confidence_scores) if confidence_scores else 0.5
            )

            # Adjust confidence based on pattern frequency and recency
            frequency_bonus = min(0.2, len(similar_patterns) * 0.05)
            confidence = min(0.95, pattern_confidence + frequency_bonus)

        else:
            # No historical data - use conservative estimates
            predicted_duration = 1.0
            predicted_success_rate = 0.5
            confidence = 0.3

        return {
            "expected_duration": predicted_duration,
            "success_probability": predicted_success_rate,
            "confidence": confidence,
            "pattern_matches": len(similar_patterns),
            "input_signature": input_signature,
        }

    async def _learn_from_execution(
        self,
        input_data: Any,
        result: Any,
        execution_time: float,
        success: bool,
        prediction: dict[str, Any],
    ) -> None:
        """
        Learn from execution results to improve future predictions
        """
        # Update overall learning metrics
        self.learning_metrics.execution_count += 1
        if success:
            self.learning_metrics.success_count += 1

        self.learning_metrics.total_execution_time += execution_time
        self.learning_metrics.average_execution_time = (
            self.learning_metrics.total_execution_time
            / self.learning_metrics.execution_count
        )

        # Update confidence history
        actual_performance = 1.0 if success else 0.0
        prediction_accuracy = 1.0 - abs(
            prediction["success_probability"] - actual_performance
        )
        self.learning_metrics.confidence_history.append(prediction_accuracy)

        # Limit history size
        if len(self.learning_metrics.confidence_history) > 100:
            self.learning_metrics.confidence_history.pop(0)

        # Update performance trend
        if len(self.learning_metrics.confidence_history) >= 10:
            recent_performance = np.mean(self.learning_metrics.confidence_history[-10:])
            older_performance = (
                np.mean(self.learning_metrics.confidence_history[-20:-10])
                if len(self.learning_metrics.confidence_history) >= 20
                else recent_performance
            )
            self.learning_metrics.performance_trend = (
                recent_performance - older_performance
            )

        # Store in performance history
        self.performance_history.append(
            {
                "timestamp": datetime.now(),
                "input_signature": prediction["input_signature"],
                "execution_time": execution_time,
                "success": success,
                "predicted_duration": prediction["expected_duration"],
                "prediction_accuracy": prediction_accuracy,
                "confidence": prediction["confidence"],
            }
        )

        # Update or create execution pattern
        await self._update_execution_pattern(
            prediction["input_signature"],
            execution_time,
            success,
            prediction["confidence"],
        )

        # Adaptive learning rate adjustment
        if prediction_accuracy < 0.5:
            # Poor prediction - increase learning rate
            self.learning_metrics.adaptation_rate = min(
                0.3, self.learning_metrics.adaptation_rate * 1.1
            )
        else:
            # Good prediction - gradually decrease learning rate
            self.learning_metrics.adaptation_rate = max(
                0.05, self.learning_metrics.adaptation_rate * 0.99
            )

        self.learning_metrics.last_updated = datetime.now()

        logger.debug(
            f"🧠 Learned from execution: {self.metadata.name} - success={success}, time={execution_time:.3f}s, accuracy={prediction_accuracy:.3f}"
        )

    def _create_input_signature(self, input_data: Any) -> str:
        """Create a signature for input data to enable pattern matching"""
        if input_data is None:
            return "none"
        if isinstance(input_data, str):
            # Text-based signature
            return f"text_len{len(input_data)}_words{len(input_data.split())}_chars{hash(input_data) % 10000}"
        if isinstance(input_data, int | float):
            # Numeric signature
            return f"num_{type(input_data).__name__}_value{abs(input_data) % 1000}"
        if isinstance(input_data, list | tuple):
            # Collection signature
            return f"collection_{type(input_data).__name__}_len{len(input_data)}_hash{hash(str(input_data)) % 10000}"
        if isinstance(input_data, dict):
            # Dictionary signature
            return f"dict_keys{len(input_data)}_hash{hash(str(sorted(input_data.keys()))) % 10000}"
        # Generic signature
        return f"object_{type(input_data).__name__}_hash{hash(str(input_data)) % 10000}"

    def _find_similar_patterns(self, input_signature: str) -> list[ExecutionPattern]:
        """Find execution patterns similar to the given input signature"""
        target_pattern_id = self._create_pattern_id(input_signature)
        similar_patterns = []

        # Exact pattern match
        if target_pattern_id in self.execution_patterns:
            similar_patterns.append(self.execution_patterns[target_pattern_id])

        # Find patterns with similar signatures
        for pattern in self.execution_patterns.values():
            if pattern.pattern_id != target_pattern_id:
                similarity = self._calculate_signature_similarity(
                    input_signature, pattern.input_signature
                )
                if similarity >= self.pattern_similarity_threshold:
                    similar_patterns.append(pattern)

        # Sort by frequency and recency
        similar_patterns.sort(key=lambda p: (p.frequency, p.last_seen), reverse=True)

        return similar_patterns[:10]  # Return top 10 similar patterns

    def _create_pattern_id(self, input_signature: str) -> str:
        """Create a pattern ID for grouping similar inputs"""
        return f"pattern_{hash(input_signature) % 1000}"

    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between two input signatures"""
        if sig1 == sig2:
            return 1.0

        # Extract features from signatures
        features1 = self._extract_signature_features(sig1)
        features2 = self._extract_signature_features(sig2)

        # Calculate feature similarity
        common_features = set(features1.keys()) & set(features2.keys())
        if not common_features:
            return 0.0

        similarities = []
        for feature in common_features:
            val1, val2 = features1[feature], features2[feature]
            if isinstance(val1, int | float) and isinstance(val2, int | float):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2), 1)
                similarities.append(1.0 - abs(val1 - val2) / max_val)
            elif val1 == val2:
                similarities.append(1.0)
            else:
                similarities.append(0.0)

        return np.mean(similarities) if similarities else 0.0

    def _extract_signature_features(self, signature: str) -> dict[str, Any]:
        """Extract features from an input signature for similarity calculation"""
        features = {}

        # Parse signature components
        if "text_len" in signature:
            parts = signature.split("_")
            for part in parts:
                if part.startswith("len"):
                    features["length"] = int(part[3:])
                elif part.startswith("words"):
                    features["words"] = int(part[5:])
                elif part.startswith("chars"):
                    features["chars"] = int(part[5:])
        elif "num_" in signature:
            parts = signature.split("_")
            for part in parts:
                if part.startswith("value"):
                    features["value"] = int(part[5:])
        elif "collection_" in signature:
            parts = signature.split("_")
            for part in parts:
                if part.startswith("len"):
                    features["length"] = int(part[3:])
        elif "dict_" in signature:
            parts = signature.split("_")
            for part in parts:
                if part.startswith("keys"):
                    features["keys"] = int(part[4:])

        return features

    def _calculate_efficiency(self, actual_time: float, predicted_time: float) -> float:
        """Calculate execution efficiency compared to prediction"""
        if predicted_time <= 0:
            return 1.0 if actual_time <= 1.0 else 0.5

        # Efficiency = how much better/worse we did compared to prediction
        if actual_time <= predicted_time:
            # Faster than predicted - efficiency > 1.0
            return min(2.0, predicted_time / max(actual_time, 0.01))
        # Slower than predicted - efficiency < 1.0
        return max(0.1, predicted_time / actual_time)

    def _calculate_adaptation_score(self) -> float:
        """Calculate how well the atom is adapting and learning"""
        if self.learning_metrics.execution_count < 5:
            return 0.5  # Not enough data

        # Components of adaptation score
        success_rate = (
            self.learning_metrics.success_count / self.learning_metrics.execution_count
        )

        confidence_improvement = 0.0
        if len(self.learning_metrics.confidence_history) >= 10:
            recent_confidence = np.mean(self.learning_metrics.confidence_history[-5:])
            older_confidence = np.mean(self.learning_metrics.confidence_history[-10:-5])
            confidence_improvement = max(0.0, recent_confidence - older_confidence)

        performance_trend_score = max(
            0.0, min(1.0, self.learning_metrics.performance_trend + 0.5)
        )

        pattern_diversity = min(1.0, len(self.execution_patterns) / 50.0)

        # Weighted combination
        adaptation_score = (
            success_rate * 0.4
            + confidence_improvement * 0.3
            + performance_trend_score * 0.2
            + pattern_diversity * 0.1
        )

        return max(0.0, min(1.0, adaptation_score))

    def _get_learning_summary(self) -> dict[str, Any]:
        """Get comprehensive learning summary"""
        return {
            "total_executions": self.learning_metrics.execution_count,
            "success_rate": self.learning_metrics.success_count
            / max(self.learning_metrics.execution_count, 1),
            "average_execution_time": self.learning_metrics.average_execution_time,
            "performance_trend": self.learning_metrics.performance_trend,
            "adaptation_rate": self.learning_metrics.adaptation_rate,
            "patterns_learned": len(self.execution_patterns),
            "adaptation_score": self._calculate_adaptation_score(),
            "confidence_trend": np.mean(self.learning_metrics.confidence_history[-10:])
            if len(self.learning_metrics.confidence_history) >= 10
            else 0.5,
            "last_updated": self.learning_metrics.last_updated.isoformat(),
        }

    async def _update_execution_pattern(
        self,
        input_signature: str,
        execution_time: float,
        success: bool,
        confidence: float,
    ) -> None:
        """Update or create execution patterns for this input type"""
        pattern_id = self._create_pattern_id(input_signature)

        if pattern_id in self.execution_patterns:
            # Update existing pattern
            pattern = self.execution_patterns[pattern_id]
            pattern.execution_times.append(execution_time)
            pattern.success_rates.append(success)
            pattern.confidence_scores.append(confidence)
            pattern.frequency += 1
            pattern.last_seen = datetime.now()

            # Limit pattern history
            max_pattern_history = 100
            if len(pattern.execution_times) > max_pattern_history:
                pattern.execution_times.pop(0)
                pattern.success_rates.pop(0)
                pattern.confidence_scores.pop(0)
        else:
            # Create new pattern
            self.execution_patterns[pattern_id] = ExecutionPattern(
                pattern_id=pattern_id,
                input_signature=input_signature,
                execution_times=[execution_time],
                success_rates=[success],
                confidence_scores=[confidence],
                frequency=1,
                last_seen=datetime.now(),
            )

        # Cleanup old patterns (keep only most recent 1000)
        if len(self.execution_patterns) > 1000:
            # Remove oldest patterns
            patterns_by_age = sorted(
                self.execution_patterns.items(), key=lambda x: x[1].last_seen
            )
            patterns_to_remove = patterns_by_age[:100]  # Remove oldest 100
            for pattern_id, _ in patterns_to_remove:
                del self.execution_patterns[pattern_id]

    def get_learning_summary(self) -> dict[str, Any]:
        """Public interface for learning summary"""
        return self._get_learning_summary()


# Concrete implementation examples
class AdaptiveTextProcessor(AdaptiveNeuralAtom):
    """🧠 Adaptive text processing neural atom with learning capabilities"""

    def __init__(self):
        metadata = NeuralAtomMetadata(
            name="adaptive_text_processor",
            description="Adaptive text processing with real-time learning",
            capabilities=[
                "text_processing",
                "adaptive_learning",
                "pattern_recognition",
            ],
            version="1.0.0",
        )
        super().__init__(metadata)

    async def execute(self, input_data: Any) -> Any:
        """Process text with adaptive learning"""
        if not isinstance(input_data, str):
            raise ValueError("AdaptiveTextProcessor requires string input")

        # Simulate text processing with variable complexity
        processing_time = len(input_data) * 0.001  # Simulate work
        await asyncio.sleep(processing_time)

        # Process the text
        result = {
            "processed_text": input_data.upper(),
            "word_count": len(input_data.split()),
            "character_count": len(input_data),
            "processing_complexity": min(1.0, len(input_data) / 1000.0),
        }

        return result

    def get_embedding(self) -> list[float]:
        """Generate semantic embedding for this text processor"""
        # Simple embedding based on capabilities and performance
        base_embedding = [0.8, 0.9, 0.7, 0.6]  # text, processing, adaptive, learning

        # Add performance-based features
        if self.learning_metrics.execution_count > 0:
            success_rate = (
                self.learning_metrics.success_count
                / self.learning_metrics.execution_count
            )
            adaptation_score = self._calculate_adaptation_score()
            performance_features = [
                success_rate,
                adaptation_score,
                self.learning_metrics.performance_trend + 0.5,
            ]
            base_embedding.extend(performance_features)

        # Pad to standard embedding size
        while len(base_embedding) < 384:
            base_embedding.append(0.0)

        return base_embedding[:384]

    def can_handle(self, task_description: str) -> float:
        """Adaptive confidence for handling text processing tasks"""
        base_confidence = 0.0

        # Check for text processing keywords
        text_keywords = ["text", "process", "analyze", "parse", "extract", "transform"]
        if any(keyword in task_description.lower() for keyword in text_keywords):
            base_confidence = 0.8

        # Adjust confidence based on learning
        if self.learning_metrics.execution_count > 10:
            # Use historical performance to adjust confidence
            success_rate = (
                self.learning_metrics.success_count
                / self.learning_metrics.execution_count
            )
            adaptation_bonus = self._calculate_adaptation_score() * 0.2

            adjusted_confidence = base_confidence * success_rate + adaptation_bonus
            return max(0.0, min(1.0, adjusted_confidence))

        return base_confidence


class AdaptiveMemoryAtom(AdaptiveNeuralAtom):
    """🧠 Adaptive memory storage with learning-based optimization"""

    def __init__(self, content: str):
        metadata = NeuralAtomMetadata(
            name=f"adaptive_memory_{uuid.uuid4().hex[:8]}",
            description="Adaptive memory storage with learning capabilities",
            capabilities=["memory_storage", "adaptive_retrieval", "pattern_learning"],
            version="1.0.0",
        )
        super().__init__(metadata)
        self.content = content
        self.retrieval_patterns: dict[str, int] = defaultdict(int)
        self.access_frequency = 0
        self.last_accessed = datetime.now()

    async def execute(self, input_data: Any = None) -> Any:
        """Retrieve memory content with adaptive access tracking"""
        self.access_frequency += 1
        self.last_accessed = datetime.now()

        # Track retrieval patterns
        if input_data and isinstance(input_data, str):
            pattern = self._extract_retrieval_pattern(input_data)
            self.retrieval_patterns[pattern] += 1

        # Simulate retrieval time based on content complexity and access patterns
        retrieval_complexity = len(self.content) / 10000.0
        frequency_bonus = min(
            0.5, self.access_frequency / 100.0
        )  # Faster for frequently accessed
        processing_time = max(0.01, retrieval_complexity - frequency_bonus)

        await asyncio.sleep(processing_time)

        return {
            "content": self.content,
            "access_frequency": self.access_frequency,
            "last_accessed": self.last_accessed.isoformat(),
            "retrieval_patterns": dict(self.retrieval_patterns),
            "memory_metadata": {
                "content_length": len(self.content),
                "retrieval_efficiency": 1.0 / max(processing_time, 0.01),
                "access_pattern_diversity": len(self.retrieval_patterns),
            },
        }

    def _extract_retrieval_pattern(self, query: str) -> str:
        """Extract pattern from retrieval query"""
        # Simple pattern extraction
        words = query.lower().split()
        if len(words) <= 2:
            return "simple_query"
        if len(words) <= 5:
            return "medium_query"
        return "complex_query"

    def get_embedding(self) -> list[float]:
        """Generate adaptive embedding based on content and usage patterns"""
        # Base content embedding (simplified)
        content_hash = hash(self.content) % 10000
        base_features = [
            len(self.content) / 10000.0,  # Content length
            self.access_frequency / 100.0,  # Access frequency
            len(self.retrieval_patterns) / 10.0,  # Pattern diversity
            (datetime.now() - self.last_accessed).total_seconds() / 86400.0,  # Recency
        ]

        # Add content hash features
        hash_features = [(content_hash >> i) & 1 for i in range(32)]

        # Combine features
        embedding = base_features + hash_features

        # Pad to standard size
        while len(embedding) < 384:
            embedding.append(0.0)

        return embedding[:384]

    def can_handle(self, task_description: str) -> float:
        """Adaptive confidence based on content relevance and access patterns"""
        base_confidence = 0.0

        # Check for memory-related keywords
        memory_keywords = [
            "remember",
            "recall",
            "retrieve",
            "memory",
            "stored",
            "saved",
        ]
        if any(keyword in task_description.lower() for keyword in memory_keywords):
            base_confidence = 0.9

        # Check for content relevance (simplified)
        query_words = set(task_description.lower().split())
        content_words = set(self.content.lower().split())

        if query_words & content_words:
            overlap_ratio = len(query_words & content_words) / len(
                query_words | content_words
            )
            base_confidence = max(base_confidence, overlap_ratio * 0.8)

        # Boost confidence based on access patterns
        if self.access_frequency > 5:
            frequency_boost = min(0.2, self.access_frequency / 50.0)
            base_confidence += frequency_boost

        return max(0.0, min(1.0, base_confidence))
