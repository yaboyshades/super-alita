"""Constitutional guardrail evaluation using similarity and rule matching."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

SentenceTransformer: Any | None = None
_SENTENCE_TRANSFORMERS_ERROR: Exception | None = None


__all__ = [
    "ConstitutionalPrinciple",
    "EvaluationResult",
    "ConstitutionalReasoner",
    "SimpleSentenceEncoder",
]


class SimpleSentenceEncoder:
    """Lightweight, deterministic embedding fallback.

    The fallback approximates semantic similarity using hashed token
    frequencies. It keeps the interface compatible with
    ``SentenceTransformer.encode`` so the rest of the runtime can operate
    without heavyweight ML dependencies.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)

        for row, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            if not tokens:
                continue

            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                # Spread token influence across the vector for stability.
                for offset in range(0, len(digest), 4):
                    bucket = int.from_bytes(
                        digest[offset : offset + 4], "big", signed=False
                    ) % self.dimensions
                    vectors[row, bucket] += 1.0

            norm = float(np.linalg.norm(vectors[row]))
            if norm > 0:
                vectors[row] /= norm

        return vectors


@dataclass(slots=True)
class ConstitutionalPrinciple:
    """Representation of a single constitutional principle."""

    identifier: str
    title: str
    description: str
    weight: float
    embedding: np.ndarray | None = None


@dataclass(slots=True)
class EvaluationResult:
    """Structured outcome of an evaluation run."""

    approved: bool
    reasoning: str
    confidence: float
    recommendations: list[str]
    violation_scores: dict[str, float]
    total_score: float


class ConstitutionalReasoner:
    """Evaluate actions against the repository constitution."""

    def __init__(
        self,
        constitution_path: str = ".github/CONSTITUTION.md",
        *,
        embedding_model: Any | None = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.embedding_model = embedding_model or self._load_default_encoder()
        self._embedding_dimension: int | None = None
        self.constitution_path = Path(constitution_path)
        self.constitution_principles = self._load_constitution(self.constitution_path)
        self.violation_patterns = self._load_violation_patterns()
        self.score_threshold = 70.0

    async def evaluate_action(
        self, action: dict[str, Any], context: dict[str, Any]
    ) -> tuple[bool, str]:
        """Evaluate an action and return approval plus reasoning."""

        try:
            action_text = self._extract_action_text(action, context)
            similarity_scores = self._calculate_similarity_scores(action_text)
            pattern_violations = self._check_pattern_violations(action_text)
            total_score = self._calculate_total_score(similarity_scores, pattern_violations)
            reasoning, recommendations = self._generate_reasoning(
                similarity_scores, pattern_violations, total_score
            )
            approved = total_score >= self.score_threshold
            confidence = min(1.0, max(0.0, total_score / 100.0))

            result = EvaluationResult(
                approved=approved,
                reasoning=reasoning,
                confidence=confidence,
                recommendations=recommendations,
                violation_scores=pattern_violations,
                total_score=total_score,
            )

            await self._log_evaluation(action, context, result)
            return approved, reasoning

        except Exception as exc:  # pragma: no cover - defensive guard
            self.logger.error("Constitutional evaluation failed: %s", exc)
            return False, f"Evaluation failed: {exc}"

    def _load_constitution(self, constitution_path: Path) -> list[ConstitutionalPrinciple]:
        if not constitution_path.exists():
            self.logger.warning(
                "Constitution file %s not found; using fallback principles", constitution_path
            )
            return self._fallback_principles()

        content = constitution_path.read_text(encoding="utf-8")
        sections = re.split(r"\n## ", content)
        principles: list[ConstitutionalPrinciple] = []

        for section in sections[1:]:
            lines = section.strip().splitlines()
            if not lines:
                continue
            title = lines[0].strip()
            description = "\n".join(lines[1:]).strip()
            identifier = re.sub(r"[^a-zA-Z0-9]", "_", title.lower())
            principle = ConstitutionalPrinciple(
                identifier=identifier,
                title=title,
                description=description,
                weight=1.0,
            )
            principle.embedding = self._encode_text(f"{title}: {description}")
            principles.append(principle)

        if not principles:
            self.logger.warning("No principles parsed; using fallback set")
            return self._fallback_principles()

        self.logger.info("Loaded %s constitutional principles", len(principles))
        return principles

    def _fallback_principles(self) -> list[ConstitutionalPrinciple]:
        defaults = [
            (
                "do_no_harm",
                "Do No Harm",
                "Never perform actions that could cause harm to users or systems.",
                1.0,
            ),
            (
                "security",
                "Security First",
                "Never execute arbitrary code or bypass security measures.",
                1.0,
            ),
            (
                "privacy",
                "Respect Privacy",
                "Protect user privacy and confidential information.",
                0.8,
            ),
        ]

        principles: list[ConstitutionalPrinciple] = []
        for identifier, title, description, weight in defaults:
            principle = ConstitutionalPrinciple(
                identifier=identifier,
                title=title,
                description=description,
                weight=weight,
            )
            principle.embedding = self._encode_text(f"{title}: {description}")
            principles.append(principle)
        return principles

    def _load_default_encoder(self) -> Any:
        """Load the preferred embedding backend with safe fallback."""

        use_sentence_transformers = os.getenv(
            "SUPER_ALITA_USE_SENTENCE_TRANSFORMERS", "false"
        ).lower() in {"1", "true", "yes", "on"}

        global SentenceTransformer, _SENTENCE_TRANSFORMERS_ERROR

        if use_sentence_transformers and SentenceTransformer is None and _SENTENCE_TRANSFORMERS_ERROR is None:
            try:  # pragma: no cover - optional dependency import
                from sentence_transformers import SentenceTransformer as _SentenceTransformer

                SentenceTransformer = _SentenceTransformer
            except Exception as exc:  # pragma: no cover - import guard
                _SENTENCE_TRANSFORMERS_ERROR = exc

        if use_sentence_transformers and SentenceTransformer is not None:
            try:
                return SentenceTransformer("all-MiniLM-L6-v2")  # type: ignore[call-arg]
            except Exception as exc:  # pragma: no cover - depends on optional deps
                self.logger.warning(
                    "Failed to initialize sentence-transformers backend; using fallback: %s",
                    exc,
                )
        elif use_sentence_transformers and _SENTENCE_TRANSFORMERS_ERROR is not None:
            self.logger.warning(
                "sentence_transformers unavailable, falling back to SimpleSentenceEncoder: %s",
                _SENTENCE_TRANSFORMERS_ERROR,
            )

        return SimpleSentenceEncoder()

    def _encode_text(self, text: str) -> np.ndarray:
        try:
            embedding = self.embedding_model.encode([text])[0]
        except Exception as exc:  # pragma: no cover - defensive guard
            self.logger.error("Embedding backend failed for text '%s': %s", text, exc)
            size = self._embedding_dimension or getattr(self.embedding_model, "dimensions", 1)
            return np.zeros(int(size), dtype=float)

        vector = np.asarray(embedding, dtype=float)
        if vector.ndim != 1:
            vector = vector.reshape(-1)
        self._embedding_dimension = vector.shape[0]
        return vector

    def _load_violation_patterns(self) -> dict[str, dict[str, Any]]:
        return {
            "code_execution": {
                "patterns": [
                    re.compile(r"exec\(.*\)", re.IGNORECASE),
                    re.compile(r"eval\(.*\)", re.IGNORECASE),
                    re.compile(r"subprocess\.[^)]*\)", re.IGNORECASE),
                    re.compile(r"os\.system\(.*\)", re.IGNORECASE),
                    re.compile(r"__import__\(.*\)", re.IGNORECASE),
                    re.compile(r"compile\(.*\)", re.IGNORECASE),
                ],
                "severity": -50.0,
                "description": "Attempted code execution",
            },
            "security_bypass": {
                "patterns": [
                    re.compile(r"bypass.*security", re.IGNORECASE),
                    re.compile(r"ignore.*permission", re.IGNORECASE),
                    re.compile(r"disable.*protection", re.IGNORECASE),
                    re.compile(r"unauthorized.*access", re.IGNORECASE),
                ],
                "severity": -40.0,
                "description": "Security bypass attempt",
            },
            "placeholder_code": {
                "patterns": [
                    re.compile(r"TODO.*security", re.IGNORECASE),
                    re.compile(r"FIXME.*later", re.IGNORECASE),
                    re.compile(r"XXX.*implement", re.IGNORECASE),
                    re.compile(r"pass.*#.*temp", re.IGNORECASE),
                ],
                "severity": -30.0,
                "description": "Placeholder security code",
            },
            "data_exfiltration": {
                "patterns": [
                    re.compile(r"send.*data.*external", re.IGNORECASE),
                    re.compile(r"upload.*sensitive", re.IGNORECASE),
                    re.compile(r"export.*confidential", re.IGNORECASE),
                    re.compile(r"leak.*information", re.IGNORECASE),
                ],
                "severity": -45.0,
                "description": "Potential data exfiltration",
            },
        }

    def _extract_action_text(self, action: dict[str, Any], context: dict[str, Any]) -> str:
        fragments: list[str] = []
        description = action.get("description")
        if description:
            fragments.append(str(description))
        code = action.get("code")
        if code:
            fragments.append(str(code))
        user_intent = context.get("user_intent")
        if user_intent:
            fragments.append(str(user_intent))
        additional = context.get("additional_context")
        if additional:
            fragments.append(str(additional))
        return " ".join(fragments)

    def _calculate_similarity_scores(self, action_text: str) -> dict[str, float]:
        if not action_text.strip():
            return {principle.identifier: 0.0 for principle in self.constitution_principles}

        action_embedding = self._encode_text(action_text).reshape(1, -1)
        scores: dict[str, float] = {}
        for principle in self.constitution_principles:
            if principle.embedding is None:
                continue
            principle_embedding = principle.embedding.reshape(1, -1)
            similarity = float(cosine_similarity(action_embedding, principle_embedding)[0][0])
            scores[principle.identifier] = max(0.0, similarity * 100.0)
        return scores

    def _check_pattern_violations(self, action_text: str) -> dict[str, float]:
        violations: dict[str, float] = {}
        for name, config in self.violation_patterns.items():
            patterns: Sequence[re.Pattern[str]] = config["patterns"]
            for pattern in patterns:
                if pattern.search(action_text):
                    violations[name] = float(config["severity"])
                    break
        return violations

    def _calculate_total_score(
        self,
        similarity_scores: dict[str, float],
        violations: dict[str, float],
    ) -> float:
        total_score = 50.0
        if similarity_scores:
            avg_similarity = sum(similarity_scores.values()) / max(len(similarity_scores), 1)
            total_score += avg_similarity * 0.5
        total_score += sum(violations.values())
        return float(max(0.0, min(100.0, total_score)))

    def _generate_reasoning(
        self,
        similarity_scores: dict[str, float],
        violations: dict[str, float],
        total_score: float,
    ) -> tuple[str, list[str]]:
        segments: list[str] = []
        recommendations: list[str] = []

        if similarity_scores:
            top_matches = sorted(
                similarity_scores.items(), key=lambda item: item[1], reverse=True
            )[:3]
            segments.append("Top principle alignments:")
            for identifier, score in top_matches:
                title = self._lookup_principle_title(identifier)
                segments.append(f"  - {title}: {score:.1f}%")

        if violations:
            segments.append("Violations detected:")
            for name, penalty in violations.items():
                description = self.violation_patterns[name]["description"]
                segments.append(f"  - {description}: {penalty} points")
                if name == "code_execution":
                    recommendations.append("Avoid dynamic code execution in user-facing features.")
                elif name == "security_bypass":
                    recommendations.append("Implement proper authentication and authorization.")
                elif name == "placeholder_code":
                    recommendations.append("Remove placeholder code before production deployment.")
                elif name == "data_exfiltration":
                    recommendations.append("Ensure data exports exclude confidential information.")
        else:
            segments.append("No critical violations detected.")

        segments.append(f"Total constitutional score: {total_score:.1f}/100")
        if total_score < self.score_threshold:
            segments.append("ACTION REJECTED: Below constitutional threshold")
            recommendations.append(
                "Review action for compliance with constitutional principles before proceeding."
            )
        else:
            segments.append("ACTION APPROVED: Meets constitutional standards")

        return "\n".join(segments), recommendations

    def _lookup_principle_title(self, identifier: str) -> str:
        for principle in self.constitution_principles:
            if principle.identifier == identifier:
                return principle.title
        return identifier

    async def _log_evaluation(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
        result: EvaluationResult,
    ) -> None:
        loop = asyncio.get_running_loop()
        log_entry = {
            "timestamp": loop.time(),
            "action_type": action.get("type", "unknown"),
            "user_intent": context.get("user_intent", "unknown"),
            "approved": result.approved,
            "total_score": result.total_score,
            "confidence": result.confidence,
            "violations": list(result.violation_scores.keys()),
            "recommendations": result.recommendations,
        }
        self.logger.info("Constitutional evaluation: %s", log_entry)


async def create_constitutional_reasoner() -> ConstitutionalReasoner:
    """Factory helper to create a configured reasoner."""

    return ConstitutionalReasoner()


__all__ = [
    "ConstitutionalPrinciple",
    "EvaluationResult",
    "ConstitutionalReasoner",
    "create_constitutional_reasoner",
]
