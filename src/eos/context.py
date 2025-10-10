"""
EOS Context Classification and Cynefin Framework

Implements context analysis and classification using the Cynefin framework
for E-UPUSF orchestration decision making.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CynefinDomain(Enum):
    """Cynefin framework domains"""

    SIMPLE = "simple"
    COMPLICATED = "complicated"
    COMPLEX = "complex"
    CHAOTIC = "chaotic"
    DISORDER = "disorder"


@dataclass
class ContextFeatures:
    """Features extracted from problem context"""

    clarity_of_cause_effect: float  # 0-1: clear cause-effect relationships
    solution_predictability: float  # 0-1: how predictable solutions are
    expertise_availability: float  # 0-1: availability of domain expertise
    time_pressure: float  # 0-1: urgency/time constraints
    stakeholder_agreement: float  # 0-1: alignment among stakeholders
    knowledge_gaps: float  # 0-1: extent of unknown unknowns
    feedback_delay: float  # 0-1: delay in feedback loops
    environmental_volatility: float  # 0-1: rate of environmental change


@dataclass
class CynefinClassification:
    """Result of Cynefin classification"""

    probabilities: dict[CynefinDomain, float]
    primary_domain: CynefinDomain
    entropy: float
    confidence: float
    features: ContextFeatures
    reasoning: list[str]


class CynefinClassifier:
    """Classifies problem contexts using Cynefin framework"""

    def __init__(self):
        """Initialize classifier with domain rules"""
        # Weights for different features in each domain
        self.domain_weights = {
            CynefinDomain.SIMPLE: {
                "clarity_of_cause_effect": 0.8,
                "solution_predictability": 0.9,
                "expertise_availability": 0.6,
                # negative: simple problems ok under pressure
                "time_pressure": -0.3,
                "stakeholder_agreement": 0.7,
                "knowledge_gaps": -0.8,  # negative: few knowledge gaps
                "feedback_delay": -0.5,  # negative: quick feedback
                # negative: stable environment
                "environmental_volatility": -0.7,
            },
            CynefinDomain.COMPLICATED: {
                "clarity_of_cause_effect": 0.6,
                "solution_predictability": 0.7,
                "expertise_availability": 0.9,  # high: needs experts
                "time_pressure": -0.1,
                "stakeholder_agreement": 0.5,
                "knowledge_gaps": -0.4,
                "feedback_delay": 0.1,
                "environmental_volatility": -0.3,
            },
            CynefinDomain.COMPLEX: {
                # negative: unclear relationships
                "clarity_of_cause_effect": -0.2,
                "solution_predictability": -0.5,  # negative: unpredictable
                "expertise_availability": 0.3,
                "time_pressure": 0.2,
                "stakeholder_agreement": -0.2,
                "knowledge_gaps": 0.6,  # positive: significant unknowns
                "feedback_delay": 0.4,
                "environmental_volatility": 0.5,
            },
            CynefinDomain.CHAOTIC: {
                # negative: no clear relationships
                "clarity_of_cause_effect": -0.8,
                # negative: highly unpredictable
                "solution_predictability": -0.9,
                # negative: expertise limited value
                "expertise_availability": -0.3,
                "time_pressure": 0.8,  # positive: high urgency
                "stakeholder_agreement": -0.6,
                "knowledge_gaps": 0.9,  # positive: many unknowns
                "feedback_delay": 0.7,
                "environmental_volatility": 0.9,  # positive: high volatility
            },
        }

    def extract_features(
        self, problem_context: dict[str, Any]
    ) -> ContextFeatures:
        """Extract context features from problem description"""

        # Default feature values
        features = ContextFeatures(
            clarity_of_cause_effect=0.5,
            solution_predictability=0.5,
            expertise_availability=0.5,
            time_pressure=0.5,
            stakeholder_agreement=0.5,
            knowledge_gaps=0.5,
            feedback_delay=0.5,
            environmental_volatility=0.5,
        )

        # Extract from problem statement and constraints
        problem_text = problem_context.get("statement", "").lower()
        constraints = problem_context.get("constraints", [])
        stakeholders = problem_context.get("stakeholders", [])
        risk_tolerance = problem_context.get("risk_tolerance", {})

        # Analyze problem statement for indicators

        # Clarity of cause-effect
        clarity_indicators = [
            "because",
            "therefore",
            "results in",
            "causes",
            "leads to",
            "deterministic",
            "predictable",
        ]
        chaos_indicators = [
            "uncertain",
            "unknown",
            "unpredictable",
            "volatile",
            "complex",
            "emergent",
            "adaptive",
        ]

        clarity_score = sum(
            1 for indicator in clarity_indicators if indicator in problem_text
        )
        chaos_score = sum(
            1 for indicator in chaos_indicators if indicator in problem_text
        )

        if clarity_score + chaos_score > 0:
            features.clarity_of_cause_effect = clarity_score / (
                clarity_score + chaos_score
            )

        # Solution predictability
        predictable_indicators = [
            "standard",
            "established",
            "proven",
            "routine",
        ]
        unpredictable_indicators = [
            "novel",
            "innovative",
            "experimental",
            "unprecedented",
        ]

        pred_score = sum(
            1
            for indicator in predictable_indicators
            if indicator in problem_text
        )
        unpred_score = sum(
            1
            for indicator in unpredictable_indicators
            if indicator in problem_text
        )

        if pred_score + unpred_score > 0:
            features.solution_predictability = pred_score / (
                pred_score + unpred_score
            )

        # Expertise availability (inferred from domain hints)
        domain_hints = problem_context.get("domain_hints", [])
        established_domains = ["dmaic", "six-sigma", "lean", "standard"]
        emerging_domains = ["ai", "ml", "blockchain", "quantum", "novel"]

        est_count = sum(
            1
            for hint in domain_hints
            for domain in established_domains
            if domain in hint.lower()
        )
        emerg_count = sum(
            1
            for hint in domain_hints
            for domain in emerging_domains
            if domain in hint.lower()
        )

        if est_count + emerg_count > 0:
            features.expertise_availability = est_count / (
                est_count + emerg_count
            )

        # Time pressure from constraints
        time_indicators = [
            "urgent",
            "immediate",
            "asap",
            "deadline",
            "quickly",
        ]
        time_pressure_score = sum(
            1
            for constraint in constraints
            for indicator in time_indicators
            if indicator in constraint.lower()
        )

        features.time_pressure = min(1.0, time_pressure_score / 3.0)

        # Stakeholder agreement (inverse of stakeholder count complexity)
        if len(stakeholders) > 0:
            # More stakeholders typically means less agreement
            features.stakeholder_agreement = max(
                0.1, 1.0 - (len(stakeholders) - 1) * 0.2
            )

        # Knowledge gaps from uncertainty
        uncertainty_indicators = [
            "unclear",
            "unknown",
            "investigate",
            "research",
        ]
        gap_score = sum(
            1
            for constraint in constraints + [problem_text]
            for indicator in uncertainty_indicators
            if indicator in str(constraint).lower()
        )

        features.knowledge_gaps = min(1.0, gap_score / 5.0)

        # Risk tolerance affects volatility perception
        risk_scores = list(risk_tolerance.values())
        if risk_scores:
            # High risk tolerance suggests volatile environment
            risk_avg = sum(
                1 if r == "high" else 0.5 if r == "medium" else 0
                for r in risk_scores
            ) / len(risk_scores)
            features.environmental_volatility = risk_avg

        return features

    def classify(
        self,
        problem_context: dict[str, Any],
        prior_distribution: dict[str, float] | None = None,
    ) -> CynefinClassification:
        """Classify problem context into Cynefin domains"""

        # Extract features
        features = self.extract_features(problem_context)

        # Calculate domain scores
        domain_scores = {}
        reasoning = []

        for domain in CynefinDomain:
            if domain == CynefinDomain.DISORDER:
                continue  # Skip disorder for now

            score = 0.0
            weights = self.domain_weights[domain]

            # Calculate weighted feature score
            for feature_name, weight in weights.items():
                feature_value = getattr(features, feature_name)
                contribution = weight * feature_value
                score += contribution

                if abs(contribution) > 0.3:  # Significant contribution
                    reasoning.append(
                        f"{domain.value}: {feature_name}={feature_value:.2f} "
                        f"(weight={weight:.2f}, contrib={contribution:.2f})"
                    )

            domain_scores[domain] = score

        # Apply prior distribution if provided
        if prior_distribution:
            for domain in domain_scores:
                prior_weight = prior_distribution.get(domain.value, 0.25)
                domain_scores[domain] = (
                    0.7 * domain_scores[domain] + 0.3 * prior_weight
                )

        # Convert scores to probabilities using softmax
        max_score = max(domain_scores.values())
        exp_scores = {
            domain: math.exp(score - max_score)
            for domain, score in domain_scores.items()
        }
        total_exp = sum(exp_scores.values())

        probabilities = {
            domain: exp_score / total_exp
            for domain, exp_score in exp_scores.items()
        }

        # Find primary domain
        primary_domain = max(
            probabilities.keys(), key=lambda d: probabilities[d]
        )

        # Calculate entropy for uncertainty measure
        entropy = -sum(
            p * math.log2(p) if p > 0 else 0 for p in probabilities.values()
        )

        # Calculate confidence (inverse of entropy, normalized)
        max_entropy = math.log2(len(probabilities))
        confidence = 1.0 - (entropy / max_entropy)

        return CynefinClassification(
            probabilities=probabilities,
            primary_domain=primary_domain,
            entropy=entropy,
            confidence=confidence,
            features=features,
            reasoning=reasoning,
        )


class ContextAnalyzer:
    """High-level context analysis for EOS orchestration"""

    def __init__(self):
        self.cynefin_classifier = CynefinClassifier()

    def analyze_context(self, eos_spec: dict[str, Any]) -> dict[str, Any]:
        """Analyze context from EOS specification"""

        problem = eos_spec.get("problem", {})
        context_config = eos_spec.get("context", {})

        # Get prior distribution
        prior_dist = context_config.get("cynefin_prior", {})

        # Classify using Cynefin framework
        classification = self.cynefin_classifier.classify(problem, prior_dist)

        # Determine recommended methods based on classification
        method_recommendations = self._recommend_methods(classification)

        # Check if semantic lifting should be triggered
        uncertainty_thresholds = context_config.get(
            "uncertainty_thresholds", {}
        )
        entropy_threshold = uncertainty_thresholds.get(
            "entropy_promote_lift", 0.9
        )
        chaotic_threshold = uncertainty_thresholds.get(
            "chaotic_emergency", 0.4
        )

        should_lift = classification.entropy > entropy_threshold
        chaotic_mode = (
            classification.probabilities.get(CynefinDomain.CHAOTIC, 0.0)
            > chaotic_threshold
        )

        return {
            "cynefin_classification": {
                "probabilities": {
                    domain.value: prob
                    for domain, prob in classification.probabilities.items()
                },
                "primary_domain": classification.primary_domain.value,
                "entropy": classification.entropy,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
            },
            "features": {
                "clarity_of_cause_effect": classification.features.clarity_of_cause_effect,
                "solution_predictability": classification.features.solution_predictability,
                "expertise_availability": classification.features.expertise_availability,
                "time_pressure": classification.features.time_pressure,
                "stakeholder_agreement": classification.features.stakeholder_agreement,
                "knowledge_gaps": classification.features.knowledge_gaps,
                "feedback_delay": classification.features.feedback_delay,
                "environmental_volatility": classification.features.environmental_volatility,
            },
            "recommendations": {
                "methods": method_recommendations,
                "should_semantic_lift": should_lift,
                "chaotic_mode": chaotic_mode,
            },
            "thresholds": {
                "entropy_promote_lift": entropy_threshold,
                "chaotic_emergency": chaotic_threshold,
            },
        }

    def _recommend_methods(
        self, classification: CynefinClassification
    ) -> list[str]:
        """Recommend methods based on Cynefin classification"""

        primary = classification.primary_domain
        probabilities = classification.probabilities

        methods = []

        # Primary method based on dominant domain
        if primary == CynefinDomain.SIMPLE:
            methods.append("POLYA")
        elif primary == CynefinDomain.COMPLICATED:
            methods.extend(["DMAIC", "TRIZ"])
        elif primary == CynefinDomain.COMPLEX:
            methods.extend(["SYSTEMS", "DESIGN"])
        elif primary == CynefinDomain.CHAOTIC:
            methods.append("CYNEFIN_EMERGENCY")

        # Add secondary methods based on significant probabilities
        for domain, prob in probabilities.items():
            if (
                domain != primary and prob > 0.25
            ):  # Significant secondary domain
                if (
                    domain == CynefinDomain.COMPLICATED
                    and "DMAIC" not in methods
                ):
                    methods.append("DMAIC")
                elif (
                    domain == CynefinDomain.COMPLEX
                    and "SYSTEMS" not in methods
                ):
                    methods.append("SYSTEMS")

        return methods

    def update_context(
        self, current_context: dict[str, Any], new_evidence: dict[str, Any]
    ) -> dict[str, Any]:
        """Update context analysis with new evidence"""

        # Update probabilities using Bayesian updating
        current_probs = current_context.get("cynefin_classification", {}).get(
            "probabilities", {}
        )

        # Simple Bayesian update (would be more sophisticated in practice)
        evidence_strength = new_evidence.get("strength", 0.1)
        evidence_domain = new_evidence.get("supports_domain", "complex")

        updated_probs = current_probs.copy()
        if evidence_domain in updated_probs:
            # Increase probability of supported domain
            updated_probs[evidence_domain] = min(
                1.0, updated_probs[evidence_domain] + evidence_strength
            )

            # Normalize probabilities
            total = sum(updated_probs.values())
            updated_probs = {k: v / total for k, v in updated_probs.items()}

        # Update context
        updated_context = current_context.copy()
        updated_context["cynefin_classification"][
            "probabilities"
        ] = updated_probs

        # Recalculate primary domain and entropy
        primary_domain = max(
            updated_probs.keys(), key=lambda k: updated_probs[k]
        )
        entropy = -sum(
            p * math.log2(p) if p > 0 else 0 for p in updated_probs.values()
        )

        updated_context["cynefin_classification"][
            "primary_domain"
        ] = primary_domain
        updated_context["cynefin_classification"]["entropy"] = entropy

        return updated_context
