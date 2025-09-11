"""Constitutional Quality Scorecard.

Implements the 13-point Constitutional Quality Scorecard that evaluates
prompts and specifications against constitutional principles with weighted scoring.
"""

from dataclasses import dataclass


@dataclass
class ScorecardResult:
    """Result from constitutional quality scorecard evaluation."""

    total_score: float  # Total score (0-42 points)
    constitutional_score: float  # Constitutional compliance (0-1)
    task_clarity_score: float  # Task clarity (0-10, 2x weighted)
    criteria_scores: dict[str, float]  # Individual criteria scores
    improvements: list[str]  # Improvement suggestions


class ConstitutionalQualityScorecard:
    """13-point Constitutional Quality Scorecard for prompt evaluation.

    Evaluates prompts against:
    1. Task Clarity (2x weight)
    2. Role Assignment
    3. Context
    4. Output Format
    5. Tone/Constraints
    6. Reasoning Request
    7. Ambiguity
    8. Library-First Compliance
    9. Test-First Requirements
    10. Simplicity Constraints
    11. Integration Testing
    12. Decision Justification
    13. Constitutional Alignment
    """

    def __init__(self):
        """Initialize the scorecard with criteria weights."""
        self.criteria_weights = {
            "task_clarity": 2.0,  # 2x weight
            "role_assignment": 1.0,
            "context": 1.0,
            "output_format": 1.0,
            "tone_constraints": 1.0,
            "reasoning_request": 1.0,
            "ambiguity": 1.0,
            "library_first": 1.0,
            "test_first": 1.0,
            "simplicity": 1.0,
            "integration": 1.0,
            "justification": 1.0,
            "constitutional_alignment": 1.0,
        }
        self.max_score = 42  # 13 criteria * 3 points + task clarity bonus

    def score_prompt(self, prompt: str) -> ScorecardResult:
        """Score a prompt using the 13-point constitutional scorecard."""
        criteria_scores = {}
        improvements = []

        # 1. Task Clarity (2x weight)
        clarity_score = self._score_task_clarity(prompt)
        criteria_scores["task_clarity"] = clarity_score
        if clarity_score < 8:
            improvements.append("Improve task clarity and specificity")

        # 2. Role Assignment
        role_score = self._score_role_assignment(prompt)
        criteria_scores["role_assignment"] = role_score
        if role_score < 2:
            improvements.append("Add clear role definition")

        # 3. Context
        context_score = self._score_context(prompt)
        criteria_scores["context"] = context_score
        if context_score < 2:
            improvements.append("Provide more context and background")

        # 4. Output Format
        format_score = self._score_output_format(prompt)
        criteria_scores["output_format"] = format_score
        if format_score < 2:
            improvements.append("Specify desired output format")

        # 5. Tone/Constraints
        tone_score = self._score_tone_constraints(prompt)
        criteria_scores["tone_constraints"] = tone_score

        # 6. Reasoning Request
        reasoning_score = self._score_reasoning_request(prompt)
        criteria_scores["reasoning_request"] = reasoning_score

        # 7. Ambiguity (reverse scored - lower ambiguity = higher score)
        ambiguity_score = self._score_ambiguity(prompt)
        criteria_scores["ambiguity"] = ambiguity_score
        if ambiguity_score < 2:
            improvements.append("Reduce ambiguous language")

        # 8-13: Constitutional criteria
        criteria_scores["library_first"] = self._score_library_first(prompt)
        criteria_scores["test_first"] = self._score_test_first(prompt)
        criteria_scores["simplicity"] = self._score_simplicity(prompt)
        criteria_scores["integration"] = self._score_integration(prompt)
        criteria_scores["justification"] = self._score_justification(prompt)
        criteria_scores["constitutional_alignment"] = (
            self._score_constitutional_alignment(prompt)
        )

        # Calculate weighted total score
        total_score = sum(
            score * self.criteria_weights[criterion]
            for criterion, score in criteria_scores.items()
        )

        # Calculate constitutional compliance (0-1 scale)
        constitutional_score = total_score / self.max_score

        return ScorecardResult(
            total_score=total_score,
            constitutional_score=constitutional_score,
            task_clarity_score=clarity_score,
            criteria_scores=criteria_scores,
            improvements=improvements,
        )

    def _score_task_clarity(self, prompt: str) -> float:
        """Score task clarity (0-10 points, 2x weighted)."""
        score = 3.0  # Base score

        # Check for clear action verbs
        action_verbs = ["build", "create", "implement", "develop", "design"]
        if any(verb in prompt.lower() for verb in action_verbs):
            score += 2

        # Check for specific requirements
        if "requirements" in prompt.lower() or "must" in prompt.lower():
            score += 2

        # Check for acceptance criteria
        if "acceptance" in prompt.lower() or "criteria" in prompt.lower():
            score += 2

        # Penalize vague language
        vague_words = ["something", "anything", "stuff", "things"]
        if any(word in prompt.lower() for word in vague_words):
            score -= 1

        return min(10.0, max(0.0, score))

    def _score_role_assignment(self, prompt: str) -> float:
        """Score role assignment clarity (0-3 points)."""
        score = 1.0  # Base score

        role_indicators = ["you are", "act as", "role", "expert", "specialist"]
        if any(indicator in prompt.lower() for indicator in role_indicators):
            score += 2

        return min(3.0, score)

    def _score_context(self, prompt: str) -> float:
        """Score context provision (0-3 points)."""
        score = 1.0  # Base score

        context_indicators = ["context", "background", "scenario", "situation"]
        if any(indicator in prompt.lower() for indicator in context_indicators):
            score += 1

        # Check for domain-specific context
        if len(prompt) > 100:  # Longer prompts likely have more context
            score += 1

        return min(3.0, score)

    def _score_output_format(self, prompt: str) -> float:
        """Score output format specification (0-3 points)."""
        score = 1.0  # Base score

        format_indicators = ["format", "output", "return", "json", "yaml", "list"]
        if any(indicator in prompt.lower() for indicator in format_indicators):
            score += 2

        return min(3.0, score)

    def _score_tone_constraints(self, prompt: str) -> float:
        """Score tone and constraints (0-3 points)."""
        return 2.0  # Default reasonable score

    def _score_reasoning_request(self, prompt: str) -> float:
        """Score reasoning request (0-3 points)."""
        score = 1.0  # Base score

        reasoning_indicators = ["explain", "justify", "reasoning", "why", "because"]
        if any(indicator in prompt.lower() for indicator in reasoning_indicators):
            score += 2

        return min(3.0, score)

    def _score_ambiguity(self, prompt: str) -> float:
        """Score ambiguity (0-3 points, higher = less ambiguous)."""
        score = 3.0  # Start with perfect score

        ambiguous_words = ["maybe", "possibly", "might", "could", "perhaps"]
        ambiguous_count = sum(1 for word in ambiguous_words if word in prompt.lower())

        # Penalize ambiguous language
        score -= min(3.0, ambiguous_count * 0.5)

        return max(0.0, score)

    def _score_library_first(self, prompt: str) -> float:
        """Score library-first compliance (0-3 points)."""
        score = 1.0  # Base score

        library_indicators = ["library", "existing", "use", "leverage", "framework"]
        if any(indicator in prompt.lower() for indicator in library_indicators):
            score += 2

        return min(3.0, score)

    def _score_test_first(self, prompt: str) -> float:
        """Score test-first requirements (0-3 points)."""
        score = 1.0  # Base score

        test_indicators = ["test", "testing", "coverage", "tdd"]
        if any(indicator in prompt.lower() for indicator in test_indicators):
            score += 2

        return min(3.0, score)

    def _score_simplicity(self, prompt: str) -> float:
        """Score simplicity emphasis (0-3 points)."""
        score = 2.0  # Default score

        complexity_indicators = ["complex", "sophisticated", "advanced"]
        if any(indicator in prompt.lower() for indicator in complexity_indicators):
            score -= 1

        simplicity_indicators = ["simple", "clear", "straightforward"]
        if any(indicator in prompt.lower() for indicator in simplicity_indicators):
            score += 1

        return min(3.0, max(0.0, score))

    def _score_integration(self, prompt: str) -> float:
        """Score integration testing emphasis (0-3 points)."""
        score = 1.0  # Base score

        integration_indicators = ["integration", "end-to-end", "e2e", "workflow"]
        if any(indicator in prompt.lower() for indicator in integration_indicators):
            score += 2

        return min(3.0, score)

    def _score_justification(self, prompt: str) -> float:
        """Score decision justification (0-3 points)."""
        score = 1.0  # Base score

        justification_indicators = ["because", "since", "reason", "justify"]
        if any(indicator in prompt.lower() for indicator in justification_indicators):
            score += 2

        return min(3.0, score)

    def _score_constitutional_alignment(self, prompt: str) -> float:
        """Score overall constitutional alignment (0-3 points)."""
        # This is a meta-score based on other constitutional criteria
        constitutional_mentions = sum(
            [
                self._score_library_first(prompt),
                self._score_test_first(prompt),
                self._score_simplicity(prompt),
                self._score_integration(prompt),
                self._score_justification(prompt),
            ]
        )

        # Normalize to 0-3 scale
        normalized_score = min(3.0, constitutional_mentions / 5.0 * 3.0)
        return normalized_score
