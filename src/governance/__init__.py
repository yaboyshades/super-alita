"""Governance and constitutional reasoning components."""

from .constitutional_reasoner import ConstitutionalReasoner, ConstitutionalPrinciple, EvaluationResult

class ConstitutionalViolationError(Exception):
    """Raised when an action violates constitutional principles."""
    
    def __init__(self, reasoning: str, violation_details: dict = None):
        self.reasoning = reasoning
        self.violation_details = violation_details or {}
        super().__init__(reasoning)

__all__ = [
    "ConstitutionalReasoner",
    "ConstitutionalPrinciple", 
    "EvaluationResult",
    "ConstitutionalViolationError"
]