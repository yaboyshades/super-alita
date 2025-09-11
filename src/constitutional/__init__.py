"""Constitutional Framework for Super-Alita SDD Implementation.

This package provides the core constitutional compliance validation and scoring
system that enforces the six constitutional articles throughout the SDD workflow.

Core Components:
- ConstitutionalScorer: Main scoring engine for all artifacts
- ArticleValidators: Specific validation logic for each constitutional article
- QualityScorecard: 13-point constitutional quality assessment
- ViolationResponse: Handles constitutional violations and corrections
"""

from .articles import (
    ClarityValidator,
    CounterfactualValidator,
    IntegrationFirstValidator,
    LibraryFirstValidator,
    SimplicityGateValidator,
    TestFirstValidator,
)
from .scorecard import ConstitutionalQualityScorecard, ScorecardResult
from .scorer import (
    ConstitutionalResult,
    ConstitutionalScorer,
    ConstitutionalViolation,
)
from .violations import ViolationResponse, ViolationResponseProtocol

__all__ = [
    "ConstitutionalScorer",
    "ConstitutionalViolation",
    "ConstitutionalResult",
    "LibraryFirstValidator",
    "TestFirstValidator",
    "SimplicityGateValidator",
    "IntegrationFirstValidator",
    "ClarityValidator",
    "CounterfactualValidator",
    "ConstitutionalQualityScorecard",
    "ScorecardResult",
    "ViolationResponse",
    "ViolationResponseProtocol",
]

__version__ = "1.0.0"
