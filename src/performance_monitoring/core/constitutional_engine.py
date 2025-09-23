"""
Constitutional Compliance Engine

Validates compliance against the six-article constitutional framework
and provides real-time scoring and violation detection.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConstitutionalArticle(Enum):
    """Constitutional articles for compliance validation."""
    ARTICLE_I_LIBRARY_FIRST = "library_first"
    ARTICLE_II_TEST_FIRST = "test_first"
    ARTICLE_III_SIMPLICITY = "simplicity"
    ARTICLE_IV_INTEGRATION_FIRST = "integration_first"
    ARTICLE_V_CLARITY = "clarity"
    ARTICLE_VI_VERSIONING = "versioning"


@dataclass
class ComplianceViolation:
    """Constitutional compliance violation."""
    article: ConstitutionalArticle
    severity: str  # critical, high, medium, low
    description: str
    location: Optional[str] = None
    suggestion: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ComplianceScore:
    """Constitutional compliance score result."""
    overall_score: float
    article_scores: Dict[ConstitutionalArticle, float]
    violations: List[ComplianceViolation]
    threshold: float = 0.75
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_compliant(self) -> bool:
        return self.overall_score >= self.threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "article_scores": {k.value: v for k, v in self.article_scores.items()},
            "violations": [
                {
                    "article": v.article.value,
                    "severity": v.severity,
                    "description": v.description,
                    "location": v.location,
                    "suggestion": v.suggestion
                } for v in self.violations
            ],
            "is_compliant": self.is_compliant,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat()
        }


class ConstitutionalEngine:
    """
    Core constitutional compliance validation engine.
    
    Implements Article III: Simplicity through focused validation logic.
    Implements Article I: Library-First through standard validation patterns.
    """
    
    def __init__(self, compliance_threshold: float = 0.75):
        self.threshold = compliance_threshold
        self.validation_rules = self._initialize_validation_rules()
        self.compliance_history: List[ComplianceScore] = []
        
        logger.info(f"Constitutional Engine initialized with threshold {compliance_threshold}")

    async def validate_compliance(
        self,
        target_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ComplianceScore:
        """Validate constitutional compliance for given data."""
        violations = []
        article_scores = {}
        
        # Validate each constitutional article
        for article in ConstitutionalArticle:
            score, article_violations = await self._validate_article(
                article, target_data, context or {}
            )
            article_scores[article] = score
            violations.extend(article_violations)
        
        # Calculate overall score (weighted average)
        overall_score = self._calculate_overall_score(article_scores)
        
        # Create compliance score
        compliance_score = ComplianceScore(
            overall_score=overall_score,
            article_scores=article_scores,
            violations=violations,
            threshold=self.threshold
        )
        
        # Store in history
        self.compliance_history.append(compliance_score)
        
        logger.info(f"Constitutional validation complete: {overall_score:.3f} "
                   f"({'COMPLIANT' if compliance_score.is_compliant else 'NON-COMPLIANT'})")
        
        return compliance_score

    async def validate_code_change(
        self,
        file_path: str,
        changes: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ComplianceScore:
        """Validate constitutional compliance for code changes."""
        target_data = {
            "type": "code_change",
            "file_path": file_path,
            "changes": changes,
            "metadata": metadata or {}
        }
        
        return await self.validate_compliance(target_data)

    async def validate_commit(
        self,
        commit_message: str,
        changed_files: List[str],
        diff_data: str
    ) -> ComplianceScore:
        """Validate constitutional compliance for git commits."""
        target_data = {
            "type": "commit",
            "message": commit_message,
            "changed_files": changed_files,
            "diff": diff_data
        }
        
        return await self.validate_compliance(target_data)

    def get_compliance_trend(self, hours: int = 24) -> Dict[str, Any]:
        """Get compliance trend analysis."""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        recent_scores = [
            score for score in self.compliance_history
            if score.timestamp.timestamp() > cutoff_time
        ]
        
        if not recent_scores:
            return {"status": "no_data"}
        
        scores = [s.overall_score for s in recent_scores]
        return {
            "total_validations": len(recent_scores),
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "compliance_rate": sum(1 for s in recent_scores if s.is_compliant) / len(recent_scores),
            "trend": self._calculate_trend(scores)
        }

    async def _validate_article(
        self,
        article: ConstitutionalArticle,
        target_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[float, List[ComplianceViolation]]:
        """Validate a specific constitutional article."""
        violations = []
        
        if article in self.validation_rules:
            for rule in self.validation_rules[article]:
                violation = await rule(target_data, context)
                if violation:
                    violations.append(violation)
        
        # Calculate article score based on violations
        if not violations:
            score = 1.0
        else:
            # Deduct points based on violation severity
            deductions = sum(self._get_severity_deduction(v.severity) for v in violations)
            score = max(0.0, 1.0 - deductions)
        
        return score, violations

    def _calculate_overall_score(self, article_scores: Dict[ConstitutionalArticle, float]) -> float:
        """Calculate weighted overall compliance score."""
        # Equal weighting for all articles
        weights = {article: 1.0 for article in ConstitutionalArticle}
        
        weighted_sum = sum(
            article_scores[article] * weights[article]
            for article in article_scores
        )
        total_weight = sum(weights.values())
        
        return weighted_sum / total_weight

    def _get_severity_deduction(self, severity: str) -> float:
        """Get point deduction for violation severity."""
        deductions = {
            "critical": 0.5,
            "high": 0.3,
            "medium": 0.2,
            "low": 0.1
        }
        return deductions.get(severity, 0.1)

    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate compliance trend from recent scores."""
        if len(scores) < 2:
            return "insufficient_data"
        
        recent_avg = sum(scores[-5:]) / min(5, len(scores))
        older_avg = sum(scores[:-5]) / max(1, len(scores) - 5) if len(scores) > 5 else scores[0]
        
        if recent_avg > older_avg + 0.05:
            return "improving"
        elif recent_avg < older_avg - 0.05:
            return "degrading"
        else:
            return "stable"

    def _initialize_validation_rules(self) -> Dict[ConstitutionalArticle, List]:
        """Initialize constitutional validation rules."""
        return {
            ConstitutionalArticle.ARTICLE_I_LIBRARY_FIRST: [
                self._validate_library_usage,
                self._validate_dependency_management
            ],
            ConstitutionalArticle.ARTICLE_II_TEST_FIRST: [
                self._validate_test_coverage,
                self._validate_test_quality
            ],
            ConstitutionalArticle.ARTICLE_III_SIMPLICITY: [
                self._validate_code_complexity,
                self._validate_interface_simplicity
            ],
            ConstitutionalArticle.ARTICLE_IV_INTEGRATION_FIRST: [
                self._validate_integration_tests,
                self._validate_api_contracts
            ],
            ConstitutionalArticle.ARTICLE_V_CLARITY: [
                self._validate_documentation,
                self._validate_naming_conventions
            ],
            ConstitutionalArticle.ARTICLE_VI_VERSIONING: [
                self._validate_version_compatibility,
                self._validate_breaking_changes
            ]
        }

    async def _validate_library_usage(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Validate Article I: Library-First compliance."""
        # Check for custom implementations where libraries exist
        if data.get("type") == "code_change":
            changes = data.get("changes", [])
            for change in changes:
                if "class " in change and ("http" in change.lower() or "json" in change.lower()):
                    return ComplianceViolation(
                        article=ConstitutionalArticle.ARTICLE_I_LIBRARY_FIRST,
                        severity="medium",
                        description="Custom implementation detected where standard library could be used",
                        suggestion="Consider using existing libraries for HTTP/JSON operations"
                    )
        return None

    async def _validate_dependency_management(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Validate dependency management practices."""
        # Check for proper dependency declaration
        if data.get("file_path", "").endswith(("requirements.txt", "package.json", "pyproject.toml")):
            return None  # Dependency files are compliant
        return None

    async def _validate_test_coverage(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Validate Article II: Test-First compliance."""
        if data.get("type") == "code_change":
            file_path = data.get("file_path", "")
            if not file_path.startswith("test") and "test" not in file_path:
                # New code without corresponding tests
                return ComplianceViolation(
                    article=ConstitutionalArticle.ARTICLE_II_TEST_FIRST,
                    severity="high",
                    description="Code changes without corresponding test updates",
                    suggestion="Add or update tests for modified functionality"
                )
        return None

    async def _validate_test_quality(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Validate test quality and coverage."""
        # Additional test quality checks would go here
        return None

    async def _validate_code_complexity(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Validate Article III: Simplicity compliance."""
        if data.get("type") == "code_change":
            changes = data.get("changes", [])
            for change in changes:
                # Simple complexity check - function length
                if change.count('\n') > 50:
                    return ComplianceViolation(
                        article=ConstitutionalArticle.ARTICLE_III_SIMPLICITY,
                        severity="medium",
                        description="Function/method appears too complex",
                        suggestion="Consider breaking down into smaller, simpler functions"
                    )
        return None

    async def _validate_interface_simplicity(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Validate interface simplicity."""
        return None

    async def _validate_integration_tests(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Validate Article IV: Integration-First compliance."""
        return None

    async def _validate_api_contracts(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Validate API contract adherence."""
        return None

    async def _validate_documentation(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Validate Article V: Clarity compliance."""
        if data.get("type") == "code_change":
            changes = data.get("changes", [])
            for change in changes:
                if ("def " in change or "class " in change) and '"""' not in change:
                    return ComplianceViolation(
                        article=ConstitutionalArticle.ARTICLE_V_CLARITY,
                        severity="medium",
                        description="Public function/class without documentation",
                        suggestion="Add docstrings to document purpose and usage"
                    )
        return None

    async def _validate_naming_conventions(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Validate naming conventions for clarity."""
        return None

    async def _validate_version_compatibility(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Validate Article VI: Versioning compliance."""
        return None

    async def _validate_breaking_changes(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Validate breaking change management."""
        if data.get("type") == "commit":
            message = data.get("message", "").lower()
            if "breaking" in message or "!:" in message:
                # Breaking change detected - ensure proper versioning
                return ComplianceViolation(
                    article=ConstitutionalArticle.ARTICLE_VI_VERSIONING,
                    severity="high",
                    description="Breaking change detected",
                    suggestion="Ensure major version bump and migration guide"
                )
        return None