"""
Advanced Constitutional Validation with Pattern Matching and Trend Analysis

Enhanced constitutional compliance system with advanced pattern recognition,
machine learning-based trend analysis, and intelligent violation prediction.
"""

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from re import Pattern
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ViolationPattern:
    """Constitutional violation pattern definition."""

    pattern_id: str
    name: str
    description: str
    article: str
    pattern_type: str  # regex, ast, semantic, behavioral
    pattern_data: dict[str, Any]
    severity: str  # low, medium, high, critical
    confidence_threshold: float = 0.7
    frequency_weight: float = 1.0


@dataclass
class TrendAnalysis:
    """Constitutional compliance trend analysis."""

    metric_name: str
    time_period: str
    trend_direction: str  # improving, stable, degrading
    trend_strength: float  # 0.0 to 1.0
    confidence: float
    key_factors: list[str]
    predictions: dict[str, float]
    recommendations: list[str]


@dataclass
class ComplianceRiskAssessment:
    """Risk assessment for constitutional compliance."""

    overall_risk_score: float  # 0.0 to 1.0
    risk_categories: dict[str, float]
    high_risk_areas: list[str]
    mitigation_strategies: list[str]
    predicted_violations: list[dict[str, Any]]
    confidence: float


class AdvancedConstitutionalValidator:
    """
    Advanced constitutional validator with pattern matching and trend analysis.

    Implements Article III: Simplicity through focused validation logic.
    Implements Article V: Clarity through clear pattern definitions and analysis.
    """

    def __init__(self, constitutional_engine, performance_monitor):
        self.constitutional_engine = constitutional_engine
        self.performance_monitor = performance_monitor

        # Pattern matching system
        self.violation_patterns: dict[str, ViolationPattern] = {}
        self.compiled_regex_patterns: dict[str, Pattern] = {}

        # Trend analysis
        self.compliance_history: list[dict[str, Any]] = []
        self.trend_analyses: dict[str, TrendAnalysis] = {}

        # Risk assessment
        self.risk_assessments: list[ComplianceRiskAssessment] = []

        # ML models (simplified)
        self.pattern_frequency: dict[str, int] = {}
        self.violation_correlations: dict[str, list[str]] = {}

        # Initialize patterns
        self._initialize_violation_patterns()

        logger.info("Advanced Constitutional Validator initialized")

    def register_violation_pattern(self, pattern: ViolationPattern) -> None:
        """Register a new violation pattern."""
        self.violation_patterns[pattern.pattern_id] = pattern

        # Compile regex patterns
        if pattern.pattern_type == "regex" and "regex" in pattern.pattern_data:
            try:
                self.compiled_regex_patterns[pattern.pattern_id] = re.compile(
                    pattern.pattern_data["regex"], re.MULTILINE | re.DOTALL
                )
            except re.error as e:
                logger.error(
                    f"Invalid regex pattern {pattern.pattern_id}: {e}"
                )

        logger.info(f"Registered violation pattern: {pattern.pattern_id}")

    async def analyze_code_patterns(
        self, code_content: str, file_path: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Analyze code for constitutional violation patterns."""
        violations = []

        for pattern_id, pattern in self.violation_patterns.items():
            try:
                matches = await self._apply_pattern(
                    pattern, code_content, file_path, context
                )
                for match in matches:
                    violations.append(
                        {
                            "pattern_id": pattern_id,
                            "pattern_name": pattern.name,
                            "article": pattern.article,
                            "severity": pattern.severity,
                            "location": match.get("location"),
                            "description": match.get("description"),
                            "confidence": match.get(
                                "confidence", pattern.confidence_threshold
                            ),
                            "suggestion": match.get("suggestion"),
                            "file_path": file_path,
                        }
                    )

                    # Update pattern frequency
                    self.pattern_frequency[pattern_id] = (
                        self.pattern_frequency.get(pattern_id, 0) + 1
                    )

            except Exception as e:
                logger.error(f"Pattern analysis error for {pattern_id}: {e}")

        return violations

    async def perform_trend_analysis(
        self, time_period_days: int = 30
    ) -> dict[str, TrendAnalysis]:
        """Perform comprehensive trend analysis of constitutional compliance."""
        cutoff_date = datetime.now(UTC) - timedelta(days=time_period_days)

        # Filter compliance history
        recent_history = [
            record
            for record in self.compliance_history
            if datetime.fromisoformat(
                record["timestamp"].replace("Z", "+00:00")
            )
            > cutoff_date
        ]

        trend_analyses = {}

        # Analyze overall compliance trend
        overall_trend = await self._analyze_compliance_trend(
            recent_history, "overall_compliance"
        )
        if overall_trend:
            trend_analyses["overall_compliance"] = overall_trend

        # Analyze article-specific trends
        for article in [
            "article_i",
            "article_ii",
            "article_iii",
            "article_iv",
            "article_v",
            "article_vi",
        ]:
            article_trend = await self._analyze_article_trend(
                recent_history, article
            )
            if article_trend:
                trend_analyses[article] = article_trend

        # Analyze violation pattern trends
        pattern_trend = await self._analyze_pattern_trends(recent_history)
        if pattern_trend:
            trend_analyses["violation_patterns"] = pattern_trend

        self.trend_analyses = trend_analyses
        return trend_analyses

    async def assess_compliance_risk(self) -> ComplianceRiskAssessment:
        """Assess constitutional compliance risk."""
        risk_scores = {}
        high_risk_areas = []
        mitigation_strategies = []
        predicted_violations = []

        # Assess risk by article
        for article in [
            "article_i",
            "article_ii",
            "article_iii",
            "article_iv",
            "article_v",
            "article_vi",
        ]:
            risk_score = await self._calculate_article_risk(article)
            risk_scores[article] = risk_score

            if risk_score > 0.7:
                high_risk_areas.append(article)
                mitigation_strategies.extend(
                    await self._get_mitigation_strategies(article)
                )

        # Assess pattern-based risks
        pattern_risks = await self._assess_pattern_risks()
        risk_scores.update(pattern_risks)

        # Predict likely violations
        predicted_violations = await self._predict_future_violations()

        # Calculate overall risk
        overall_risk = (
            sum(risk_scores.values()) / len(risk_scores)
            if risk_scores
            else 0.0
        )

        # Calculate confidence
        confidence = self._calculate_risk_confidence(risk_scores)

        risk_assessment = ComplianceRiskAssessment(
            overall_risk_score=overall_risk,
            risk_categories=risk_scores,
            high_risk_areas=high_risk_areas,
            mitigation_strategies=mitigation_strategies,
            predicted_violations=predicted_violations,
            confidence=confidence,
        )

        self.risk_assessments.append(risk_assessment)
        return risk_assessment

    async def generate_intelligent_recommendations(
        self,
        violations: list[dict[str, Any]],
        trends: dict[str, TrendAnalysis],
        risk_assessment: ComplianceRiskAssessment,
    ) -> list[dict[str, Any]]:
        """Generate intelligent recommendations based on analysis."""
        recommendations = []

        # Violation-based recommendations
        violation_recs = await self._generate_violation_recommendations(
            violations
        )
        recommendations.extend(violation_recs)

        # Trend-based recommendations
        trend_recs = await self._generate_trend_recommendations(trends)
        recommendations.extend(trend_recs)

        # Risk-based recommendations
        risk_recs = await self._generate_risk_recommendations(risk_assessment)
        recommendations.extend(risk_recs)

        # Sort by priority and impact
        recommendations.sort(
            key=lambda x: (
                x.get("priority_score", 0),
                x.get("impact_score", 0),
            ),
            reverse=True,
        )

        return recommendations[:10]  # Top 10 recommendations

    def update_compliance_history(
        self, compliance_data: dict[str, Any]
    ) -> None:
        """Update compliance history for trend analysis."""
        compliance_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "overall_score": compliance_data.get("overall_score", 0.0),
            "article_scores": compliance_data.get("article_scores", {}),
            "violations": compliance_data.get("violations", []),
            "context": compliance_data.get("context", {}),
        }

        self.compliance_history.append(compliance_record)

        # Maintain history limit
        if len(self.compliance_history) > 1000:
            self.compliance_history = self.compliance_history[-1000:]

    def get_pattern_statistics(self) -> dict[str, Any]:
        """Get violation pattern statistics."""
        total_matches = sum(self.pattern_frequency.values())

        if total_matches == 0:
            return {"status": "no_data"}

        pattern_stats = {}
        for pattern_id, frequency in self.pattern_frequency.items():
            pattern = self.violation_patterns.get(pattern_id)
            if pattern:
                pattern_stats[pattern_id] = {
                    "name": pattern.name,
                    "frequency": frequency,
                    "percentage": (frequency / total_matches) * 100,
                    "severity": pattern.severity,
                    "article": pattern.article,
                }

        # Sort by frequency
        sorted_patterns = sorted(
            pattern_stats.items(),
            key=lambda x: x[1]["frequency"],
            reverse=True,
        )

        return {
            "total_matches": total_matches,
            "unique_patterns": len(pattern_stats),
            "top_patterns": dict(sorted_patterns[:5]),
            "pattern_distribution": pattern_stats,
        }

    def _initialize_violation_patterns(self) -> None:
        """Initialize default violation patterns."""

        # Article I: Library-First patterns
        self.register_violation_pattern(
            ViolationPattern(
                pattern_id="custom_http_implementation",
                name="Custom HTTP Implementation",
                description="Custom HTTP implementation instead of using standard libraries",
                article="article_i",
                pattern_type="regex",
                pattern_data={
                    "regex": r"class\s+\w*Http\w*|def\s+\w*http\w*\(.*?\):|def\s+\w*request\w*\(.*?\):"
                },
                severity="medium",
                confidence_threshold=0.6,
            )
        )

        # Article II: Test-First patterns
        self.register_violation_pattern(
            ViolationPattern(
                pattern_id="missing_test_file",
                name="Missing Test File",
                description="Source file without corresponding test file",
                article="article_ii",
                pattern_type="semantic",
                pattern_data={"check_type": "test_file_existence"},
                severity="high",
                confidence_threshold=0.9,
            )
        )

        # Article III: Simplicity patterns
        self.register_violation_pattern(
            ViolationPattern(
                pattern_id="complex_function",
                name="Overly Complex Function",
                description="Function with high complexity",
                article="article_iii",
                pattern_type="ast",
                pattern_data={"max_lines": 50, "max_complexity": 10},
                severity="medium",
                confidence_threshold=0.8,
            )
        )

        # Article IV: Integration-First patterns
        self.register_violation_pattern(
            ViolationPattern(
                pattern_id="missing_integration_test",
                name="Missing Integration Test",
                description="API endpoint without integration test",
                article="article_iv",
                pattern_type="semantic",
                pattern_data={"check_type": "integration_test_coverage"},
                severity="high",
                confidence_threshold=0.7,
            )
        )

        # Article V: Clarity patterns
        self.register_violation_pattern(
            ViolationPattern(
                pattern_id="missing_docstring",
                name="Missing Docstring",
                description="Public function or class without documentation",
                article="article_v",
                pattern_type="regex",
                pattern_data={
                    "regex": r"(?:def|class)\s+[^_]\w*\s*\([^)]*\):\s*(?!\s*['\"])(?!\s*#)"
                },
                severity="medium",
                confidence_threshold=0.8,
            )
        )

        # Article VI: Versioning patterns
        self.register_violation_pattern(
            ViolationPattern(
                pattern_id="breaking_change_without_version",
                name="Breaking Change Without Version Bump",
                description="Breaking API change without proper versioning",
                article="article_vi",
                pattern_type="semantic",
                pattern_data={"check_type": "breaking_change_detection"},
                severity="critical",
                confidence_threshold=0.8,
            )
        )

    async def _apply_pattern(
        self,
        pattern: ViolationPattern,
        code_content: str,
        file_path: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Apply a violation pattern to code content."""
        matches = []

        if pattern.pattern_type == "regex":
            matches = await self._apply_regex_pattern(
                pattern, code_content, file_path
            )
        elif pattern.pattern_type == "semantic":
            matches = await self._apply_semantic_pattern(
                pattern, code_content, file_path, context
            )
        elif pattern.pattern_type == "ast":
            matches = await self._apply_ast_pattern(
                pattern, code_content, file_path
            )

        return matches

    async def _apply_regex_pattern(
        self, pattern: ViolationPattern, code_content: str, file_path: str
    ) -> list[dict[str, Any]]:
        """Apply regex pattern matching."""
        matches = []

        if pattern.pattern_id not in self.compiled_regex_patterns:
            return matches

        regex = self.compiled_regex_patterns[pattern.pattern_id]

        for match in regex.finditer(code_content):
            line_num = code_content[: match.start()].count("\n") + 1

            matches.append(
                {
                    "location": f"line {line_num}",
                    "description": f"{pattern.description} at line {line_num}",
                    "confidence": pattern.confidence_threshold
                    + 0.1,  # Slight boost for exact matches
                    "suggestion": self._get_pattern_suggestion(
                        pattern.pattern_id, match.group()
                    ),
                }
            )

        return matches

    async def _apply_semantic_pattern(
        self,
        pattern: ViolationPattern,
        code_content: str,
        file_path: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Apply semantic pattern matching."""
        matches = []
        check_type = pattern.pattern_data.get("check_type")

        if check_type == "test_file_existence":
            # Check if corresponding test file exists
            if not file_path.startswith("test") and "test" not in file_path:
                test_files = [
                    file_path.replace("src/", "tests/test_"),
                    file_path.replace(".py", "_test.py"),
                    f"tests/test_{Path(file_path).name}",
                ]

                test_exists = any(
                    Path(test_file).exists() for test_file in test_files
                )

                if not test_exists:
                    matches.append(
                        {
                            "location": "file level",
                            "description": f"No test file found for {file_path}",
                            "confidence": 0.9,
                            "suggestion": f"Create test file: tests/test_{Path(file_path).name}",
                        }
                    )

        return matches

    async def _apply_ast_pattern(
        self, pattern: ViolationPattern, code_content: str, file_path: str
    ) -> list[dict[str, Any]]:
        """Apply AST-based pattern matching."""
        matches = []

        # Simple line counting for complexity (simplified AST analysis)
        lines = code_content.split("\n")
        current_function = None
        function_start = 0

        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                if current_function:
                    # Check previous function
                    function_lines = i - function_start
                    max_lines = pattern.pattern_data.get("max_lines", 50)

                    if function_lines > max_lines:
                        matches.append(
                            {
                                "location": f"lines {function_start}-{i}",
                                "description": f"Function '{current_function}' is too long ({function_lines} lines)",
                                "confidence": 0.8,
                                "suggestion": f"Break down '{current_function}' into smaller functions",
                            }
                        )

                # Start new function
                current_function = (
                    line.strip().split("(")[0].replace("def ", "")
                )
                function_start = i + 1

        return matches

    async def _analyze_compliance_trend(
        self, history: list[dict[str, Any]], metric_name: str
    ) -> TrendAnalysis | None:
        """Analyze compliance trend for a metric."""
        if len(history) < 5:
            return None

        scores = [record.get("overall_score", 0.0) for record in history]

        # Calculate trend direction and strength
        recent_avg = sum(scores[-5:]) / 5
        older_avg = (
            sum(scores[:-5]) / len(scores[:-5])
            if len(scores) > 5
            else scores[0]
        )

        trend_direction = "stable"
        trend_strength = 0.0

        if recent_avg > older_avg * 1.05:
            trend_direction = "improving"
            trend_strength = min(1.0, (recent_avg - older_avg) / older_avg)
        elif recent_avg < older_avg * 0.95:
            trend_direction = "degrading"
            trend_strength = min(1.0, (older_avg - recent_avg) / older_avg)

        # Generate predictions
        predictions = {
            "next_week": recent_avg
            + (
                trend_strength * 0.1
                if trend_direction == "improving"
                else -trend_strength * 0.1
            ),
            "next_month": recent_avg
            + (
                trend_strength * 0.3
                if trend_direction == "improving"
                else -trend_strength * 0.3
            ),
        }

        # Generate recommendations
        recommendations = []
        if trend_direction == "degrading":
            recommendations.append(
                "Investigate causes of compliance degradation"
            )
            recommendations.append("Implement additional validation measures")
        elif trend_direction == "improving":
            recommendations.append("Continue current improvement initiatives")
            recommendations.append(
                "Document successful practices for replication"
            )

        return TrendAnalysis(
            metric_name=metric_name,
            time_period=f"{len(history)} records",
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            confidence=0.8,  # Simplified confidence calculation
            key_factors=[
                "Recent development practices",
                "Validation process changes",
            ],
            predictions=predictions,
            recommendations=recommendations,
        )

    async def _analyze_article_trend(
        self, history: list[dict[str, Any]], article: str
    ) -> TrendAnalysis | None:
        """Analyze trend for specific constitutional article."""
        # Simplified article trend analysis
        return None  # Would implement article-specific trend analysis

    async def _analyze_pattern_trends(
        self, history: list[dict[str, Any]]
    ) -> TrendAnalysis | None:
        """Analyze violation pattern trends."""
        # Simplified pattern trend analysis
        return None  # Would implement pattern frequency trend analysis

    async def _calculate_article_risk(self, article: str) -> float:
        """Calculate risk score for constitutional article."""
        # Simplified risk calculation
        base_risk = 0.3

        # Check recent violations
        recent_violations = sum(
            1
            for record in self.compliance_history[-10:]
            if any(
                v.get("article") == article
                for v in record.get("violations", [])
            )
        )

        violation_risk = min(0.5, recent_violations * 0.1)

        return base_risk + violation_risk

    async def _assess_pattern_risks(self) -> dict[str, float]:
        """Assess risks based on violation patterns."""
        pattern_risks = {}

        for pattern_id, frequency in self.pattern_frequency.items():
            pattern = self.violation_patterns.get(pattern_id)
            if pattern:
                # Risk based on frequency and severity
                severity_multiplier = {
                    "low": 0.2,
                    "medium": 0.5,
                    "high": 0.8,
                    "critical": 1.0,
                }
                base_risk = (
                    frequency
                    * 0.1
                    * severity_multiplier.get(pattern.severity, 0.5)
                )
                pattern_risks[f"pattern_{pattern_id}"] = min(1.0, base_risk)

        return pattern_risks

    async def _predict_future_violations(self) -> list[dict[str, Any]]:
        """Predict likely future violations."""
        predictions = []

        # Based on pattern frequency trends
        for pattern_id, frequency in self.pattern_frequency.items():
            if frequency > 5:  # Frequently occurring patterns
                pattern = self.violation_patterns.get(pattern_id)
                if pattern:
                    predictions.append(
                        {
                            "pattern_id": pattern_id,
                            "pattern_name": pattern.name,
                            "likelihood": min(0.9, frequency * 0.1),
                            "severity": pattern.severity,
                            "article": pattern.article,
                        }
                    )

        return predictions

    def _calculate_risk_confidence(
        self, risk_scores: dict[str, float]
    ) -> float:
        """Calculate confidence in risk assessment."""
        # Simplified confidence based on data availability
        data_points = len(self.compliance_history)
        pattern_diversity = len(self.pattern_frequency)

        confidence = min(
            0.95, (data_points / 100) * 0.5 + (pattern_diversity / 10) * 0.5
        )
        return max(0.1, confidence)

    async def _generate_violation_recommendations(
        self, violations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate recommendations based on violations."""
        recommendations = []

        # Group violations by pattern
        pattern_groups = {}
        for violation in violations:
            pattern_id = violation.get("pattern_id")
            if pattern_id not in pattern_groups:
                pattern_groups[pattern_id] = []
            pattern_groups[pattern_id].append(violation)

        # Generate recommendations for frequent patterns
        for pattern_id, pattern_violations in pattern_groups.items():
            if len(pattern_violations) > 2:  # Frequent pattern
                pattern = self.violation_patterns.get(pattern_id)
                if pattern:
                    recommendations.append(
                        {
                            "type": "pattern_remediation",
                            "title": f"Address {pattern.name} pattern",
                            "description": f"Found {len(pattern_violations)} instances of {pattern.name}",
                            "priority_score": len(pattern_violations) * 2,
                            "impact_score": (
                                3
                                if pattern.severity in ["high", "critical"]
                                else 1
                            ),
                            "actions": [
                                f"Review and fix {len(pattern_violations)} violations",
                                "Add automated detection for this pattern",
                                "Update development guidelines",
                            ],
                        }
                    )

        return recommendations

    async def _generate_trend_recommendations(
        self, trends: dict[str, TrendAnalysis]
    ) -> list[dict[str, Any]]:
        """Generate recommendations based on trends."""
        recommendations = []

        for trend_name, trend in trends.items():
            if (
                trend.trend_direction == "degrading"
                and trend.trend_strength > 0.3
            ):
                recommendations.append(
                    {
                        "type": "trend_intervention",
                        "title": f"Address degrading {trend_name} trend",
                        "description": f"{trend_name} showing {trend.trend_direction} trend",
                        "priority_score": int(trend.trend_strength * 10),
                        "impact_score": 4,
                        "actions": trend.recommendations,
                    }
                )

        return recommendations

    async def _generate_risk_recommendations(
        self, risk_assessment: ComplianceRiskAssessment
    ) -> list[dict[str, Any]]:
        """Generate recommendations based on risk assessment."""
        recommendations = []

        if risk_assessment.overall_risk_score > 0.7:
            recommendations.append(
                {
                    "type": "risk_mitigation",
                    "title": "High constitutional compliance risk detected",
                    "description": f"Overall risk score: {risk_assessment.overall_risk_score:.2f}",
                    "priority_score": 10,
                    "impact_score": 5,
                    "actions": risk_assessment.mitigation_strategies[:3],
                }
            )

        return recommendations

    async def _get_mitigation_strategies(self, article: str) -> list[str]:
        """Get mitigation strategies for constitutional article."""
        strategies = {
            "article_i": [
                "Review library usage",
                "Update dependency guidelines",
            ],
            "article_ii": [
                "Increase test coverage",
                "Implement TDD practices",
            ],
            "article_iii": ["Refactor complex code", "Simplify interfaces"],
            "article_iv": ["Add integration tests", "Improve API contracts"],
            "article_v": [
                "Improve documentation",
                "Clarify naming conventions",
            ],
            "article_vi": [
                "Update versioning strategy",
                "Implement breaking change detection",
            ],
        }
        return strategies.get(article, ["Review constitutional compliance"])

    def _get_pattern_suggestion(
        self, pattern_id: str, matched_text: str
    ) -> str:
        """Get suggestion for pattern violation."""
        suggestions = {
            "custom_http_implementation": "Consider using requests library or urllib",
            "missing_docstring": "Add docstring describing function purpose",
            "complex_function": "Break down into smaller, focused functions",
            "missing_test_file": "Create corresponding test file",
            "missing_integration_test": "Add integration test for API endpoint",
            "breaking_change_without_version": "Update version number for breaking change",
        }
        return suggestions.get(pattern_id, "Review and fix violation")
