"""
Constitutional Compliance Engine

Implements SDD constitutional principle validation and guidance:
- Nine constitutional articles from SDD methodology
- Compliance scoring and reporting
- Principle-specific guidance and recommendations
- Advisory-only operation for seamless integration
"""

from enum import Enum


class ConstitutionalArticle(Enum):
    """Enumeration of SDD constitutional articles."""

    LIBRARY_FIRST = "library_first"
    TEST_FIRST = "test_first"
    SIMPLICITY_GATE = "simplicity_gate"
    INTEGRATION_FIRST = "integration_first"
    CLARITY_UNAMBIGUITY = "clarity_unambiguity"
    COUNTERFACTUAL_JUSTIFICATION = "counterfactual_justification"
    DOCUMENTATION_DRIVEN = "documentation_driven"
    TEMPLATE_DRIVEN = "template_driven"
    CLI_INTERFACE = "cli_interface"


class ConstitutionalEngine:
    """
    Constitutional compliance analysis and guidance engine.

    Provides advisory-only constitutional principle validation
    without blocking operations or throwing exceptions.
    """

    def __init__(self):
        """Initialize the constitutional engine with article definitions."""
        self.articles = {
            ConstitutionalArticle.LIBRARY_FIRST: {
                "title": "Library-First Development",
                "description": "Prioritize existing libraries and proven solutions over custom implementations",
                "keywords": [
                    "library",
                    "package",
                    "dependency",
                    "framework",
                    "reuse",
                    "existing",
                ],
                "violations": [
                    "implement from scratch",
                    "custom solution",
                    "build our own",
                ],
                "weight": 1.5,  # Higher priority
            },
            ConstitutionalArticle.TEST_FIRST: {
                "title": "Test-First Development",
                "description": "Write tests before implementation, maintain high coverage",
                "keywords": [
                    "test",
                    "coverage",
                    "tdd",
                    "pytest",
                    "unit test",
                    "integration test",
                ],
                "violations": ["no tests", "skip testing", "test later"],
                "weight": 1.5,  # Higher priority
            },
            ConstitutionalArticle.SIMPLICITY_GATE: {
                "title": "Simplicity Gate",
                "description": "Prefer simple solutions, avoid over-engineering",
                "keywords": ["simple", "minimal", "clean", "straightforward"],
                "violations": [
                    "complex",
                    "sophisticated",
                    "advanced",
                    "enterprise",
                ],
                "weight": 1.2,
            },
            ConstitutionalArticle.INTEGRATION_FIRST: {
                "title": "Integration-First Testing",
                "description": "Validate system integration before unit details",
                "keywords": [
                    "integration",
                    "end-to-end",
                    "e2e",
                    "system test",
                    "contract",
                ],
                "violations": ["mock everything", "isolate completely"],
                "weight": 1.0,
            },
            ConstitutionalArticle.CLARITY_UNAMBIGUITY: {
                "title": "Clarity and Unambiguity",
                "description": "Clear specifications and unambiguous requirements",
                "keywords": [
                    "clear",
                    "specific",
                    "explicit",
                    "detailed",
                    "precise",
                ],
                "violations": [
                    "ambiguous",
                    "unclear",
                    "vague",
                    "needs clarification",
                ],
                "weight": 1.3,
            },
            ConstitutionalArticle.COUNTERFACTUAL_JUSTIFICATION: {
                "title": "Counterfactual Justification",
                "description": "Justify decisions by explaining alternatives not chosen",
                "keywords": [
                    "alternative",
                    "considered",
                    "compared",
                    "versus",
                    "instead",
                ],
                "violations": [
                    "only option",
                    "no alternatives",
                    "obvious choice",
                ],
                "weight": 1.0,
            },
            ConstitutionalArticle.DOCUMENTATION_DRIVEN: {
                "title": "Documentation-Driven Development",
                "description": "Comprehensive documentation drives implementation",
                "keywords": [
                    "documentation",
                    "readme",
                    "docs",
                    "documented",
                    "guide",
                ],
                "violations": ["undocumented", "no docs", "figure it out"],
                "weight": 1.0,
            },
            ConstitutionalArticle.TEMPLATE_DRIVEN: {
                "title": "Template-Driven Development",
                "description": "Use templates and structured approaches for consistency",
                "keywords": [
                    "template",
                    "structure",
                    "pattern",
                    "standard",
                    "consistent",
                ],
                "violations": ["ad hoc", "inconsistent", "no structure"],
                "weight": 1.0,
            },
            ConstitutionalArticle.CLI_INTERFACE: {
                "title": "CLI Interface Design",
                "description": "Command-line interfaces should be intuitive and well-designed",
                "keywords": [
                    "cli",
                    "command",
                    "interface",
                    "usable",
                    "intuitive",
                ],
                "violations": [
                    "confusing interface",
                    "hard to use",
                    "no help",
                ],
                "weight": 0.8,
            },
        }

    def analyze_compliance(
        self, text: str, context: dict | None = None
    ) -> dict:
        """
        Analyze constitutional compliance of given text.

        Args:
            text: Text to analyze for constitutional compliance
            context: Optional context information

        Returns:
            Compliance analysis with scores and recommendations
        """
        if not text or not text.strip():
            return self._empty_analysis()

        text_lower = text.lower()
        results = {
            "overall_score": 0.0,
            "article_scores": {},
            "violations": [],
            "strengths": [],
            "recommendations": [],
            "analysis_metadata": {
                "text_length": len(text),
                "has_context": context is not None,
                "analyzed_articles": len(self.articles),
            },
        }

        total_weighted_score = 0.0
        total_weight = 0.0

        for article, config in self.articles.items():
            score = self._analyze_article_compliance(text_lower, config)
            weight = config["weight"]

            results["article_scores"][article.value] = {
                "score": score,
                "weight": weight,
                "weighted_score": score * weight,
                "title": config["title"],
            }

            total_weighted_score += score * weight
            total_weight += weight

            # Track violations and strengths
            if score < 0.4:
                results["violations"].append(
                    {
                        "article": article.value,
                        "title": config["title"],
                        "score": score,
                        "severity": "high" if score < 0.2 else "medium",
                    }
                )
            elif score > 0.7:
                results["strengths"].append(
                    {
                        "article": article.value,
                        "title": config["title"],
                        "score": score,
                    }
                )

        # Calculate overall score
        if total_weight > 0:
            results["overall_score"] = total_weighted_score / total_weight

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        return results

    def _analyze_article_compliance(self, text: str, config: dict) -> float:
        """Analyze compliance for a specific constitutional article."""
        positive_score = 0.0
        negative_score = 0.0

        # Check for positive indicators
        for keyword in config["keywords"]:
            if keyword in text:
                positive_score += 1.0

        # Check for violations
        for violation in config["violations"]:
            if violation in text:
                negative_score += 1.0

        # Normalize scores
        max_positive = len(config["keywords"])
        max_negative = len(config["violations"])

        if max_positive > 0:
            positive_normalized = min(1.0, positive_score / max_positive)
        else:
            positive_normalized = 0.5  # Neutral if no keywords

        if max_negative > 0:
            negative_normalized = min(1.0, negative_score / max_negative)
        else:
            negative_normalized = 0.0

        # Combine scores (positive evidence minus violations)
        final_score = max(0.0, positive_normalized - negative_normalized)
        return min(1.0, final_score)

    def _generate_recommendations(self, analysis: dict) -> list[dict]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Recommendations for major violations
        for violation in analysis["violations"]:
            if violation["severity"] == "high":
                recommendations.append(
                    {
                        "type": "violation",
                        "priority": "high",
                        "article": violation["article"],
                        "title": f"Address {violation['title']} compliance",
                        "description": self._get_violation_guidance(
                            violation["article"]
                        ),
                    }
                )

        # General improvement recommendations
        overall_score = analysis["overall_score"]
        if overall_score < 0.6:
            recommendations.append(
                {
                    "type": "general",
                    "priority": "medium",
                    "title": "Improve constitutional compliance",
                    "description": "Consider constitutional principles in design and implementation",
                }
            )

        # Specific article recommendations
        article_scores = analysis["article_scores"]
        low_scoring_articles = [
            (article, data)
            for article, data in article_scores.items()
            if data["score"] < 0.5
        ]

        for article, data in low_scoring_articles[
            :3
        ]:  # Top 3 improvement areas
            recommendations.append(
                {
                    "type": "improvement",
                    "priority": "medium",
                    "article": article,
                    "title": f"Strengthen {data['title']}",
                    "description": self._get_improvement_guidance(article),
                }
            )

        return recommendations

    def _get_violation_guidance(self, article: str) -> str:
        """Get specific guidance for constitutional violations."""
        guidance_map = {
            "library_first": "Research existing libraries before implementing. Check PyPI, GitHub, and documentation for proven solutions.",
            "test_first": "Write tests before implementation. Aim for high coverage and test edge cases.",
            "simplicity_gate": "Simplify the approach. Avoid over-engineering and complex patterns when simple solutions exist.",
            "integration_first": "Focus on system integration tests before detailed unit tests.",
            "clarity_unambiguity": "Make specifications more specific and detailed. Remove ambiguous language.",
            "counterfactual_justification": "Explain why alternatives were not chosen. Document the decision-making process.",
            "documentation_driven": "Write comprehensive documentation. Include examples and usage guides.",
            "template_driven": "Use consistent templates and structured approaches.",
            "cli_interface": "Design intuitive command-line interfaces with clear help and error messages.",
        }
        return guidance_map.get(
            article, "Review constitutional principles for this area."
        )

    def _get_improvement_guidance(self, article: str) -> str:
        """Get specific improvement guidance for constitutional articles."""
        improvement_map = {
            "library_first": "Consider leveraging more existing libraries and frameworks",
            "test_first": "Expand test coverage and adopt test-driven development practices",
            "simplicity_gate": "Look for opportunities to simplify and reduce complexity",
            "integration_first": "Add more integration and end-to-end testing",
            "clarity_unambiguity": "Improve specification clarity and remove ambiguities",
            "counterfactual_justification": "Document alternative approaches and justify decisions",
            "documentation_driven": "Enhance documentation and user guides",
            "template_driven": "Adopt more consistent templates and patterns",
            "cli_interface": "Improve command-line interface design and usability",
        }
        return improvement_map.get(
            article, "Focus on this constitutional principle."
        )

    def _empty_analysis(self) -> dict:
        """Return empty analysis for invalid input."""
        return {
            "overall_score": 0.0,
            "article_scores": {},
            "violations": [],
            "strengths": [],
            "recommendations": [],
            "analysis_metadata": {
                "text_length": 0,
                "has_context": False,
                "analyzed_articles": 0,
                "error": "Empty or invalid input",
            },
        }

    def get_article_guidance(self, article: str) -> dict | None:
        """Get detailed guidance for a specific constitutional article."""
        try:
            article_enum = ConstitutionalArticle(article)
            config = self.articles[article_enum]

            return {
                "article": article,
                "title": config["title"],
                "description": config["description"],
                "keywords": config["keywords"],
                "common_violations": config["violations"],
                "weight": config["weight"],
                "guidance": self._get_improvement_guidance(article),
                "examples": self._get_article_examples(article),
            }
        except (ValueError, KeyError):
            return None

    def _get_article_examples(self, article: str) -> list[str]:
        """Get examples for constitutional article application."""
        examples_map = {
            "library_first": [
                "Use requests instead of implementing HTTP client",
                "Leverage FastAPI instead of building web framework",
                "Use pytest instead of custom testing framework",
            ],
            "test_first": [
                "Write test cases before implementing functions",
                "Define expected behavior through tests",
                "Use TDD red-green-refactor cycle",
            ],
            "simplicity_gate": [
                "Choose simple data structures over complex ones",
                "Prefer composition over inheritance",
                "Use standard patterns over custom solutions",
            ],
        }
        return examples_map.get(article, [])

    def get_all_articles(self) -> list[dict]:
        """Get information about all constitutional articles."""
        return [
            {
                "article": article.value,
                "title": config["title"],
                "description": config["description"],
                "weight": config["weight"],
            }
            for article, config in self.articles.items()
        ]

    def validate_against_template(
        self, text: str, template_requirements: list[str]
    ) -> dict:
        """Validate text against specific template requirements."""
        analysis = self.analyze_compliance(text)

        # Check template-specific requirements
        template_score = 0.0
        missing_requirements = []

        for requirement in template_requirements:
            if requirement.lower() in text.lower():
                template_score += 1.0
            else:
                missing_requirements.append(requirement)

        if template_requirements:
            template_score /= len(template_requirements)

        analysis["template_validation"] = {
            "score": template_score,
            "missing_requirements": missing_requirements,
            "total_requirements": len(template_requirements),
        }

        return analysis

    def generate_compliance_report(self, analysis: dict) -> str:
        """Generate a human-readable compliance report."""
        if not analysis or "overall_score" not in analysis:
            return "Invalid analysis data provided."

        lines = [
            "Constitutional Compliance Report",
            "=" * 35,
            f"Overall Score: {analysis['overall_score']:.2f}/1.00",
            "",
        ]

        # Article scores
        if analysis["article_scores"]:
            lines.append("Article Scores:")
            for article, data in analysis["article_scores"].items():
                score_str = f"{data['score']:.2f}"
                title = data.get("title", article.replace("_", " ").title())
                lines.append(f"  {title}: {score_str}")
            lines.append("")

        # Violations
        if analysis["violations"]:
            lines.append("Areas for Improvement:")
            for violation in analysis["violations"]:
                severity = violation.get("severity", "medium").upper()
                title = violation.get("title", violation["article"])
                lines.append(f"  [{severity}] {title}")
            lines.append("")

        # Recommendations
        if analysis["recommendations"]:
            lines.append("Recommendations:")
            for rec in analysis["recommendations"][:5]:  # Top 5
                title = rec.get("title", "Improve compliance")
                lines.append(f"  • {title}")
            lines.append("")

        return "\n".join(lines)
