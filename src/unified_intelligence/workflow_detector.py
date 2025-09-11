"""
SDD Workflow Pattern Detection

Detects and classifies user inputs according to SDD workflow patterns:
- /new_feature - Feature specification requests
- /generate_plan - Implementation planning requests
- Constitutional review requests
- Code quality and testing patterns
- General development guidance
"""

import re
from enum import Enum


class WorkflowPattern(Enum):
    """Enumeration of supported SDD workflow patterns."""

    NEW_FEATURE = "new_feature"
    GENERATE_PLAN = "generate_plan"
    SPECIFICATION = "specification"
    CONSTITUTIONAL = "constitutional"
    TEMPLATE = "template"
    TEST_FIRST = "test_first"
    LIBRARY_FIRST = "library_first"
    SIMPLICITY = "simplicity"
    CLI_INTERFACE = "cli_interface"
    INTEGRATION_FIRST = "integration_first"
    GENERAL = "general"


class WorkflowDetector:
    """
    Detects SDD workflow patterns from user input.

    Provides pattern detection with confidence scoring and context extraction.
    """

    def __init__(self):
        """Initialize the workflow detector with pattern definitions."""
        self.patterns = {
            WorkflowPattern.NEW_FEATURE: [
                r"/new_feature\b",
                r"\bnew_feature\b",
                r"\bcreate\s+feature\b",
                r"\bnew\s+feature\b",
                r"\bimplement\s+feature\b",
                r"\bbuild\s+feature\b",
            ],
            WorkflowPattern.GENERATE_PLAN: [
                r"/generate_plan\b",
                r"\bgenerate_plan\b",
                r"\bimplementation\s+plan\b",
                r"\bplan\s+implementation\b",
                r"\barchitecture\s+plan\b",
                r"\bdesign\s+plan\b",
            ],
            WorkflowPattern.SPECIFICATION: [
                r"\bspecification\b",
                r"\bspec\b",
                r"\brequirements\b",
                r"\bprd\b",
                r"\buser\s+story\b",
                r"\bacceptance\s+criteria\b",
            ],
            WorkflowPattern.CONSTITUTIONAL: [
                r"\bconstitutional\b",
                r"\bconstitution\b",
                r"\barticle\b",
                r"\bgate\b",
                r"\bcompliance\b",
                r"\bprinciples\b",
            ],
            WorkflowPattern.TEMPLATE: [
                r"\btemplate\b",
                r"\bconstraint\b",
                r"\bneeds\s+clarification\b",
                r"\b\[NEEDS\s+CLARIFICATION\]\b",
                r"\bstructured\s+approach\b",
            ],
            WorkflowPattern.TEST_FIRST: [
                r"\btest\b",
                r"\btesting\b",
                r"\bcoverage\b",
                r"\btdd\b",
                r"\btest[-_]first\b",
                r"\bunit\s+test\b",
            ],
            WorkflowPattern.LIBRARY_FIRST: [
                r"\blibrary\b",
                r"\bpackage\b",
                r"\bdependency\b",
                r"\bframework\b",
                r"\bmodule\b",
                r"\breusable\b",
            ],
            WorkflowPattern.SIMPLICITY: [
                r"\bcomplex\b",
                r"\bsimplify\b",
                r"\brefactor\b",
                r"\bsimple\b",
                r"\bcomplexity\b",
                r"\bover[-_]engineering\b",
            ],
            WorkflowPattern.CLI_INTERFACE: [
                r"\bcli\b",
                r"\bcommand\s+line\b",
                r"\binterface\b",
                r"\bargparse\b",
                r"\bclick\b",
                r"\bterminal\b",
            ],
            WorkflowPattern.INTEGRATION_FIRST: [
                r"\bintegration\b",
                r"\bcontract\b",
                r"\breal\s+database\b",
                r"\bapi\s+integration\b",
                r"\bend[-_]to[-_]end\b",
                r"\be2e\b",
            ],
        }

        # Pattern priority order (higher priority patterns checked first)
        self.priority_order = [
            WorkflowPattern.NEW_FEATURE,
            WorkflowPattern.GENERATE_PLAN,
            WorkflowPattern.CONSTITUTIONAL,
            WorkflowPattern.SPECIFICATION,
            WorkflowPattern.TEMPLATE,
            WorkflowPattern.TEST_FIRST,
            WorkflowPattern.LIBRARY_FIRST,
            WorkflowPattern.SIMPLICITY,
            WorkflowPattern.CLI_INTERFACE,
            WorkflowPattern.INTEGRATION_FIRST,
        ]

    def detect(self, user_input: str) -> str:
        """
        Detect the primary workflow pattern from user input.

        Args:
            user_input: User's input text

        Returns:
            String name of detected pattern
        """
        pattern_info = self.detect_with_confidence(user_input)
        return pattern_info["pattern"]

    def detect_with_confidence(self, user_input: str) -> dict:
        """
        Detect workflow pattern with confidence scoring and metadata.

        Args:
            user_input: User's input text

        Returns:
            Dictionary with pattern, confidence, matches, and metadata
        """
        user_lower = user_input.lower().strip()

        # Track all pattern matches with scores
        pattern_scores = {}
        pattern_matches = {}

        for pattern in self.priority_order:
            patterns = self.patterns[pattern]
            matches = []
            score = 0

            for regex in patterns:
                found_matches = re.findall(regex, user_lower)
                if found_matches:
                    matches.extend(found_matches)
                    # Higher score for exact command matches
                    if regex.startswith(r"/"):
                        score += 10
                    else:
                        score += 1

            if matches:
                pattern_scores[pattern] = score
                pattern_matches[pattern] = matches

        # Determine best match
        if pattern_scores:
            best_pattern = max(pattern_scores.keys(), key=lambda p: pattern_scores[p])
            confidence = min(1.0, pattern_scores[best_pattern] / 5.0)  # Normalize
        else:
            best_pattern = WorkflowPattern.GENERAL
            confidence = 0.0

        return {
            "pattern": best_pattern.value,
            "confidence": confidence,
            "matches": pattern_matches.get(best_pattern, []),
            "all_scores": {p.value: s for p, s in pattern_scores.items()},
            "input_length": len(user_input),
            "normalized_input": user_lower,
        }

    def detect_multiple(self, user_input: str, threshold: float = 0.3) -> list:
        """
        Detect multiple workflow patterns above confidence threshold.

        Args:
            user_input: User's input text
            threshold: Minimum confidence for pattern inclusion

        Returns:
            List of patterns with confidence scores
        """
        detection_info = self.detect_with_confidence(user_input)
        all_scores = detection_info["all_scores"]

        results = []
        for pattern, raw_score in all_scores.items():
            confidence = min(1.0, raw_score / 5.0)
            if confidence >= threshold:
                results.append(
                    {
                        "pattern": pattern,
                        "confidence": confidence,
                        "raw_score": raw_score,
                    }
                )

        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def get_pattern_description(self, pattern: str) -> str:
        """Get human-readable description of a workflow pattern."""
        descriptions = {
            "new_feature": "Feature specification and requirements definition",
            "generate_plan": "Implementation planning and architecture design",
            "specification": "Requirements and specification documentation",
            "constitutional": "Constitutional compliance and principle validation",
            "template": "Template-driven development and structured approaches",
            "test_first": "Test-driven development and quality assurance",
            "library_first": "Library-based design and reusable components",
            "simplicity": "Simplicity principles and complexity management",
            "cli_interface": "Command-line interface design and development",
            "integration_first": "Integration testing and system validation",
            "general": "General development guidance and assistance",
        }
        return descriptions.get(pattern, "Unknown workflow pattern")

    def get_supported_patterns(self) -> list:
        """Get list of all supported workflow patterns."""
        return [
            {
                "pattern": pattern.value,
                "description": self.get_pattern_description(pattern.value),
                "regex_count": len(self.patterns[pattern]),
            }
            for pattern in WorkflowPattern
        ]

    def validate_pattern(self, pattern: str) -> bool:
        """Check if a pattern name is valid."""
        try:
            WorkflowPattern(pattern)
            return True
        except ValueError:
            return False

    def get_pattern_examples(self, pattern: str) -> list:
        """Get example inputs for a specific pattern."""
        examples = {
            "new_feature": [
                "/new_feature User authentication system",
                "Create a new chat feature",
                "Implement real-time notifications",
            ],
            "generate_plan": [
                "/generate_plan WebSocket messaging with Redis",
                "Create implementation plan for user profiles",
                "Design architecture for file upload system",
            ],
            "constitutional": [
                "Check constitutional compliance",
                "Review against SDD principles",
                "Validate article compliance",
            ],
            "test_first": [
                "How do I implement TDD for this API?",
                "Write tests for user authentication",
                "Test coverage for payment system",
            ],
            "library_first": [
                "Should I use an existing library for this?",
                "Find reusable components for data processing",
                "Design as a standalone library",
            ],
        }
        return examples.get(pattern, [])

    def extract_entities(self, user_input: str, pattern: str) -> dict:
        """Extract relevant entities based on detected pattern."""
        entities = {}

        if pattern == "new_feature":
            # Extract feature name/description
            feature_match = re.search(
                r"(?:new_feature|feature|implement|create|build)\s+(.+?)(?:\s|$)",
                user_input.lower(),
            )
            if feature_match:
                entities["feature_name"] = feature_match.group(1).strip()

        elif pattern == "generate_plan":
            # Extract technology/component mentions
            tech_words = re.findall(
                r"\b(?:websocket|redis|database|api|rest|graphql|react|vue|"
                r"python|node|docker|kubernetes)\b",
                user_input.lower(),
            )
            if tech_words:
                entities["technologies"] = list(set(tech_words))

        # Extract general entities
        entities.update(
            {
                "mentions_testing": bool(re.search(r"\btest\b", user_input.lower())),
                "mentions_database": bool(
                    re.search(r"\bdatabase\b", user_input.lower())
                ),
                "mentions_api": bool(re.search(r"\bapi\b", user_input.lower())),
                "has_clarification_request": "[NEEDS CLARIFICATION]" in user_input,
            }
        )

        return entities
