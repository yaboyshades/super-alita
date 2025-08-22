#!/usr/bin/env python3
"""
DTA 2.0 Validation Pipeline Module

Multi-layer validation system for the Deep Thinking Architecture.
Provides syntax validation, semantic analysis, confidence assessment,
safety checks, and comprehensive quality assurance for LLM responses.
"""

import ast
import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# PII detection with fallback
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None


class ValidationLevel(Enum):
    """Validation strictness levels."""

    PERMISSIVE = "permissive"
    NORMAL = "normal"
    STRICT = "strict"
    PARANOID = "paranoid"


class ValidationResult(Enum):
    """Validation result status."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    ERROR = "error"


class ValidationType(Enum):
    """Types of validation performed."""

    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    CONFIDENCE = "confidence"
    SAFETY = "safety"
    PII = "pii"
    CONTENT = "content"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during analysis."""

    type: ValidationType
    severity: ValidationResult
    message: str
    location: str | None = None
    suggestion: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert issue to dictionary format."""
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "message": self.message,
            "location": self.location,
            "suggestion": self.suggestion,
            "metadata": self.metadata,
        }


@dataclass
class DTAValidationResult:
    """Comprehensive validation result for DTA processing."""

    overall_status: ValidationResult
    confidence_score: float
    issues: list[ValidationIssue] = field(default_factory=list)
    validation_metadata: dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.overall_status in [ValidationResult.PASS, ValidationResult.WARNING]

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(issue.severity == ValidationResult.WARNING for issue in self.issues)

    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(
            issue.severity in [ValidationResult.FAIL, ValidationResult.ERROR]
            for issue in self.issues
        )

    def get_issues_by_type(
        self, validation_type: ValidationType
    ) -> list[ValidationIssue]:
        """Get all issues of a specific type."""
        return [issue for issue in self.issues if issue.type == validation_type]

    def to_dict(self) -> dict[str, Any]:
        """Convert validation result to dictionary format."""
        return {
            "overall_status": self.overall_status.value,
            "confidence_score": self.confidence_score,
            "is_valid": self.is_valid,
            "has_warnings": self.has_warnings,
            "has_errors": self.has_errors,
            "issues": [issue.to_dict() for issue in self.issues],
            "validation_metadata": self.validation_metadata,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseValidator(ABC):
    """Abstract base class for all validators."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.level = ValidationLevel(self.config.get("level", "normal"))
        self.logger = logging.getLogger(f"dta.validator.{self.__class__.__name__}")

    @abstractmethod
    async def validate(
        self, data: Any, context: dict[str, Any] | None = None
    ) -> list[ValidationIssue]:
        """Perform validation and return list of issues."""

    def create_issue(
        self,
        severity: ValidationResult,
        message: str,
        location: str | None = None,
        suggestion: str | None = None,
        **metadata,
    ) -> ValidationIssue:
        """Helper method to create validation issues."""
        return ValidationIssue(
            type=self.get_validation_type(),
            severity=severity,
            message=message,
            location=location,
            suggestion=suggestion,
            metadata=metadata,
        )

    @abstractmethod
    def get_validation_type(self) -> ValidationType:
        """Return the type of validation this validator performs."""


class SyntaxValidator(BaseValidator):
    """Validates Python code syntax and structure."""

    def get_validation_type(self) -> ValidationType:
        return ValidationType.SYNTAX

    async def validate(
        self, data: Any, context: dict[str, Any] | None = None
    ) -> list[ValidationIssue]:
        """Validate Python code syntax."""
        if not self.enabled:
            return []

        issues = []

        # Handle different data types gracefully
        if isinstance(data, dict):
            # For dict data, try to extract Python code
            code_content = data.get("python_code", "")
            if not code_content and "thinking_trace" in data:
                # Extract code from thinking trace if needed
                content = str(data.get("thinking_trace", ""))
                if "chat(" in content or "use_tool(" in content:
                    code_content = (
                        content.split("\n")[-1] if "\n" in content else content
                    )
            data = code_content if code_content else str(data)
        elif not isinstance(data, str):
            issues.append(
                self.create_issue(
                    ValidationResult.WARNING,  # Changed from ERROR to WARNING
                    f"Unexpected data type for syntax validation: {type(data).__name__}",
                    suggestion="Consider providing string or dict with python_code field",
                )
            )
            return issues

        # Check if it looks like Python code
        if not self._looks_like_python_code(data):
            if self.level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                issues.append(
                    self.create_issue(
                        ValidationResult.WARNING,
                        "Data does not appear to be Python code",
                        suggestion="Ensure the content is valid Python code",
                    )
                )
            return issues

        # Extract Python code blocks
        code_blocks = self._extract_code_blocks(data)

        for i, code_block in enumerate(code_blocks):
            block_issues = await self._validate_code_block(code_block, f"block_{i}")
            issues.extend(block_issues)

        return issues

    def _looks_like_python_code(self, text: str) -> bool:
        """Check if text appears to be Python code."""
        python_indicators = [
            "def ",
            "class ",
            "import ",
            "from ",
            "if ",
            "for ",
            "while ",
            "try:",
            "except:",
            "with ",
            "async def",
            "await ",
            "lambda",
            "    ",
            "\t",  # Indentation patterns
        ]

        return any(indicator in text for indicator in python_indicators)

    def _extract_code_blocks(self, text: str) -> list[str]:
        """Extract Python code blocks from text."""
        # Look for code in triple backticks
        code_block_pattern = r"```python\s*\n(.*?)\n```"
        matches = re.findall(code_block_pattern, text, re.DOTALL)

        if matches:
            return matches

        # Look for code in single backticks
        inline_code_pattern = r"`([^`\n]+)`"
        inline_matches = re.findall(inline_code_pattern, text)

        # Filter for Python-like code
        python_inline = [
            match for match in inline_matches if self._looks_like_python_code(match)
        ]

        if python_inline:
            return python_inline

        # If no code blocks found, treat entire text as potential code
        if self._looks_like_python_code(text):
            return [text]

        return []

    async def _validate_code_block(
        self, code: str, location: str
    ) -> list[ValidationIssue]:
        """Validate a single code block."""
        issues = []

        try:
            # Parse the code
            parsed = ast.parse(code)

            # Additional syntax checks
            await self._check_code_quality(code, parsed, location, issues)

        except SyntaxError as e:
            issues.append(
                self.create_issue(
                    ValidationResult.FAIL,
                    f"Syntax error: {e.msg}",
                    location=f"{location}:line_{e.lineno}",
                    suggestion="Fix syntax error before execution",
                    error_type="SyntaxError",
                    line_number=e.lineno,
                    column=e.offset,
                )
            )

        except Exception as e:
            issues.append(
                self.create_issue(
                    ValidationResult.ERROR,
                    f"Unexpected error during parsing: {e!s}",
                    location=location,
                    error_type=type(e).__name__,
                )
            )

        return issues

    async def _check_code_quality(
        self,
        code: str,
        parsed_ast: ast.AST,
        location: str,
        issues: list[ValidationIssue],
    ):
        """Perform additional code quality checks."""

        # Check for potentially dangerous operations
        dangerous_patterns = [
            (r"\beval\s*\(", "Use of eval() can be dangerous"),
            (r"\bexec\s*\(", "Use of exec() can be dangerous"),
            (r"__import__\s*\(", "Dynamic imports can be risky"),
            (r'\bopen\s*\(.*[\'"]w[\'"]', "File writing detected"),
            (r"\bos\.system\s*\(", "System command execution detected"),
            (r"\bsubprocess\b", "Subprocess execution detected"),
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                severity = (
                    ValidationResult.WARNING
                    if self.level == ValidationLevel.PERMISSIVE
                    else ValidationResult.FAIL
                )
                issues.append(
                    self.create_issue(
                        severity,
                        message,
                        location=location,
                        suggestion="Review code for security implications",
                    )
                )

        # Check for proper function structure
        function_calls = [
            node for node in ast.walk(parsed_ast) if isinstance(node, ast.Call)
        ]

        if not function_calls and self.level in [
            ValidationLevel.STRICT,
            ValidationLevel.PARANOID,
        ]:
            issues.append(
                self.create_issue(
                    ValidationResult.WARNING,
                    "No function calls found in code block",
                    location=location,
                    suggestion="Ensure code contains expected function calls",
                )
            )


class SemanticValidator(BaseValidator):
    """Validates semantic meaning and logical consistency."""

    def get_validation_type(self) -> ValidationType:
        return ValidationType.SEMANTIC

    async def validate(
        self, data: Any, context: dict[str, Any] | None = None
    ) -> list[ValidationIssue]:
        """Validate semantic consistency."""
        if not self.enabled:
            return []

        issues = []

        # Extract reasoning components if available
        thinking_trace = context.get("thinking_trace", "") if context else ""
        reasoning_summary = context.get("reasoning_summary", "") if context else ""
        python_code = context.get("python_code", "") if context else ""

        # Validate reasoning consistency
        if thinking_trace and reasoning_summary:
            consistency_issues = await self._check_reasoning_consistency(
                thinking_trace, reasoning_summary
            )
            issues.extend(consistency_issues)

        # Validate code-reasoning alignment
        if python_code and reasoning_summary:
            alignment_issues = await self._check_code_reasoning_alignment(
                python_code, reasoning_summary
            )
            issues.extend(alignment_issues)

        # Validate logical flow
        if thinking_trace:
            logic_issues = await self._check_logical_flow(thinking_trace)
            issues.extend(logic_issues)

        return issues

    async def _check_reasoning_consistency(
        self, thinking_trace: str, reasoning_summary: str
    ) -> list[ValidationIssue]:
        """Check consistency between thinking trace and reasoning summary."""
        issues = []

        # Extract key concepts from both
        thinking_concepts = self._extract_key_concepts(thinking_trace)
        summary_concepts = self._extract_key_concepts(reasoning_summary)

        # Check for major concept mismatches
        missing_concepts = thinking_concepts - summary_concepts
        if missing_concepts and self.level in [
            ValidationLevel.STRICT,
            ValidationLevel.PARANOID,
        ]:
            issues.append(
                self.create_issue(
                    ValidationResult.WARNING,
                    f"Key concepts from thinking not reflected in summary: {', '.join(missing_concepts)}",
                    location="reasoning_summary",
                    suggestion="Ensure summary captures all key points from thinking",
                )
            )

        return issues

    async def _check_code_reasoning_alignment(
        self, python_code: str, reasoning_summary: str
    ) -> list[ValidationIssue]:
        """Check alignment between generated code and reasoning."""
        issues = []

        # Extract function names from code
        function_pattern = r"\b(\w+)\s*\("
        code_functions = set(re.findall(function_pattern, python_code))

        # Check if reasoning mentions the functions used
        for func in code_functions:
            if func not in reasoning_summary and self.level in [
                ValidationLevel.NORMAL,
                ValidationLevel.STRICT,
                ValidationLevel.PARANOID,
            ]:
                issues.append(
                    self.create_issue(
                        ValidationResult.WARNING,
                        f"Function '{func}' used in code but not explained in reasoning",
                        location="code_reasoning_alignment",
                        suggestion=f"Explain why '{func}' was chosen in the reasoning",
                    )
                )

        return issues

    async def _check_logical_flow(self, thinking_trace: str) -> list[ValidationIssue]:
        """Check logical flow in thinking process."""
        issues = []

        # Look for numbered steps or logical progression
        step_pattern = r"(\d+)\.\s*(.+?)(?=\n\d+\.|\n\n|$)"
        steps = re.findall(step_pattern, thinking_trace, re.DOTALL)

        if len(steps) < 2 and self.level in [
            ValidationLevel.STRICT,
            ValidationLevel.PARANOID,
        ]:
            issues.append(
                self.create_issue(
                    ValidationResult.WARNING,
                    "Thinking process lacks clear logical steps",
                    location="thinking_trace",
                    suggestion="Structure thinking with numbered steps or clear progression",
                )
            )

        # Check for contradictions (basic patterns)
        contradiction_patterns = [
            (r"\bnot\s+\w+.*\n.*\bis\s+\w+", "Potential contradiction detected"),
            (
                r"\bimpossible.*\n.*\bpossible",
                "Contradiction in possibility assessment",
            ),
            (r"\bcannot.*\n.*\bcan\b", "Contradiction in capability assessment"),
        ]

        for pattern, message in contradiction_patterns:
            if re.search(pattern, thinking_trace, re.IGNORECASE):
                issues.append(
                    self.create_issue(
                        ValidationResult.WARNING,
                        message,
                        location="thinking_trace",
                        suggestion="Review reasoning for logical consistency",
                    )
                )

        return issues

    def _extract_key_concepts(self, text: str) -> set[str]:
        """Extract key concepts from text."""
        # Simple keyword extraction
        # In production, this could use NLP libraries for better concept extraction

        # Remove common words and extract meaningful terms
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
        }

        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        concepts = {word for word in words if word not in stopwords}

        return concepts


class ConfidenceValidator(BaseValidator):
    """Validates confidence scores and uncertainty assessment."""

    def get_validation_type(self) -> ValidationType:
        return ValidationType.CONFIDENCE

    async def validate(
        self, data: Any, context: dict[str, Any] | None = None
    ) -> list[ValidationIssue]:
        """Validate confidence assessment."""
        if not self.enabled:
            return []

        issues = []
        confidence_score = context.get("confidence_score", 0.5) if context else 0.5

        # Validate confidence score range
        if not (0.0 <= confidence_score <= 1.0):
            issues.append(
                self.create_issue(
                    ValidationResult.FAIL,
                    f"Confidence score {confidence_score} outside valid range [0.0, 1.0]",
                    location="confidence_score",
                    suggestion="Ensure confidence score is between 0.0 and 1.0",
                )
            )

        # Check for overconfidence patterns
        reasoning_summary = context.get("reasoning_summary", "") if context else ""
        if confidence_score > 0.9 and reasoning_summary:
            uncertainty_indicators = [
                "maybe",
                "perhaps",
                "possibly",
                "uncertain",
                "unclear",
                "ambiguous",
            ]
            if any(
                indicator in reasoning_summary.lower()
                for indicator in uncertainty_indicators
            ):
                issues.append(
                    self.create_issue(
                        ValidationResult.WARNING,
                        "High confidence despite uncertainty indicators in reasoning",
                        location="confidence_assessment",
                        suggestion="Review confidence score given expressed uncertainties",
                    )
                )

        # Check for underconfidence patterns
        if confidence_score < 0.3 and reasoning_summary:
            certainty_indicators = [
                "clearly",
                "obviously",
                "definitely",
                "certainly",
                "sure",
                "confident",
            ]
            if any(
                indicator in reasoning_summary.lower()
                for indicator in certainty_indicators
            ):
                issues.append(
                    self.create_issue(
                        ValidationResult.WARNING,
                        "Low confidence despite certainty indicators in reasoning",
                        location="confidence_assessment",
                        suggestion="Review confidence score given expressed certainties",
                    )
                )

        return issues


class SafetyValidator(BaseValidator):
    """Validates content for safety and security concerns."""

    def get_validation_type(self) -> ValidationType:
        return ValidationType.SAFETY

    async def validate(
        self, data: Any, context: dict[str, Any] | None = None
    ) -> list[ValidationIssue]:
        """Validate safety and security."""
        if not self.enabled:
            return []

        issues = []

        # Convert data to string for analysis
        if isinstance(data, dict):
            text_content = json.dumps(data, default=str)
        else:
            text_content = str(data)

        # Check for sensitive information patterns
        await self._check_sensitive_patterns(text_content, issues)

        # Check for potentially harmful instructions
        await self._check_harmful_content(text_content, issues)

        # Check for code injection patterns
        await self._check_injection_patterns(text_content, issues)

        return issues

    async def _check_sensitive_patterns(self, text: str, issues: list[ValidationIssue]):
        """Check for sensitive information patterns."""

        sensitive_patterns = [
            (r"\b\d{3}-\d{2}-\d{4}\b", "Potential SSN detected"),
            (
                r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                "Potential credit card number detected",
            ),
            (
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "Email address detected",
            ),
            (
                r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
                "Phone number detected",
            ),
            (r"\bpassword\s*[:=]\s*[^\s]+", "Password detected"),
            (r"\bapi[_-]?key\s*[:=]\s*[^\s]+", "API key detected"),
            (r"\btoken\s*[:=]\s*[^\s]+", "Token detected"),
        ]

        for pattern, message in sensitive_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                severity = (
                    ValidationResult.FAIL
                    if self.level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]
                    else ValidationResult.WARNING
                )
                issues.append(
                    self.create_issue(
                        severity,
                        message,
                        location="content_analysis",
                        suggestion="Remove or redact sensitive information",
                    )
                )

    async def _check_harmful_content(self, text: str, issues: list[ValidationIssue]):
        """Check for potentially harmful content."""

        harmful_patterns = [
            (r"\bdelete\s+.*\bfiles?\b", "File deletion instruction detected"),
            (r"\bformat\s+.*\bdrive\b", "Drive formatting instruction detected"),
            (r"\brm\s+-rf\s+/", "Dangerous file removal command detected"),
            (r"\bdel\s+/[sS]\s+", "Dangerous deletion command detected"),
            (r"\bshutdown\s+", "System shutdown command detected"),
            (r"\breboot\s+", "System reboot command detected"),
        ]

        for pattern, message in harmful_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(
                    self.create_issue(
                        ValidationResult.FAIL,
                        message,
                        location="content_analysis",
                        suggestion="Remove potentially harmful instructions",
                    )
                )

    async def _check_injection_patterns(self, text: str, issues: list[ValidationIssue]):
        """Check for code injection patterns."""

        injection_patterns = [
            (r";\s*(drop|delete|update|insert)\s+", "SQL injection pattern detected"),
            (r"<script\b", "Script injection pattern detected"),
            (r"javascript\s*:", "JavaScript injection pattern detected"),
            (r"data\s*:\s*text/html", "HTML injection pattern detected"),
        ]

        for pattern, message in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(
                    self.create_issue(
                        ValidationResult.FAIL,
                        message,
                        location="content_analysis",
                        suggestion="Remove potential injection vectors",
                    )
                )


class PIIValidator(BaseValidator):
    """Validates content for Personally Identifiable Information."""

    def get_validation_type(self) -> ValidationType:
        return ValidationType.PII

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.nlp_model = None

        # Initialize spaCy model if available
        if SPACY_AVAILABLE and self.config.get("use_nlp", True):
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
            except OSError:
                # Model not available, will use pattern matching only
                pass

    async def validate(
        self, data: Any, context: dict[str, Any] | None = None
    ) -> list[ValidationIssue]:
        """Validate for PII content."""
        if not self.enabled:
            return []

        issues = []

        # Convert data to string for analysis
        if isinstance(data, dict):
            text_content = json.dumps(data, default=str)
        else:
            text_content = str(data)

        # Pattern-based PII detection
        await self._check_pii_patterns(text_content, issues)

        # NLP-based entity detection if available
        if self.nlp_model:
            await self._check_nlp_entities(text_content, issues)

        return issues

    async def _check_pii_patterns(self, text: str, issues: list[ValidationIssue]):
        """Check for PII using pattern matching."""

        # Common non-PII phrases to exclude
        common_phrases = {
            "user is",
            "asking for",
            "the user",
            "simple greeting",
            "how can",
            "help you",
            "user wants",
            "user needs",
            "can help",
            "let me",
            "i can",
            "you can",
            "this is",
            "that is",
            "it is",
            "we are",
            "they are",
        }

        pii_patterns = [
            (r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "Potential name detected"),
            (r"\b\d{3}-\d{2}-\d{4}\b", "Social Security Number detected"),
            (
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "Email address detected",
            ),
            (
                r"\b\d{1,5}\s+([A-Za-z\s]+)\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b",
                "Address detected",
            ),
            (
                r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
                "Phone number detected",
            ),
            (
                r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                "Credit card number detected",
            ),
        ]

        for pattern, message in pii_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                matched_text = match.group().lower()

                # Skip if it's a common non-PII phrase
                if any(phrase in matched_text for phrase in common_phrases):
                    continue

                # Skip obviously non-PII patterns for name detection
                if "name detected" in message:
                    # Skip common words that might match name pattern
                    if matched_text in [
                        "user is",
                        "can help",
                        "you can",
                        "let me",
                        "it is",
                    ]:
                        continue
                severity = (
                    ValidationResult.WARNING
                    if self.level == ValidationLevel.PERMISSIVE
                    else ValidationResult.FAIL
                )
                issues.append(
                    self.create_issue(
                        severity,
                        message,
                        location=f"position_{match.start()}",
                        suggestion="Remove or redact PII before processing",
                        matched_text=match.group(),
                    )
                )

    async def _check_nlp_entities(self, text: str, issues: list[ValidationIssue]):
        """Check for PII using NLP entity recognition."""
        if not self.nlp_model:
            return

        try:
            doc = self.nlp_model(text)

            pii_entity_types = {"PERSON", "ORG", "GPE", "MONEY", "DATE", "TIME"}

            for ent in doc.ents:
                if ent.label_ in pii_entity_types:
                    severity = (
                        ValidationResult.WARNING
                        if self.level == ValidationLevel.PERMISSIVE
                        else ValidationResult.FAIL
                    )
                    issues.append(
                        self.create_issue(
                            severity,
                            f"PII entity detected: {ent.label_}",
                            location=f"position_{ent.start_char}",
                            suggestion="Review and redact sensitive entities",
                            entity_type=ent.label_,
                            entity_text=ent.text,
                            confidence=(
                                ent._.confidence
                                if hasattr(ent._, "confidence")
                                else None
                            ),
                        )
                    )

        except Exception as e:
            # NLP processing failed, log but don't fail validation
            self.logger.warning(f"NLP entity detection failed: {e}")


class ValidationPipeline:
    """
    Orchestrates multiple validators for comprehensive validation.

    Provides a unified interface for running all validation checks
    and aggregating results into a comprehensive validation report.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.validation_level = ValidationLevel(self.config.get("level", "normal"))

        # Initialize validators
        self.validators = {
            "syntax": SyntaxValidator(self.config.get("syntax", {})),
            "semantic": SemanticValidator(self.config.get("semantic", {})),
            "confidence": ConfidenceValidator(self.config.get("confidence", {})),
            "safety": SafetyValidator(self.config.get("safety", {})),
            "pii": PIIValidator(self.config.get("pii", {})),
        }

        # Enable/disable validators based on config
        for validator_name, validator_config in self.config.get(
            "validators", {}
        ).items():
            if validator_name in self.validators:
                self.validators[validator_name].enabled = validator_config.get(
                    "enabled", True
                )

        self.logger = logging.getLogger("dta.validation_pipeline")

    async def validate(
        self, data: Any, context: dict[str, Any] | None = None
    ) -> DTAValidationResult:
        """
        Run comprehensive validation pipeline.

        Args:
            data: The data to validate (could be response text, structured data, etc.)
            context: Additional context including thinking_trace, reasoning_summary, etc.

        Returns:
            DTAValidationResult with comprehensive validation results
        """
        start_time = asyncio.get_event_loop().time()

        if not self.enabled:
            return DTAValidationResult(
                overall_status=ValidationResult.PASS,
                confidence_score=1.0,
                validation_metadata={"validation_disabled": True},
            )

        all_issues = []
        validation_metadata = {
            "validation_level": self.validation_level.value,
            "validators_run": [],
            "total_validators": len([v for v in self.validators.values() if v.enabled]),
        }

        # Run each enabled validator
        for validator_name, validator in self.validators.items():
            if not validator.enabled:
                continue

            try:
                self.logger.debug(f"Running {validator_name} validator")
                validator_issues = await validator.validate(data, context)

                all_issues.extend(validator_issues)
                validation_metadata["validators_run"].append(validator_name)
                validation_metadata[f"{validator_name}_issues"] = len(validator_issues)

            except Exception as e:
                self.logger.error(f"Error in {validator_name} validator: {e}")

                # Add error as validation issue
                error_issue = ValidationIssue(
                    type=validator.get_validation_type(),
                    severity=ValidationResult.ERROR,
                    message=f"Validator error: {e!s}",
                    location=f"{validator_name}_validator",
                    metadata={"error_type": type(e).__name__},
                )
                all_issues.append(error_issue)

        # Determine overall status
        overall_status = self._determine_overall_status(all_issues)

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(all_issues, context)

        # Calculate processing time
        processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        # Create final result
        result = DTAValidationResult(
            overall_status=overall_status,
            confidence_score=confidence_score,
            issues=all_issues,
            validation_metadata=validation_metadata,
            processing_time_ms=processing_time_ms,
        )

        self.logger.info(
            f"Validation complete: {overall_status.value}, {len(all_issues)} issues, {processing_time_ms:.1f}ms"
        )

        return result

    def _determine_overall_status(
        self, issues: list[ValidationIssue]
    ) -> ValidationResult:
        """Determine overall validation status from individual issues."""
        if not issues:
            return ValidationResult.PASS

        # Check for blocking issues
        if any(issue.severity == ValidationResult.ERROR for issue in issues):
            return ValidationResult.ERROR

        if any(issue.severity == ValidationResult.FAIL for issue in issues):
            return ValidationResult.FAIL

        if any(issue.severity == ValidationResult.WARNING for issue in issues):
            return ValidationResult.WARNING

        return ValidationResult.PASS

    def _calculate_confidence_score(
        self, issues: list[ValidationIssue], context: dict[str, Any] | None
    ) -> float:
        """Calculate confidence score based on validation results."""

        # Start with baseline confidence from context or default
        base_confidence = context.get("confidence_score", 0.8) if context else 0.8

        # Reduce confidence based on issues
        confidence_penalty = 0.0

        for issue in issues:
            if issue.severity == ValidationResult.ERROR:
                confidence_penalty += 0.3
            elif issue.severity == ValidationResult.FAIL:
                confidence_penalty += 0.2
            elif issue.severity == ValidationResult.WARNING:
                confidence_penalty += 0.1

        # Apply penalty and ensure bounds
        final_confidence = max(0.0, min(1.0, base_confidence - confidence_penalty))

        return final_confidence

    async def validate_response(
        self,
        thinking_trace: str,
        reasoning_summary: str,
        confidence_level: float,
        python_code: str,
    ) -> DTAValidationResult:
        """
        Convenience method for validating complete DTA response.

        Args:
            thinking_trace: The thinking process text
            reasoning_summary: Summary of reasoning
            confidence_level: Confidence score (0.0-1.0)
            python_code: Generated Python code

        Returns:
            DTAValidationResult with comprehensive validation
        """
        # Prepare data and context
        data = {
            "thinking_trace": thinking_trace,
            "reasoning_summary": reasoning_summary,
            "python_code": python_code,
            "confidence_level": confidence_level,
        }

        context = {
            "thinking_trace": thinking_trace,
            "reasoning_summary": reasoning_summary,
            "confidence_score": confidence_level,
            "python_code": python_code,
        }

        return await self.validate(data, context)


# Utility functions for easy validation setup
def create_validation_pipeline(
    config: dict[str, Any] | None = None,
) -> ValidationPipeline:
    """Create a ValidationPipeline instance with default configuration."""
    default_config = {
        "enabled": True,
        "level": "normal",
        "validators": {
            "syntax": {"enabled": True, "level": "normal"},
            "semantic": {"enabled": True, "level": "normal"},
            "confidence": {"enabled": True, "level": "normal"},
            "safety": {"enabled": True, "level": "strict"},
            "pii": {"enabled": True, "level": "strict", "use_nlp": False},
        },
    }

    if config:
        # Deep merge config
        for key, value in config.items():
            if key == "validators" and isinstance(value, dict):
                for validator_name, validator_config in value.items():
                    if validator_name in default_config["validators"]:
                        default_config["validators"][validator_name].update(
                            validator_config
                        )
                    else:
                        default_config["validators"][validator_name] = validator_config
            else:
                default_config[key] = value

    return ValidationPipeline(default_config)


# Example usage and testing
async def example_validation_usage():
    """Example of how to use DTA validation."""

    # Create validation pipeline
    validator = create_validation_pipeline(
        {
            "level": "normal",
            "validators": {
                "safety": {"level": "strict"},
                "pii": {"enabled": True, "use_nlp": False},
            },
        }
    )

    # Example DTA response components
    thinking_trace = """
    1. What is the user asking for?
       - They want to calculate the area of a circle
       - They provided radius = 5

    2. What function should I use?
       - I need a calculator tool
       - Formula: π * r²

    3. Am I confident in this approach?
       - Yes, this is a straightforward mathematical calculation
    """

    reasoning_summary = (
        "User wants to calculate circle area with radius 5 using π * r² formula"
    )
    confidence_level = 0.9
    python_code = 'use_tool("calculator", expression="3.14159 * 5 * 5")'

    # Run validation
    result = await validator.validate_response(
        thinking_trace, reasoning_summary, confidence_level, python_code
    )

    print(f"Validation result: {result.overall_status.value}")
    print(f"Confidence score: {result.confidence_score}")
    print(f"Issues found: {len(result.issues)}")

    for issue in result.issues:
        print(f"- {issue.severity.value}: {issue.message}")

    print(f"Processing time: {result.processing_time_ms:.1f}ms")

    return result


if __name__ == "__main__":
    # Run example
    asyncio.run(example_validation_usage())
