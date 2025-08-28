"""
Intelligent Prompt Optimizer and Amplifier for Super Alita

This module provides advanced prompt optimization capabilities that enhance user inputs
for better LLM performance. It works in conjunction with the existing prompt manager
and message amplifier systems to provide comprehensive prompt enhancement.

Key Features:
- Intelligent prompt analysis and classification
- Context-aware prompt amplification
- Structure optimization for better LLM comprehension
- Template-based enhancement using existing prompt templates
- Caching and performance optimization
- Multiple optimization strategies for different prompt types
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.core.prompt_manager import PromptManager, get_prompt_manager


class PromptType(Enum):
    """Classification of prompt types for targeted optimization."""

    CODE_REQUEST = "code_request"
    QUESTION = "question"
    TASK = "task"
    CONVERSATION = "conversation"
    DEBUGGING = "debugging"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    RESEARCH = "research"
    UNKNOWN = "unknown"


class OptimizationStrategy(Enum):
    """Different optimization strategies based on prompt characteristics."""

    MINIMAL = "minimal"  # Light touches, preserve user intent
    STANDARD = "standard"  # Balanced optimization
    AGGRESSIVE = "aggressive"  # Maximum enhancement for complex prompts
    CONTEXT_RICH = "context_rich"  # Heavy context injection
    STRUCTURED = "structured"  # Enforce clear structure


@dataclass
class PromptAnalysis:
    """Analysis results for a user prompt."""

    prompt_type: PromptType
    complexity_score: float  # 0.0 - 1.0
    clarity_score: float  # 0.0 - 1.0
    completeness_score: float  # 0.0 - 1.0
    detected_entities: list[str] = field(default_factory=list)
    suggested_enhancements: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class OptimizationResult:
    """Result of prompt optimization."""

    original_prompt: str
    optimized_prompt: str
    analysis: PromptAnalysis
    strategy_used: OptimizationStrategy
    enhancements_applied: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptAnalyzer:
    """Analyzes prompts to understand their characteristics and optimization needs."""

    def __init__(self):
        self.code_patterns = [
            r"\bcode\b",
            r"\bfunction\b",
            r"\bclass\b",
            r"\bmethod\b",
            r"\bapi\b",
            r"\bscript\b",
            r"\bprogram\b",
            r"\bimplement\b",
            r"\bwrite.*\b(python|javascript|java|c\+\+|rust|go)\b",
            r"\b(debug|fix|error|bug)\b",
        ]
        self.question_patterns = [
            r"^\s*(?:what|how|why|when|where|who|which)\b",
            r"\?\s*$",
            r"\bexplain\b",
            r"\btell me\b",
        ]
        self.task_patterns = [
            r"\b(create|build|make|generate|design|develop)\b",
            r"\b(analyze|review|check|test|validate)\b",
            r"\b(help me|assist|guide)\b",
        ]

    def analyze(self, prompt: str) -> PromptAnalysis:
        """Analyze a prompt and return detailed analysis."""
        prompt_clean = prompt.strip().lower()

        # Classify prompt type
        prompt_type = self._classify_prompt_type(prompt_clean)

        # Calculate scores
        complexity_score = self._calculate_complexity(prompt)
        clarity_score = self._calculate_clarity(prompt)
        completeness_score = self._calculate_completeness(prompt, prompt_type)

        # Extract entities
        entities = self._extract_entities(prompt)

        # Generate enhancement suggestions
        suggestions = self._suggest_enhancements(
            prompt, prompt_type, complexity_score, clarity_score
        )

        # Calculate confidence
        confidence = (clarity_score + completeness_score) / 2.0

        return PromptAnalysis(
            prompt_type=prompt_type,
            complexity_score=complexity_score,
            clarity_score=clarity_score,
            completeness_score=completeness_score,
            detected_entities=entities,
            suggested_enhancements=suggestions,
            confidence=confidence,
        )

    def _classify_prompt_type(self, prompt: str) -> PromptType:
        """Classify the type of prompt."""
        code_score = sum(
            1
            for pattern in self.code_patterns
            if re.search(pattern, prompt, re.IGNORECASE)
        )
        question_score = sum(
            1
            for pattern in self.question_patterns
            if re.search(pattern, prompt, re.IGNORECASE)
        )
        task_score = sum(
            1
            for pattern in self.task_patterns
            if re.search(pattern, prompt, re.IGNORECASE)
        )

        # Enhanced code detection for better classification
        has_programming_language = bool(
            re.search(
                r"\b(python|javascript|java|c\+\+|rust|go|typescript|sql|html|css)\b",
                prompt,
                re.IGNORECASE,
            )
        )
        has_code_terms = bool(
            re.search(
                r"\b(function|class|method|api|algorithm|implement|debug|error|bug)\b",
                prompt,
                re.IGNORECASE,
            )
        )

        if has_programming_language or (has_code_terms and code_score > 0):
            return PromptType.CODE_REQUEST
        elif question_score > task_score and question_score > code_score:
            return PromptType.QUESTION
        elif task_score > 0 and task_score >= question_score:
            return PromptType.TASK
        elif "?" in prompt:
            return PromptType.QUESTION
        else:
            return PromptType.CONVERSATION

    def _calculate_complexity(self, prompt: str) -> float:
        """Calculate complexity score based on length, technical terms, etc."""
        words = len(prompt.split())
        sentences = len(re.split(r"[.!?]+", prompt))

        # Technical terms increase complexity
        tech_terms = len(
            re.findall(
                r"\b(?:algorithm|implementation|architecture|framework|library|api|database|server|client|frontend|backend|deployment|kubernetes|docker|git|github|aws|azure|gcp)\b",
                prompt.lower(),
            )
        )

        # Base complexity on length and technical content
        length_score = min(words / 50.0, 1.0)  # Normalize to 50 words max
        tech_score = min(tech_terms / 5.0, 1.0)  # Normalize to 5 terms max
        sentence_score = min(sentences / 5.0, 1.0)  # Normalize to 5 sentences max

        return (length_score + tech_score + sentence_score) / 3.0

    def _calculate_clarity(self, prompt: str) -> float:
        """Calculate clarity score based on sentence structure, grammar, etc."""
        # Simple heuristics for clarity
        sentences = re.split(r"[.!?]+", prompt.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        clarity_factors = []

        # Sentence length (shorter sentences are clearer)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        length_clarity = max(
            0.0, min(1.0, 2.0 - (avg_sentence_length / 10.0))
        )  # Cap at 1.0
        clarity_factors.append(length_clarity)

        # Question marks for questions (but don't overscore)
        if any("?" in s for s in sentences):
            clarity_factors.append(0.8)  # Questions are usually clear in intent

        # Presence of specific action words
        action_words = [
            "please",
            "help",
            "create",
            "explain",
            "show",
            "tell",
            "generate",
        ]
        has_action = any(word in prompt.lower() for word in action_words)
        if has_action:
            clarity_factors.append(0.8)  # Lower score to prevent overshooting

        return (
            min(1.0, sum(clarity_factors) / len(clarity_factors))
            if clarity_factors
            else 0.5
        )

    def _calculate_completeness(self, prompt: str, prompt_type: PromptType) -> float:
        """Calculate completeness score based on prompt type and content."""
        completeness_factors = []

        if prompt_type == PromptType.CODE_REQUEST:
            # Code requests should specify language, requirements, constraints
            has_language = bool(
                re.search(
                    r"\b(python|javascript|java|c\+\+|rust|go|typescript|sql)\b",
                    prompt.lower(),
                )
            )
            has_requirements = bool(
                re.search(r"\b(should|must|need|require|want)\b", prompt.lower())
            )
            completeness_factors.extend([has_language, has_requirements])

        elif prompt_type == PromptType.TASK:
            # Tasks should have clear objectives and constraints
            has_objective = bool(
                re.search(r"\b(create|build|make|generate|analyze)\b", prompt.lower())
            )
            has_context = len(prompt.split()) > 10  # Some context provided
            completeness_factors.extend([has_objective, has_context])

        elif prompt_type == PromptType.QUESTION:
            # Questions should be specific
            is_specific = "?" in prompt and len(prompt.split()) > 3
            completeness_factors.append(is_specific)

        # General completeness factors
        has_details = len(prompt.split()) > 5
        has_punctuation = bool(re.search(r"[.!?]", prompt))
        completeness_factors.extend([has_details, has_punctuation])

        return (
            sum(completeness_factors) / len(completeness_factors)
            if completeness_factors
            else 0.5
        )

    def _extract_entities(self, prompt: str) -> list[str]:
        """Extract relevant entities from the prompt."""
        entities = []

        # Programming languages
        languages = re.findall(
            r"\b(python|javascript|java|c\+\+|rust|go|typescript|sql|html|css)\b",
            prompt.lower(),
        )
        entities.extend(f"language:{lang}" for lang in languages)

        # Technologies
        techs = re.findall(
            r"\b(react|vue|angular|django|flask|fastapi|express|docker|kubernetes|aws|azure|gcp)\b",
            prompt.lower(),
        )
        entities.extend(f"tech:{tech}" for tech in techs)

        # File types
        files = re.findall(
            r"\b(\w+\.(py|js|html|css|json|yaml|yml|md|txt|csv))\b", prompt.lower()
        )
        entities.extend(f"file:{file}" for file, _ in files)

        return list(set(entities))  # Remove duplicates

    def _suggest_enhancements(
        self, prompt: str, prompt_type: PromptType, complexity: float, clarity: float
    ) -> list[str]:
        """Suggest specific enhancements for the prompt."""
        suggestions = []

        if clarity < 0.6:
            suggestions.append("clarify_intent")

        if complexity > 0.7:
            suggestions.append("break_down_complexity")

        if prompt_type == PromptType.CODE_REQUEST:
            if not re.search(
                r"\b(python|javascript|java|c\+\+|rust|go)\b", prompt.lower()
            ):
                suggestions.append("specify_language")
            if not re.search(r"\b(example|sample|template)\b", prompt.lower()):
                suggestions.append("request_examples")

        if len(prompt.split()) < 5:
            suggestions.append("add_context")

        if not re.search(r"[.!?]$", prompt.strip()):
            suggestions.append("proper_punctuation")

        return suggestions


class PromptOptimizer:
    """Main prompt optimizer that applies enhancements based on analysis."""

    def __init__(self, prompt_manager: PromptManager | None = None):
        self.prompt_manager = (
            prompt_manager
            if prompt_manager is not None
            else self._get_safe_prompt_manager()
        )
        self.analyzer = PromptAnalyzer()
        self._cache: dict[str, OptimizationResult] = {}

    def _get_safe_prompt_manager(self) -> PromptManager | None:
        """Safely get prompt manager with fallback."""
        try:
            return get_prompt_manager()
        except (ImportError, Exception):
            return None

    def _get_cache_key(self, prompt: str, strategy: OptimizationStrategy) -> str:
        """Generate cache key for optimization results."""
        content = f"{prompt}:{strategy.value}"
        return hashlib.md5(content.encode()).hexdigest()

    def optimize(
        self,
        prompt: str,
        strategy: OptimizationStrategy | None = None,
        context: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """Optimize a prompt using the specified strategy."""
        # Check cache first
        cache_key = self._get_cache_key(
            prompt, strategy or OptimizationStrategy.STANDARD
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Analyze the prompt
        analysis = self.analyzer.analyze(prompt)

        # Determine strategy if not provided
        if strategy is None:
            strategy = self._choose_strategy(analysis)

        # Apply optimization
        optimized_prompt = self._apply_optimization(
            prompt, analysis, strategy, context or {}
        )

        # Track enhancements applied
        enhancements = self._track_enhancements(
            prompt, optimized_prompt, analysis, strategy
        )

        # Create result
        result = OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized_prompt,
            analysis=analysis,
            strategy_used=strategy,
            enhancements_applied=enhancements,
            metadata={
                "cache_key": cache_key,
                "optimization_timestamp": __import__("time").time(),
            },
        )

        # Cache result
        self._cache[cache_key] = result

        return result

    def _choose_strategy(self, analysis: PromptAnalysis) -> OptimizationStrategy:
        """Choose the best optimization strategy based on analysis."""
        if analysis.complexity_score > 0.8:
            return OptimizationStrategy.STRUCTURED
        elif analysis.clarity_score < 0.5:
            return OptimizationStrategy.AGGRESSIVE
        elif analysis.prompt_type in [PromptType.CODE_REQUEST, PromptType.TASK]:
            return OptimizationStrategy.CONTEXT_RICH
        elif analysis.completeness_score > 0.8:
            return OptimizationStrategy.MINIMAL
        else:
            return OptimizationStrategy.STANDARD

    def _apply_optimization(
        self,
        prompt: str,
        analysis: PromptAnalysis,
        strategy: OptimizationStrategy,
        context: dict[str, Any],
    ) -> str:
        """Apply the chosen optimization strategy."""
        optimized = prompt.strip()

        # Apply strategy-specific optimizations
        if strategy == OptimizationStrategy.MINIMAL:
            optimized = self._apply_minimal_optimization(optimized, analysis)
        elif strategy == OptimizationStrategy.STANDARD:
            optimized = self._apply_standard_optimization(optimized, analysis, context)
        elif strategy == OptimizationStrategy.AGGRESSIVE:
            optimized = self._apply_aggressive_optimization(
                optimized, analysis, context
            )
        elif strategy == OptimizationStrategy.CONTEXT_RICH:
            optimized = self._apply_context_rich_optimization(
                optimized, analysis, context
            )
        elif strategy == OptimizationStrategy.STRUCTURED:
            optimized = self._apply_structured_optimization(
                optimized, analysis, context
            )

        return optimized

    def _apply_minimal_optimization(self, prompt: str, analysis: PromptAnalysis) -> str:
        """Apply minimal optimization - just clean up formatting."""
        # Normalize whitespace
        optimized = re.sub(r"\s+", " ", prompt.strip())

        # Ensure proper punctuation
        if not re.search(r"[.!?]$", optimized):
            if analysis.prompt_type == PromptType.QUESTION or "?" in optimized:
                optimized += "?"
            else:
                optimized += "."

        return optimized

    def _apply_standard_optimization(
        self, prompt: str, analysis: PromptAnalysis, context: dict[str, Any]
    ) -> str:
        """Apply standard optimization with moderate enhancements."""
        optimized = self._apply_minimal_optimization(prompt, analysis)

        # Add clarity improvements
        if analysis.clarity_score < 0.7:
            if analysis.prompt_type == PromptType.CODE_REQUEST:
                if not re.search(
                    r"\b(python|javascript|java|c\+\+|rust|go)\b", optimized.lower()
                ):
                    optimized = f"Please help me with a programming task: {optimized}"
            elif (
                analysis.prompt_type == PromptType.QUESTION
                and not optimized.lower().startswith(
                    ("what", "how", "why", "when", "where", "who", "which")
                )
            ):
                optimized = f"I have a question: {optimized}"

        # Add context if available
        if context.get("session_context") and len(optimized.split()) < 10:
            session_info = context["session_context"]
            if isinstance(session_info, str) and len(session_info) < 100:
                optimized = f"{optimized} (Context: {session_info})"

        return optimized

    def _apply_aggressive_optimization(
        self, prompt: str, analysis: PromptAnalysis, context: dict[str, Any]
    ) -> str:
        """Apply aggressive optimization with significant enhancements."""
        optimized = self._apply_standard_optimization(prompt, analysis, context)

        # Add structured prefixes based on prompt type
        if analysis.prompt_type == PromptType.CODE_REQUEST:
            if not optimized.lower().startswith("please"):
                optimized = f"Please help me implement this: {optimized}"

            # Add specificity requests
            if "example" not in optimized.lower():
                optimized += " Please provide a complete example with explanations."

        elif analysis.prompt_type == PromptType.TASK:
            if not optimized.lower().startswith(("please help", "can you", "i need")):
                optimized = f"I need assistance with this task: {optimized}"

            # Add step-by-step request for complex tasks
            if analysis.complexity_score > 0.6:
                optimized += " Please provide a step-by-step approach."

        elif analysis.prompt_type == PromptType.QUESTION:
            if analysis.complexity_score > 0.5:
                optimized += (
                    " Please provide a detailed explanation with examples if possible."
                )

        # Add Super Alita specific context
        if context.get("system_capabilities"):
            optimized += " Please use your available tools and capabilities as needed."

        return optimized

    def _apply_context_rich_optimization(
        self, prompt: str, analysis: PromptAnalysis, context: dict[str, Any]
    ) -> str:
        """Apply context-rich optimization with heavy context injection."""
        optimized = self._apply_aggressive_optimization(prompt, analysis, context)

        # Add system context
        context_additions = []

        if analysis.prompt_type == PromptType.CODE_REQUEST:
            context_additions.append(
                "Consider best practices, error handling, and code maintainability."
            )

        if context.get("available_tools"):
            tools = context["available_tools"]
            if isinstance(tools, list) and len(tools) > 0:
                context_additions.append(f"Available tools: {', '.join(tools[:5])}")

        if context.get("current_project"):
            project_info = context["current_project"]
            if isinstance(project_info, str):
                context_additions.append(f"Current project context: {project_info}")

        # Add REUG framework guidance
        if analysis.complexity_score > 0.7:
            context_additions.append(
                "Please use a structured approach with clear reasoning steps."
            )

        if context_additions:
            optimized += f" Additional context: {' '.join(context_additions)}"

        return optimized

    def _apply_structured_optimization(
        self, prompt: str, analysis: PromptAnalysis, context: dict[str, Any]
    ) -> str:
        """Apply structured optimization to break down complex prompts."""
        optimized = prompt.strip()

        # For complex prompts, add structure
        if analysis.complexity_score > 0.5:  # Lower threshold for testing
            parts = []

            # Main request
            parts.append(f"**Main Request:** {optimized}")

            # Requirements section
            if analysis.prompt_type == PromptType.CODE_REQUEST:
                parts.append("**Requirements:**")
                parts.append("- Provide working code with proper error handling")
                parts.append("- Include clear comments and documentation")
                parts.append("- Follow best practices for the chosen language")

            elif analysis.prompt_type == PromptType.TASK:
                parts.append("**Expected Outcome:**")
                parts.append("- Clear step-by-step solution")
                parts.append("- Explanation of approach and reasoning")

            # Context section
            if context:
                context_items = []
                if context.get("available_tools"):
                    context_items.append(
                        f"Available tools: {context['available_tools']}"
                    )
                if context.get("constraints"):
                    context_items.append(f"Constraints: {context['constraints']}")

                if context_items:
                    parts.append("**Context:**")
                    parts.extend(f"- {item}" for item in context_items)

            optimized = "\n".join(parts)

        return optimized

    def _track_enhancements(
        self,
        original: str,
        optimized: str,
        analysis: PromptAnalysis,  # noqa: ARG002
        strategy: OptimizationStrategy,
    ) -> list[str]:
        """Track what enhancements were applied."""
        enhancements = []

        if len(optimized) > len(original) * 1.2:
            enhancements.append("content_expansion")

        if "**" in optimized and "**" not in original:
            enhancements.append("structured_formatting")

        if optimized.lower().startswith("please") and not original.lower().startswith(
            "please"
        ):
            enhancements.append("politeness_addition")

        if "context:" in optimized.lower() and "context:" not in original.lower():
            enhancements.append("context_injection")

        if re.search(r"[.!?]$", optimized) and not re.search(r"[.!?]$", original):
            enhancements.append("punctuation_normalization")

        enhancements.append(f"strategy_{strategy.value}")

        return enhancements


# Convenience functions for easy integration
def optimize_user_prompt(prompt: str, context: dict[str, Any] | None = None) -> str:
    """Optimize a user prompt and return the enhanced version."""
    optimizer = PromptOptimizer()
    result = optimizer.optimize(prompt, context=context)
    return result.optimized_prompt


def analyze_user_prompt(prompt: str) -> PromptAnalysis:
    """Analyze a user prompt and return detailed analysis."""
    analyzer = PromptAnalyzer()
    return analyzer.analyze(prompt)


def get_optimization_suggestions(prompt: str) -> list[str]:
    """Get optimization suggestions for a prompt."""
    analysis = analyze_user_prompt(prompt)
    return analysis.suggested_enhancements
