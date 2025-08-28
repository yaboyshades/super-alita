"""Enhanced message amplifier middleware with intelligent optimization.

Provides both lightweight normalization and advanced prompt optimization capabilities.
Supports bypass mechanisms and integrates with the prompt optimizer for enhanced
user input processing. The amplifier can operate in different modes based on
configuration and user preferences.
"""

from __future__ import annotations

import os
import re
from typing import Any

from src.reug_runtime.message_mw import MessageContext, register

# Optional import of the prompt optimizer
try:
    from src.prompt.optimizer import (
        OptimizationStrategy,
        PromptOptimizer,
        analyze_user_prompt,
    )

    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False

_WS_RE = re.compile(r"\s+", re.MULTILINE)

# Configuration from environment
AMPLIFIER_MODE = os.getenv(
    "AMPLIFIER_MODE", "standard"
)  # minimal, standard, aggressive
ENABLE_INTELLIGENT_OPTIMIZATION = (
    os.getenv("ENABLE_INTELLIGENT_OPTIMIZATION", "true").lower() == "true"
)
MAX_OPTIMIZATION_LENGTH = int(os.getenv("MAX_OPTIMIZATION_LENGTH", "1000"))


def _normalize(msg: str) -> str:
    """Basic whitespace normalization."""
    msg = msg.strip()
    msg = _WS_RE.sub(" ", msg)
    return msg


def _should_optimize(message: str, ctx: MessageContext) -> bool:  # noqa: ARG001
    """Determine if message should undergo intelligent optimization."""
    if not ENABLE_INTELLIGENT_OPTIMIZATION or not OPTIMIZER_AVAILABLE:
        return False

    # Skip optimization for very short messages
    if len(message.strip()) < 10:
        return False

    # Skip optimization for very long messages to avoid performance issues
    if len(message) > MAX_OPTIMIZATION_LENGTH:
        return False

    # Check for bypass indicators
    bypass_indicators = ["noopt:", "raw:", "literal:"]
    return not any(
        message.lower().startswith(indicator) for indicator in bypass_indicators
    )


def _get_optimization_context(ctx: MessageContext) -> dict[str, Any]:
    """Extract relevant context for optimization from message context."""
    optimization_context = {}

    # Add session context if available
    if hasattr(ctx, "session_id") and ctx.session_id:
        optimization_context["session_id"] = ctx.session_id

    # Add any available system capabilities
    if hasattr(ctx, "available_tools") and ctx.available_tools:
        optimization_context["available_tools"] = ctx.available_tools

    # Add current project context if available
    if hasattr(ctx, "project_context") and ctx.project_context:
        optimization_context["current_project"] = ctx.project_context

    # Add system capabilities flag
    optimization_context["system_capabilities"] = True

    return optimization_context


def amplify_message(message: str, ctx: MessageContext) -> tuple[str, dict[str, str]]:
    """
    Amplify and optimize a user message based on configuration and context.

    Args:
        message: The user's input message
        ctx: Message context containing session and system information

    Returns:
        Tuple of (optimized_message, metadata)
    """
    # Handle explicit bypass
    bypass_indicators = ["noopt:", "raw:", "literal:"]
    for indicator in bypass_indicators:
        if message.lower().startswith(indicator):
            clean_msg = message[len(indicator) :].lstrip()
            return clean_msg, {
                "step": "amplify",
                "bypass": "true",
                "bypass_reason": indicator.rstrip(":"),
            }

    original_len = len(message)

    # Start with basic normalization
    normalized = _normalize(message)

    # Initialize metadata
    meta = {
        "step": "amplify",
        "bypass": "false",
        "len_in": str(original_len),
        "amplifier_mode": AMPLIFIER_MODE,
    }

    # Apply intelligent optimization if enabled and conditions are met
    if _should_optimize(message, ctx) and AMPLIFIER_MODE != "minimal":
        try:
            # Get optimization context
            opt_context = _get_optimization_context(ctx)

            # Determine optimization strategy based on amplifier mode
            if AMPLIFIER_MODE == "aggressive":
                strategy = OptimizationStrategy.AGGRESSIVE
            elif AMPLIFIER_MODE == "structured":
                strategy = OptimizationStrategy.STRUCTURED
            elif AMPLIFIER_MODE == "context_rich":
                strategy = OptimizationStrategy.CONTEXT_RICH
            else:  # standard mode
                strategy = OptimizationStrategy.STANDARD

            # Perform optimization
            optimizer = PromptOptimizer()
            result = optimizer.optimize(
                normalized, strategy=strategy, context=opt_context
            )

            optimized = result.optimized_prompt

            # Update metadata with optimization info
            meta.update(
                {
                    "intelligent_optimization": "true",
                    "optimization_strategy": result.strategy_used.value,
                    "prompt_type": result.analysis.prompt_type.value,
                    "complexity_score": f"{result.analysis.complexity_score:.2f}",
                    "clarity_score": f"{result.analysis.clarity_score:.2f}",
                    "enhancements_applied": ",".join(result.enhancements_applied),
                    "len_out": str(len(optimized)),
                }
            )

            return optimized, meta

        except Exception as e:
            # Fallback to normalized version if optimization fails
            meta.update(
                {
                    "intelligent_optimization": "failed",
                    "optimization_error": str(e)[:100],  # Truncate error message
                    "len_out": str(len(normalized)),
                }
            )
            return normalized, meta

    # Fallback to basic amplification
    meta["len_out"] = str(len(normalized))
    meta["intelligent_optimization"] = (
        "disabled" if not OPTIMIZER_AVAILABLE else "skipped"
    )

    return normalized, meta


def analyze_message(message: str) -> dict[str, Any]:
    """
    Analyze a message and return insights without modification.

    Args:
        message: The message to analyze

    Returns:
        Dictionary with analysis results
    """
    if not OPTIMIZER_AVAILABLE:
        return {"error": "Prompt optimizer not available"}

    try:
        analysis = analyze_user_prompt(message)
        return {
            "prompt_type": analysis.prompt_type.value,
            "complexity_score": analysis.complexity_score,
            "clarity_score": analysis.clarity_score,
            "completeness_score": analysis.completeness_score,
            "detected_entities": analysis.detected_entities,
            "suggested_enhancements": analysis.suggested_enhancements,
            "confidence": analysis.confidence,
        }
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


def get_amplifier_status() -> dict[str, Any]:
    """Get current amplifier configuration and status."""
    return {
        "amplifier_mode": AMPLIFIER_MODE,
        "intelligent_optimization_enabled": ENABLE_INTELLIGENT_OPTIMIZATION,
        "optimizer_available": OPTIMIZER_AVAILABLE,
        "max_optimization_length": MAX_OPTIMIZATION_LENGTH,
        "bypass_indicators": ["noopt:", "raw:", "literal:"],
    }


# Register on import so router can just import the module when enabled.
register(amplify_message)  # type: ignore[arg-type]
