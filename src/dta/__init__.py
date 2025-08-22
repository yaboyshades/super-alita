#!/usr/bin/env python3
"""
DTA 2.0 - Cognitive Airlock Package

Enhanced cognitive processing capabilities with DTA 2.0 architecture.
Provides structured cognitive turns, advanced planning, and neural integration.

This package provides a complete implementation of the Deep Thinking Architecture
for LLM-based intelligent preprocessing with cognitive enhancement features:

- Cognitive Turn Processing: Structured reasoning cycles with validation
- Advanced Planning: REUG methodology with multi-level cognitive planning
- Neural Integration: Seamless integration with Super Alita's Neural Atoms
- Circuit Breaker Reliability: Fault tolerance for production environments
- Comprehensive Monitoring: Observability and performance metrics

Example Usage:
    from src.dta import CognitiveTurnRecord, generate_master_cognitive_plan, DTAConfig

    # Generate cognitive plan
    plan = generate_master_cognitive_plan("Analyze customer data trends")

    # Process cognitive turn
    turn = CognitiveTurnRecord(
        state_readout="Processing user request",
        activation_protocol=ActivationProtocol(
            pattern_recognition="analytical",
            confidence_score=8
        ),
        synthesis=Synthesis(
            key_findings=["Data analysis required"],
            final_answer_summary="Comprehensive analysis plan created"
        ),
        state_update=StateUpdate(directive="ignore"),
        confidence_calibration=ConfidenceCalibration(
            final_confidence=9,
            uncertainty_gaps="None"
        )
    )
"""

# Version information
__version__ = "2.0.0"
__author__ = "DTA Development Team"
__license__ = "MIT"

# Import enhanced cognitive components
from .cognitive_plan import (
    CognitivePlan,
    create_tool_plan,
    generate_master_cognitive_plan,
)
from .config import (
    CircuitBreakerConfig,
    CognitiveTurnConfig,
    DTAConfig,
    LLMConfig,
    create_default_config,
    load_config_from_yaml,
    process_config_dict,
)

# Legacy components for backward compatibility
from .types import (
    ActivationProtocol,
    CognitiveTurnRecord,
    ConfidenceCalibration,
    DTAContext,
    DTAProcessingMetrics,
    DTARequest,
    DTAResult,
    DTAStatus,
    DTAValidationResult,
    StateUpdate,
    StrategicPlan,
    Synthesis,
)

# Import existing runtime components (if available)
try:
    # Commented out unused imports - uncomment when needed
    # from .cache import DTACache, create_cache
    # from .monitoring import DTAMonitoring, create_monitoring
    # from .reliability import AsyncCircuitBreaker, CircuitState
    # from .runtime import AsyncDTARuntime
    # from .validators import ValidationPipeline, create_validation_pipeline

    FULL_RUNTIME_AVAILABLE = True
except ImportError:
    # Graceful degradation if runtime components not available
    FULL_RUNTIME_AVAILABLE = False

__all__ = [
    # Core Configuration
    "DTAConfig",
    "CircuitBreakerConfig",
    "LLMConfig",
    "CognitiveTurnConfig",
    # Cognitive Turn Processing
    "CognitiveTurnRecord",
    "ActivationProtocol",
    "Synthesis",
    "StateUpdate",
    "ConfidenceCalibration",
    "StrategicPlan",
    # Planning and Execution
    "CognitivePlan",
    "generate_master_cognitive_plan",
    "create_tool_plan",
    # Core Results and Status
    "DTAResult",
    "DTAStatus",
    # Legacy Support
    "DTAContext",
    "DTARequest",
    "DTAProcessingMetrics",
    "DTAValidationResult",
    "create_default_config",
    "load_config_from_yaml",
    "process_config_dict",
    # Package info
    "__version__",
    "__author__",
    "__license__",
    "FULL_RUNTIME_AVAILABLE",
]

# Add runtime components if available
if FULL_RUNTIME_AVAILABLE:
    __all__.extend(
        [
            # Commented out until imports are re-enabled
            # "AsyncCircuitBreaker",
            # "AsyncDTARuntime",
            # "CircuitState",
            # "DTACache",
            # "DTAMonitoring",
            # "ValidationPipeline",
            # "create_cache",
            # "create_monitoring",
            # "create_validation_pipeline",
        ]
    )

# Package-level configuration
import logging

# Set up package logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Prevent "No handler" warnings

# Package initialization message
logger.debug(f"DTA 2.0 Cognitive Airlock package initialized (version {__version__})")


def get_version() -> str:
    """Get the DTA package version."""
    return __version__


def get_package_info() -> dict:
    """Get comprehensive package information."""
    return {
        "name": "dta",
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": "Deep Thinking Architecture with Cognitive Airlock capabilities",
        "cognitive_features": [
            "Structured cognitive turns",
            "Advanced planning with REUG methodology",
            "Neural integration with Super Alita",
            "Circuit breaker reliability patterns",
            "Comprehensive cognitive monitoring",
        ],
        "full_runtime": FULL_RUNTIME_AVAILABLE,
    }


def create_cognitive_turn_template(user_message: str) -> dict:
    """
    Create a template for cognitive turn processing.

    Args:
        user_message: The user's input message

    Returns:
        Dictionary template for CognitiveTurnRecord
    """
    return {
        "state_readout": f"Processing user request: {user_message}",
        "activation_protocol": {
            "pattern_recognition": "analytical",
            "confidence_score": 8,
            "planning_requirement": True,
            "quality_speed_tradeoff": "balance",
            "evidence_threshold": "medium",
            "audience_level": "professional",
            "meta_cycle_check": "analysis",
        },
        "strategic_plan": {"is_required": True},
        "execution_log": ["Initial cognitive processing initiated"],
        "synthesis": {
            "key_findings": ["User input analyzed"],
            "counterarguments": [],
            "final_answer_summary": "Cognitive processing in progress",
        },
        "state_update": {"directive": "ignore"},
        "confidence_calibration": {
            "final_confidence": 8,
            "uncertainty_gaps": "None identified",
        },
    }


# Export template function for convenience
__all__.append("create_cognitive_turn_template")
