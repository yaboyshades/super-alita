"""
Event Schemas for Unified Orchestrator Monitoring
Defines structured event schemas for observability and analytics.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class StageType(Enum):
    """Types of orchestrator stages."""

    SDD_WORKFLOW = "sdd_workflow"
    GENERATION = "generation"
    VALIDATION = "validation"
    OTHER = "other"


class ConstitutionalArticle(Enum):
    """Constitutional framework articles."""

    LIBRARY_FIRST = "library_first"
    TEST_FIRST = "test_first"
    SIMPLICITY_GATE = "simplicity_gate"
    INTEGRATION_FIRST = "integration_first"
    CLARITY_UNAMBIGUITY = "clarity_unambiguity"
    COUNTERFACTUAL_JUSTIFICATION = "counterfactual_justification"


@dataclass
class BaseEvent:
    """Base event schema."""

    event_id: str
    timestamp: float
    run_id: str
    session_id: str
    event_type: str


@dataclass
class RunStartedEvent(BaseEvent):
    """Schema for run started events."""

    prompt: str
    sdd_mode: bool
    enabled_stages: dict[str, bool]
    configuration: dict[str, Any]


@dataclass
class RunCompletedEvent(BaseEvent):
    """Schema for run completed events."""

    success: bool
    total_duration_ms: int
    stages_executed: int
    stages_successful: int
    stage_results: dict[str, dict[str, Any]]


@dataclass
class StageEvent(BaseEvent):
    """Base schema for stage events."""

    stage_name: str
    stage_type: StageType


@dataclass
class StageStartedEvent(StageEvent):
    """Schema for stage started events."""

    pass


@dataclass
class StageCompletedEvent(StageEvent):
    """Schema for stage completed events."""

    success: bool
    duration_ms: int
    output_summary: dict[str, Any]
    error_message: str | None = None


@dataclass
class SDDValidationEvent(BaseEvent):
    """Schema for SDD validation events."""

    sdd_phase: str
    validation_type: str
    compliance_score: float
    threshold: float
    passed: bool
    article_scores: dict[str, float]


@dataclass
class ConstitutionalGateEvent(BaseEvent):
    """Schema for constitutional gate events."""

    gate_name: str
    article: ConstitutionalArticle
    threshold: float
    actual_score: float
    passed: bool


@dataclass
class ErrorEvent(BaseEvent):
    """Schema for error events."""

    error_type: str
    error_message: str
    stage_name: str | None
    stack_trace: str | None


@dataclass
class MetricEvent(BaseEvent):
    """Schema for metric events."""

    metric_name: str
    metric_value: float | int
    metric_unit: str
    tags: dict[str, str]


class EventSchemaRegistry:
    """Registry for event schemas and validation."""

    SCHEMAS = {
        "UnifiedRunStarted": RunStartedEvent,
        "UnifiedRunCompleted": RunCompletedEvent,
        "UnifiedStageStarted": StageStartedEvent,
        "UnifiedStageSucceeded": StageCompletedEvent,
        "UnifiedStageFailed": StageCompletedEvent,
        "SDDValidationStarted": SDDValidationEvent,
        "SDDValidationCompleted": SDDValidationEvent,
        "ConstitutionalGateCheck": ConstitutionalGateEvent,
        "ConstitutionalViolation": ErrorEvent,
        "MetricEmitted": MetricEvent,
        "ErrorOccurred": ErrorEvent,
    }

    @classmethod
    def get_schema(cls, event_type: str) -> type:
        """Get schema class for event type."""
        return cls.SCHEMAS.get(event_type, BaseEvent)

    @classmethod
    def validate_event(
        cls, event_type: str, event_data: dict[str, Any]
    ) -> bool:
        """Validate event data against schema."""
        schema_class = cls.get_schema(event_type)
        try:
            # Basic validation - check required fields exist
            {f.name for f in schema_class.__dataclass_fields__.values()}
            required_fields = {
                f.name
                for f in schema_class.__dataclass_fields__.values()
                if f.default == dataclass.MISSING
                and f.default_factory == dataclass.MISSING
            }

            missing_fields = required_fields - set(event_data.keys())
            return not missing_fields
        except Exception:
            return False


# Event builder utilities
def build_run_started_event(
    run_id: str, session_id: str, prompt: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Build run started event."""
    return {
        "event_type": "UnifiedRunStarted",
        "run_id": run_id,
        "session_id": session_id,
        "prompt": prompt,
        "sdd_mode": config.get("sdd_mode", False),
        "enabled_stages": {
            k: v for k, v in config.items() if k.startswith("enable_")
        },
        "configuration": config,
    }


def build_stage_event(
    event_type: str, run_id: str, session_id: str, stage_name: str, **kwargs
) -> dict[str, Any]:
    """Build stage event."""
    return {
        "event_type": event_type,
        "run_id": run_id,
        "session_id": session_id,
        "stage_name": stage_name,
        **kwargs,
    }


def build_sdd_validation_event(
    run_id: str,
    session_id: str,
    phase: str,
    compliance_score: float,
    threshold: float,
    passed: bool,
    **kwargs,
) -> dict[str, Any]:
    """Build SDD validation event."""
    return {
        "event_type": "SDDValidationCompleted",
        "run_id": run_id,
        "session_id": session_id,
        "sdd_phase": phase,
        "compliance_score": compliance_score,
        "threshold": threshold,
        "passed": passed,
        **kwargs,
    }


def build_constitutional_gate_event(
    run_id: str,
    session_id: str,
    gate_name: str,
    article: str,
    threshold: float,
    actual_score: float,
    passed: bool,
) -> dict[str, Any]:
    """Build constitutional gate event."""
    return {
        "event_type": "ConstitutionalGateCheck",
        "run_id": run_id,
        "session_id": session_id,
        "gate_name": gate_name,
        "article": article,
        "threshold": threshold,
        "actual_score": actual_score,
        "passed": passed,
    }


def build_metric_event(
    run_id: str,
    session_id: str,
    metric_name: str,
    metric_value: float | int,
    metric_unit: str,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build metric event."""
    return {
        "event_type": "MetricEmitted",
        "run_id": run_id,
        "session_id": session_id,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_unit": metric_unit,
        "tags": tags or {},
    }
