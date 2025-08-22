#!/usr/bin/env python3
"""
DTA 2.0 Core Data Structures - Upgraded for Cognitive Turn

Enhanced type definitions for DTA 2.0 cognitive processing with
comprehensive data models for cognitive turns and neural processing.
"""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DTAStatus(Enum):
    """Status of DTA processing."""

    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PENDING = "pending"


class ConfidenceLevel(Enum):
    """Confidence levels for DTA responses."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# DTA 2.0 Cognitive Turn Models
class ActivationProtocol(BaseModel):
    """Cognitive activation protocol for processing turns."""

    pattern_recognition: str
    confidence_score: int = Field(..., ge=1, le=10)
    planning_requirement: bool = True
    quality_speed_tradeoff: str = "balance"
    evidence_threshold: str = "medium"
    audience_level: str = "professional"
    meta_cycle_check: str = "analysis"


class StrategicPlan(BaseModel):
    """Strategic planning component for cognitive turns."""

    is_required: bool = True
    steps: list[str] = Field(default_factory=list)
    estimated_duration: float | None = None
    resource_requirements: list[str] = Field(default_factory=list)


class Synthesis(BaseModel):
    """Synthesis component for cognitive processing."""

    key_findings: list[str]
    counterarguments: list[str] = Field(default_factory=list)
    final_answer_summary: str
    supporting_evidence: list[str] = Field(default_factory=list)
    uncertainty_areas: list[str] = Field(default_factory=list)


class StateUpdate(BaseModel):
    """State update directives for cognitive processing."""

    directive: str = "ignore"  # ignore, memory_stream_add, context_update, etc.
    memory_stream_add: dict[str, Any] | None = None
    context_updates: dict[str, Any] | None = None
    system_notifications: list[str] = Field(default_factory=list)


class ConfidenceCalibration(BaseModel):
    """Confidence calibration for cognitive processing."""

    final_confidence: int = Field(..., ge=1, le=10)
    uncertainty_gaps: str = "None"
    risk_assessment: str = "low"
    verification_methods: list[str] = Field(default_factory=list)


class CognitiveTurnRecord(BaseModel):
    """The complete Pydantic model for a single, auditable cognitive turn."""

    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    state_readout: str
    activation_protocol: ActivationProtocol
    strategic_plan: StrategicPlan | None = None
    execution_log: list[str] = Field(default_factory=list)
    synthesis: Synthesis
    state_update: StateUpdate
    confidence_calibration: ConfidenceCalibration
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    processing_metadata: dict[str, Any] = Field(default_factory=dict)


# Legacy DTA Types for Backward Compatibility
@dataclass
class DTAContext:
    """Context information for DTA processing."""

    session_id: str
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class DTARequest:
    """Request for DTA processing."""

    user_message: str
    context: DTAContext
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DTAProcessingMetrics:
    """Metrics for DTA processing."""

    processing_time_ms: float = 0.0
    thinking_steps: int = 0
    validation_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class DTAValidationResult:
    """Validation result for DTA processing."""

    is_valid: bool
    confidence_score: float
    issues: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DTAResult:
    """Result of DTA processing with cognitive turn integration."""

    status: DTAStatus
    cognitive_turn: CognitiveTurnRecord | None = None
    python_code: str | None = None
    thinking_trace: str | None = None
    reasoning_summary: str | None = None
    confidence_score: float = 0.5
    validation_result: DTAValidationResult | None = None
    processing_metrics: DTAProcessingMetrics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Compatibility properties for legacy plugin code
    @property
    def success(self) -> bool:
        """Legacy compatibility: True if status is SUCCESS."""
        return self.status == DTAStatus.SUCCESS

    @property
    def intent(self) -> str | None:
        """Legacy compatibility: Extract intent from metadata."""
        return self.metadata.get("intent")

    @intent.setter
    def intent(self, value: str):
        """Legacy compatibility: Set intent in metadata."""
        self.metadata["intent"] = value

    @property
    def function_call(self) -> str | None:
        """Legacy compatibility: Extract function call from metadata."""
        return self.metadata.get("function_call")

    @function_call.setter
    def function_call(self, value: str):
        """Legacy compatibility: Set function call in metadata."""
        self.metadata["function_call"] = value

    @property
    def code(self) -> str | None:
        """Legacy compatibility: Alias for python_code."""
        return self.python_code

    @property
    def intent_type(self) -> str | None:
        """Legacy compatibility: Alias for intent."""
        return self.intent

    @property
    def error_message(self) -> str | None:
        """Legacy compatibility: Extract error from metadata."""
        return self.metadata.get("error_message")
