"""
Unified Intelligence Layer Contracts
Pydantic models derived from contracts.yaml schema
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Request(BaseModel):
    """Base request schema for unified intelligence."""

    request_id: str
    ts: str  # ISO8601 timestamp
    intent_text: str
    code_refs: list[str] | None = Field(default_factory=list)
    context: dict[str, Any] | None = Field(default_factory=dict)


class MangleResult(BaseModel):
    """Result from Mangle bridge analysis."""

    ok: bool
    facts: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    findings: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    errors: list[str] = Field(default_factory=list)


class ConstitutionResult(BaseModel):
    """Result from constitutional compliance analysis."""

    ok: bool
    article_scores: dict[str, float] = Field(default_factory=dict)
    overall: float = Field(ge=0.0, le=1.0)
    infractions: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    errors: list[str] = Field(default_factory=list)


class WorkflowResult(BaseModel):
    """Result from workflow detection."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    features: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class CopilotEnhancement(BaseModel):
    """Result from copilot enhancement."""

    ok: bool
    templates_applied: list[str] = Field(default_factory=list)
    guidance: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class UnifiedAdvice(BaseModel):
    """Final orchestrator output."""

    ok: bool
    decision: str  # "proceed" | "revise" | "block"
    reasons: list[str] = Field(default_factory=list)
    recommendations: list[dict[str, Any]] = Field(default_factory=list)
    scores: dict[str, Any] = Field(default_factory=dict)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class FusionConfig(BaseModel):
    """Configuration for score fusion."""

    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "mangle_base": 0.35,
            "constitution_base": 0.45,
            "workflow_base": 0.20,
        }
    )
    modifiers: dict[str, float] = Field(
        default_factory=lambda: {
            "code_task_boost": 0.15,
            "constitutional_boost": 0.15,
        }
    )
    confidence_floor: float = 0.5
    decision_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"proceed": 0.7, "revise": 0.5, "block": 0.0}
    )
    constitution_gate: float = 0.4


class CodeAnalysisRequest(BaseModel):
    """Request for code analysis."""

    repo_path: str
    include_tests: bool = True
    rules_to_run: list[str] | None = None


class Finding(BaseModel):
    """Individual finding from code analysis."""

    rule_name: str
    symbol: str | None = None
    file: str | None = None
    complexity: float | None = None
    indegree: int | None = None
    file_a: str | None = None
    file_b: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodeAnalysisResponse(BaseModel):
    """Response from code analysis."""

    repo_path: str
    total_files: int
    total_symbols: int
    findings: dict[str, list[Finding]] = Field(default_factory=dict)
    summary: dict[str, int] = Field(default_factory=dict)
    analysis_time: float
    success: bool
    error_message: str | None = None


class TelemetryHeaders(BaseModel):
    """Headers for telemetry."""

    X_UI_Decision: str = Field(alias="X-UI-Decision")
    X_UI_Fused_Score: str = Field(alias="X-UI-Fused-Score")
    X_UI_Workflow: str = Field(alias="X-UI-Workflow")
    X_UI_Constitution: str = Field(alias="X-UI-Constitution")
    X_UI_Mangle_Conf: str = Field(alias="X-UI-Mangle-Conf")

    class Config:
        allow_population_by_field_name = True


# Convenience functions for creating instances
def create_request(
    intent_text: str, request_id: str | None = None, **kwargs
) -> Request:
    """Create a request with current timestamp."""
    if request_id is None:
        request_id = f"req_{int(datetime.now().timestamp() * 1000)}"
    return Request(
        request_id=request_id,
        ts=datetime.now().isoformat(),
        intent_text=intent_text,
        **kwargs,
    )


def create_empty_response(model_class, **overrides):
    """Create an empty response of the given model class."""
    defaults = {"ok": False, "confidence": 0.0, "errors": []}
    defaults.update(overrides)
    return model_class(**defaults)
