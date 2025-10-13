"""Data models for the Intelligence Consolidation Engine domain layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from pydantic import BaseModel, Field, RootModel, validator


class ConsolidationEnvelope(BaseModel):
    """Normalized input for the consolidation pipeline."""

    session_id: str = Field(..., min_length=1)
    turn_id: str = Field(..., min_length=1)
    timestamp: datetime
    agent_snapshot: Mapping[str, Any] = Field(default_factory=dict)
    reasoning_trace: list[Mapping[str, Any]] = Field(default_factory=list)
    tool_outputs: list[Mapping[str, Any]] = Field(default_factory=list)
    metadata: Mapping[str, Any] = Field(default_factory=dict)
    deduplication_key: str = Field(..., min_length=1)

    @validator("timestamp", pre=True)
    def _coerce_timestamp(cls, value: Any) -> datetime:  # noqa: D401
        """Ensure timestamp inputs are timezone-aware datetimes."""

        if isinstance(value, datetime):
            return value
        raise TypeError("timestamp must be a datetime instance")


class ConsolidationRequestContext(BaseModel):
    """Execution context supplied by the REUG orchestrator."""

    trace_id: str = Field(..., min_length=1)
    orchestrator_version: str = Field(..., min_length=1)
    feature_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    feature_flag_state: bool = False


class ConsolidationPatch(BaseModel):
    """Representation of the mutations applied to ACE state."""

    operations: list[Mapping[str, Any]] = Field(default_factory=list)
    checksum: str | None = None


class ACEUpdateReceipt(BaseModel):
    """Receipt returned by the ACE store adapter."""

    applied: bool
    dedupe_hit: bool = False
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class ConsolidationEventPayload(BaseModel):
    """Versioned payload emitted onto the event bus."""

    session_id: str
    turn_id: str
    status: str
    latency_ms: float
    patterns: list[Mapping[str, Any]] = Field(default_factory=list)
    validation: Mapping[str, Any] = Field(default_factory=dict)
    ace_patch: Mapping[str, Any] = Field(default_factory=dict)
    skip_reason: str | None = None
    trace_id: str
    schema_version: str = Field(default="v1", const=True)


class ConsolidationEvent(BaseModel):
    """Envelope describing the event topic and payload."""

    topic: str = Field(default="reug.consolidation.v1")
    event_type: str = Field(default="ConsolidationEvent")
    payload: ConsolidationEventPayload


class ConsolidationResult(BaseModel):
    """Return value for consolidation invocations."""

    status: str
    latency_ms: float | None = None
    patterns: list[Mapping[str, Any]] = Field(default_factory=list)
    validation: Mapping[str, Any] = Field(default_factory=dict)
    ace_receipt: ACEUpdateReceipt | None = None
    skip_reason: str | None = None


class ConsolidationResultList(RootModel[list[ConsolidationResult]]):
    """Helper container for batch operations and property testing."""

    root: list[ConsolidationResult]
