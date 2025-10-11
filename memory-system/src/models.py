"""Core data models for the memory system."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

try:  # pragma: no cover - compatibility shim
    from pydantic import field_validator as validator
except ImportError:  # pragma: no cover
    from pydantic import validator  # type: ignore


class Role(str, Enum):
    """Supported message roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MemoryType(str, Enum):
    """Kinds of memories tracked by the system."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"


class Message(BaseModel):
    """Inbound message captured by the system."""

    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    role: Role
    content: str
    meta: Dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=datetime.utcnow)


class Memory(BaseModel):
    """Primary memory record."""

    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    kind: MemoryType = MemoryType.EPISODIC
    embeddings: Optional[List[float]] = None
    tags: List[str] = Field(default_factory=list)
    source: str = "conversation"
    importance: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    ttl_days: int = Field(gt=0, default=90)
    access_count: int = Field(ge=0, default=0)
    last_access: datetime = Field(default_factory=datetime.utcnow)
    ts: datetime = Field(default_factory=datetime.utcnow)
    link: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @validator("importance")
    def clamp_importance(cls, value: float) -> float:
        """Clamp importance to the [0, 1] interval."""

        return max(0.0, min(1.0, value))


class Decision(BaseModel):
    """Inspector decision surfaced alongside retrieved context."""

    claim: str
    evidence_ids: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    caveats: List[str] = Field(default_factory=list)
    actions: List[
        Literal[
            "summarize_for_context",
            "ask_for_clarification",
            "promote",
            "demote",
            "discard",
            "consolidate",
        ]
    ] = Field(default_factory=list)


class ContextPack(BaseModel):
    """Response payload for /context.

    Provenance keys include the originating query, policy, memory and decision counts,
    and, when ACE evolution is enabled, cycle identifiers and applied operations.
    """

    text: str
    citations: List[str]
    decisions: List[Decision]
    provenance: Dict[str, str]
    budget_used: int
    budget_total: int


class ConsolidationBatch(BaseModel):
    """Metadata emitted when episodic memories are consolidated."""

    episodic_ids: List[str]
    semantic_id: Optional[str] = None
    summary: str
    confidence: float
    evidence_count: int
    ts: datetime = Field(default_factory=datetime.utcnow)


class Conflict(BaseModel):
    """Conflict surfaced between two memories."""

    memory_a: str
    memory_b: str
    conflict_type: Literal["contradiction", "temporal", "confidence"]
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None


class ACEContextStrategy(BaseModel):
    """ACE-inspired context evolution strategy metadata."""

    strategy_id: str
    trigger_condition: str = Field(
        ..., description="contradiction_detected|low_confidence|new_insight"
    )
    context_transform: str = Field(
        ..., description="expand_evidence|add_counterexamples|restructure"
    )
    success_metrics: List[str] = Field(default_factory=list)
    last_applied: Optional[datetime] = None
    success_rate: float = Field(ge=0.0, le=1.0, default=0.0)


class EvolvableMemory(Memory):
    """Memory enriched with ACE evolution telemetry."""

    context_strategies: List[ACEContextStrategy] = Field(default_factory=list)
    revision_history: List[Dict[str, Any]] = Field(default_factory=list)
    contradiction_count: int = Field(ge=0, default=0)
    clarity_score: float = Field(ge=0.0, le=1.0, default=1.0)
