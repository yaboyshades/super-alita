"""Unified contracts for cross-component communication.

All adapters and core services communicate via these contracts.
No direct module-to-module calls across domain boundaries.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

# ============================================================================
# Core Event Contract
# ============================================================================


class UnifiedEvent(BaseModel):
    """Universal event schema for all cross-component communication.

    Attributes:
        event_id: Unique identifier for this event instance
        event_type: Type/category of the event
        source: Component that originated the event
        target: Optional destination component (None = broadcast)
        payload: Event-specific data (JSON-serializable dict)
        ts: Unix timestamp (seconds since epoch)
        corr_id: Correlation ID for request tracing
        version: Schema version for evolution/compatibility
    """

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: Literal[
        "boot",
        "shutdown",
        "component_ready",
        "component_degraded",
        "sdd_command",
        "sdd_specify",
        "sdd_plan",
        "sdd_tasks",
        "sdd_validate",
        "code_generate",
        "code_review",
        "compliance_check",
        "compliance_violation",
        "test_run",
        "test_result",
        "memory_store",
        "memory_retrieve",
        "health_check",
        "metric_report",
        "error",
    ]
    source: Literal[
        "orchestrator",
        "codex",
        "super_alita",
        "cma",
        "sdd",
        "compliance",
        "memory",
        "testing",
        "monitoring",
    ]
    target: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: float = Field(default_factory=time.time)
    corr_id: str = Field(default_factory=lambda: str(uuid4()))
    version: str = "1.0.0"


# ============================================================================
# Domain-Specific Contracts
# ============================================================================


class Task(BaseModel):
    """Specification-Driven Development task."""

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: Literal["not_started", "in_progress", "completed", "failed"] = (
        "not_started"
    )
    assignee: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class Specification(BaseModel):
    """SDD specification document."""

    spec_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: str
    requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    alternatives_considered: list[str] = Field(default_factory=list)
    test_requirements: list[str] = Field(default_factory=list)
    complexity_budget: dict[str, Any] = Field(default_factory=dict)
    constitutional_score: float | None = None
    created_at: float = Field(default_factory=time.time)


class Plan(BaseModel):
    """SDD implementation plan."""

    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    spec_id: str
    title: str
    description: str
    milestones: list[dict[str, Any]] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    risk_assessment: dict[str, Any] = Field(default_factory=dict)
    constitutional_compliance: float | None = None
    created_at: float = Field(default_factory=time.time)


class Violation(BaseModel):
    """Constitutional or policy violation."""

    violation_id: str = Field(default_factory=lambda: str(uuid4()))
    violation_type: Literal[
        "constitutional",
        "security",
        "complexity",
        "test_coverage",
        "integration",
        "clarity",
    ]
    severity: Literal["low", "medium", "high", "critical"]
    article: str | None = None  # e.g., "Article II: Test-First"
    description: str
    artifact: str  # What was being validated
    recommendation: str | None = None
    auto_remediation: str | None = None
    detected_at: float = Field(default_factory=time.time)
    corr_id: str


class HealthStatus(BaseModel):
    """Component health status."""

    component: str
    status: Literal["healthy", "degraded", "unhealthy"]
    message: str | None = None
    latency_ms: float | None = None
    last_check: float = Field(default_factory=time.time)
    details: dict[str, Any] = Field(default_factory=dict)


class MetricReport(BaseModel):
    """Performance/observability metric."""

    metric_name: str
    metric_type: Literal["counter", "gauge", "histogram", "summary"]
    value: float
    unit: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


# ============================================================================
# Adapter Base Class
# ============================================================================


class Adapter(ABC):
    """Base class for all adapters.

    Adapters are the only components that interact with external systems.
    They communicate with the core via EventBus using UnifiedEvent.
    """

    name: str

    def __init__(self, bus: Any):
        """Initialize adapter with EventBus reference.

        Args:
            bus: EventBus instance (typed as Any to avoid circular imports)
        """
        self.bus = bus

    @abstractmethod
    async def handle(self, evt: UnifiedEvent) -> None:
        """Handle an incoming event.

        Args:
            evt: Event to process
        """
        ...

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check health of external dependencies.

        Returns:
            Current health status
        """
        ...

    async def emit(
        self,
        evt_type: str,
        payload: dict[str, Any],
        corr: str,
        target: str | None = None,
    ) -> None:
        """Emit an event to the bus.

        Args:
            evt_type: Type of event
            payload: Event data
            corr: Correlation ID
            target: Optional target component
        """
        evt = UnifiedEvent(
            event_id=str(uuid4()),
            event_type=evt_type,  # type: ignore[arg-type]
            source=self.name,  # type: ignore[arg-type]
            target=target,
            payload=payload,
            ts=time.time(),
            corr_id=corr,
        )
        await self.bus.publish(evt)


# ============================================================================
# Service Interfaces
# ============================================================================


class Memory(ABC):
    """Abstract memory/knowledge store interface."""

    @abstractmethod
    async def put(self, item: dict[str, Any], corr_id: str) -> str:
        """Store an item, return its ID."""
        ...

    @abstractmethod
    async def search(
        self, query: str, k: int = 8, corr_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Search for similar items."""
        ...

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check memory store health."""
        ...


class Compliance(ABC):
    """Abstract compliance/validation interface."""

    @abstractmethod
    async def validate(
        self, artifact: str, kind: str, corr_id: str
    ) -> dict[str, Any]:
        """Validate artifact against policies.

        Returns:
            dict with 'score' (float), 'violations' (list), 'details' (dict)
        """
        ...

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check compliance engine health."""
        ...
