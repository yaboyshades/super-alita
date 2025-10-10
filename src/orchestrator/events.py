"""Unified event definitions for the orchestrator layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class UnifiedEvent(BaseModel):
    """Canonical event envelope exchanged across the unified EventBus."""

    topic: str
    payload: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str
    causation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"
        frozen = True
