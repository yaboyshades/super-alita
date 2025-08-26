from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class LadderStage(str, Enum):
    LOCALIZE = "L"
    ASSESS = "A"
    DECOMPOSE = "D1"
    DECIDE = "D2"
    EXECUTE = "E"
    REVIEW = "R"


class TodoStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    CANCELLED = "cancelled"


class Evidence(BaseModel):
    kind: Literal["log", "trace", "metric", "artifact", "note"] = "note"
    ref: str | None = None
    summary: str | None = None
    score: float | None = None  # confidence or relevance


class ExitCriteria(BaseModel):
    description: str
    validator: str | None = None  # name of a check/tool
    must_pass: bool = True


class Todo(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str | None = None

    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    depends_on: set[str] = Field(default_factory=set)  # DAG edges

    stage: LadderStage = LadderStage.LOCALIZE
    status: TodoStatus = TodoStatus.PENDING
    energy: float = 0.0
    priority: float = 0.0
    confidence: float = 0.0

    owner: str | None = None
    tool_hint: str | None = None
    exit_criteria: list[ExitCriteria] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True


class TodoEvent(BaseModel):
    kind: str  # e.g., "todo.created", "plan.decomposed"
    todo_id: str | None = None
    payload: dict = Field(default_factory=dict)
    ts: datetime = Field(default_factory=datetime.utcnow)
