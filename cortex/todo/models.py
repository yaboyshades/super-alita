from __future__ import annotations
from typing import List, Optional, Dict, Literal, Set
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


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
    ref: Optional[str] = None
    summary: Optional[str] = None
    score: Optional[float] = None  # confidence or relevance


class ExitCriteria(BaseModel):
    description: str
    validator: Optional[str] = None  # name of a check/tool
    must_pass: bool = True


class Todo(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: Optional[str] = None

    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    depends_on: Set[str] = Field(default_factory=set)  # DAG edges

    stage: LadderStage = LadderStage.LOCALIZE
    status: TodoStatus = TodoStatus.PENDING
    energy: float = 0.0
    priority: float = 0.0
    confidence: float = 0.0

    owner: Optional[str] = None
    tool_hint: Optional[str] = None
    exit_criteria: List[ExitCriteria] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True


class TodoEvent(BaseModel):
    kind: str  # e.g., "todo.created", "plan.decomposed"
    todo_id: Optional[str] = None
    payload: Dict = Field(default_factory=dict)
    ts: datetime = Field(default_factory=datetime.utcnow)
