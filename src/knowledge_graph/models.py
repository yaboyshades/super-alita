from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Set
from uuid import uuid4


class EntityType(Enum):
    GOAL = "goal"
    OUTCOME = "outcome"
    PATTERN = "pattern"
    CONTEXT = "context"
    ERROR = "error"
    TASK = "task"


class RelationType(Enum):
    SUCCEEDED_BY = "succeeded_by"
    FAILED_WITH = "failed_with"
    RELATED_TO = "related_to"
    FOLLOWS = "follows"


@dataclass
class KnowledgeEntity:
    entity_type: EntityType
    name: str
    description: str
    properties: Dict[str, Any]
    id: str = field(default_factory=lambda: f"entity_{uuid4().hex[:10]}")
    confidence: float = 0.8
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


@dataclass
class KnowledgeRelation:
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    strength: float = 1.0
    context: Dict[str, Any] | None = None
    id: str = field(default_factory=lambda: f"relation_{uuid4().hex[:10]}")


@dataclass
class PlanningPattern:
    pattern_name: str
    goal_template: str
    decomposition_steps: List[str]
    domain: str
    complexity_level: int
    required_tools: List[str]
    preconditions: List[str]
    id: str = field(default_factory=lambda: f"pattern_{uuid4().hex[:10]}")
    success_rate: float = 0.5
    usage_count: int = 0
    last_updated: float = field(default_factory=time.time)


@dataclass
class PlanningContext:
    session_id: str
    goal: str
    domain: str
    user_context: Dict[str, Any]
    success: bool
    execution_time: float
    task_count: int
    created_at: float = field(default_factory=time.time)


@dataclass
class KnowledgeQuery:
    goal: str
    domain: str
    context: Dict[str, Any]
    entity_types: Set[EntityType] | None = None
    relation_types: Set[RelationType] | None = None
    include_patterns: bool = False
    max_results: int = 10
    min_confidence: float = 0.5


@dataclass
class KnowledgeQueryResult:
    entities: List[KnowledgeEntity] = field(default_factory=list)
    relations: List[KnowledgeRelation] = field(default_factory=list)
    patterns: List[PlanningPattern] = field(default_factory=list)
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    total_results: int = 0
    query_time: float = 0.0
