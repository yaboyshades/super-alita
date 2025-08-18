"""
Event schemas for the Super Alita agent communication bus.
All events are versioned, typed, and can carry semantic embeddings.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class BaseEvent(BaseModel):
    """Base event class with common fields for all events."""
    
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    version: str = "1.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_plugin: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CognitiveEvent(BaseEvent):
    """Events related to cognitive processes."""
    
    class Config:
        extra = "allow"


class SkillProposalEvent(CognitiveEvent):
    """Event for proposing new skills to the agent."""
    
    event_type: str = "skill_proposal"
    skill_name: str
    skill_description: str
    skill_code: str
    parent_skills: List[str] = Field(default_factory=list)
    proposer: str
    confidence: float = Field(ge=0.0, le=1.0)


class SkillEvaluationEvent(CognitiveEvent):
    """Event for evaluating skill performance."""
    
    event_type: str = "skill_evaluation"
    skill_name: str
    task_id: str
    performance_score: float
    execution_time: float
    success: bool
    feedback: str = ""


class MemoryEvent(BaseEvent):
    """Events related to memory operations."""
    
    operation: str  # "store", "retrieve", "update", "delete"
    memory_type: str  # "semantic", "episodic", "procedural"
    content: Any
    query: Optional[str] = None
    similarity_threshold: Optional[float] = None


class PlanningEvent(BaseEvent):
    """Events related to planning and reasoning."""
    
    goal: str
    current_state: Dict[str, Any]
    action_space: List[str]
    plan: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None


class PlanningDecisionEvent(BaseEvent):
    """Event for LADDER-AOG planning decisions."""
    
    event_type: str = "planning_decision"
    plan_id: str
    decision: str
    confidence_score: float
    causal_factors: List[str] = Field(default_factory=list)


class FSMStateEvent(BaseEvent):
    """Events for semantic FSM state transitions."""
    
    event_type: str = "fsm_state_change"
    from_state: str
    to_state: str
    trigger: str
    semantic_similarity: Optional[float] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class EvolutionEvent(BaseEvent):
    """Events for evolutionary processes."""
    
    event_type: str = "evolution"
    generation: int
    population_size: int
    best_fitness: float
    mutation_rate: float
    selection_method: str
    evolved_entities: List[Dict[str, Any]] = Field(default_factory=list)


class AtomBirthEvent(BaseEvent):
    """Event for tracking atom creation and genealogy."""
    
    event_type: str = "atom_birth"
    atom_key: str
    parent_keys: List[str] = Field(default_factory=list)
    birth_context: str
    lineage_metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolCallEvent(BaseEvent):
    """Event for tool calls."""
    
    event_type: str = "tool_call"
    conversation_id: str
    session_id: str
    tool_name: str
    parameters: Dict[str, Any]
    tool_call_id: str


class ToolResultEvent(BaseEvent):
    """Event for tool results."""
    
    event_type: str = "tool_result"
    conversation_id: str
    session_id: str
    tool_call_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SystemEvent(BaseEvent):
    """System-level events."""
    
    event_type: str = "system"
    level: str  # "info", "warning", "error", "critical"
    message: str
    component: str


class HealthCheckEvent(BaseEvent):
    """Events for system health monitoring."""
    
    event_type: str = "health_check"
    component: str
    status: str  # "healthy", "degraded", "failed"
    metrics: Dict[str, float] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)


# Event type registry for dynamic event creation
EVENT_TYPES = {
    "skill_proposal": SkillProposalEvent,
    "skill_evaluation": SkillEvaluationEvent,
    "memory": MemoryEvent,
    "planning": PlanningEvent,
    "fsm_state_change": FSMStateEvent,
    "evolution": EvolutionEvent,
    "atom_birth": AtomBirthEvent,
    "tool_call": ToolCallEvent,
    "tool_result": ToolResultEvent,
    "system": SystemEvent,
    "health_check": HealthCheckEvent,
}


def create_event(event_type: str, **kwargs) -> BaseEvent:
    """Factory function to create events by type."""
    
    event_class = EVENT_TYPES.get(event_type, BaseEvent)
    return event_class(event_type=event_type, **kwargs)


def serialize_event(event: BaseEvent) -> Dict[str, Any]:
    """Serialize event to dictionary for transmission."""
    
    return event.model_dump()


def deserialize_event(data: Dict[str, Any]) -> BaseEvent:
    """Deserialize event from dictionary."""
    
    event_type = data.get("event_type", "base")
    event_class = EVENT_TYPES.get(event_type, BaseEvent)
    return event_class(**data)
