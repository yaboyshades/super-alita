# Version: 3.0.0
# Description: Centralized Pydantic schemas for type safety and validation.

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Types of cognitive tasks in the 8-stage processing cycle."""

    PERCEPTION = "perception"
    MEMORY = "memory"
    PREDICTION = "prediction"
    PLANNING = "planning"
    SELECTION = "selection"
    EXECUTION = "execution"
    LEARNING = "learning"
    IMPROVEMENT = "improvement"


class AttentionLevel(str, Enum):
    """Attention priority levels for the Global Workspace."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskRequest(BaseModel):
    """Schema for requests entering the cognitive processing cycle."""

    task_id: str = Field(..., description="Unique task identifier")
    task_type: TaskType = Field(..., description="Type of cognitive task")
    description: str = Field(..., description="Natural language task description")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )
    priority: int = Field(default=5, ge=1, le=10, description="Task priority (1-10)")
    timeout_seconds: float | None = Field(
        default=30.0, description="Maximum execution time"
    )
    requester: str = Field(default="unknown", description="Entity requesting the task")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Task metadata")

    class Config:
        use_enum_values = True


class TaskResult(BaseModel):
    """Schema for task completion results."""

    task_id: str = Field(..., description="Original task identifier")
    success: bool = Field(..., description="Whether the task completed successfully")
    result: Any = Field(default=None, description="Task execution result")
    error: str | None = Field(default=None, description="Error message if failed")
    execution_time: float = Field(..., description="Time taken to execute (seconds)")
    neural_atoms_used: list[str] = Field(
        default_factory=list, description="Neural Atoms involved"
    )
    performance_metrics: dict[str, float] = Field(
        default_factory=dict, description="Performance data"
    )
    stage_completed: TaskType = Field(..., description="Cognitive stage that completed")
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Result confidence"
    )

    class Config:
        use_enum_values = True


class CapabilityGapEvent(BaseModel):
    """Schema for capability gap detection events."""

    gap_id: str = Field(..., description="Unique gap identifier")
    description: str = Field(..., description="Description of the missing capability")
    priority: int = Field(default=5, ge=1, le=10, description="Gap priority")
    detected_by: str = Field(..., description="Component that detected the gap")
    context: dict[str, Any] = Field(default_factory=dict, description="Gap context")
    suggested_solution: str | None = Field(
        default=None, description="Suggested solution approach"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="When gap was detected"
    )


class NeuralAtomSpec(BaseModel):
    """Schema for Neural Atom specifications."""

    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="What this atom does")
    capabilities: list[str] = Field(..., description="List of capabilities")
    version: str = Field(default="1.0.0", description="Version string")
    tags: list[str] = Field(default_factory=list, description="Classification tags")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Configuration parameters"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="Required dependencies"
    )


class WorkspaceEvent(BaseModel):
    """Schema for Global Workspace events."""

    timestamp: float = Field(..., description="Event timestamp")
    data: Any = Field(..., description="Event payload")
    source: str = Field(..., description="Source component")
    attention_level: AttentionLevel = Field(
        default=AttentionLevel.MEDIUM, description="Attention priority"
    )
    broadcast: bool = Field(default=True, description="Whether to broadcast this event")
    subscribers_notified: list[str] = Field(
        default_factory=list, description="Notified subscribers"
    )

    class Config:
        use_enum_values = True


class CREATORStage(str, Enum):
    """Stages in the CREATOR framework for autonomous capability generation."""

    ABSTRACT_SPECIFICATION = "abstract_specification"
    DESIGN_DECISION = "design_decision"
    IMPLEMENTATION = "implementation"
    RECTIFICATION = "rectification"


class CREATORRequest(BaseModel):
    """Schema for CREATOR framework requests."""

    request_id: str = Field(..., description="Unique request identifier")
    capability_description: str = Field(
        ..., description="Description of needed capability"
    )
    context: dict[str, Any] = Field(default_factory=dict, description="Request context")
    priority: int = Field(default=5, ge=1, le=10, description="Request priority")
    requester: str = Field(..., description="Entity requesting the capability")
    constraints: list[str] = Field(
        default_factory=list, description="Implementation constraints"
    )
    examples: list[str] = Field(default_factory=list, description="Usage examples")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Request timestamp"
    )


class CREATORResult(BaseModel):
    """Schema for CREATOR framework results."""

    request_id: str = Field(..., description="Original request identifier")
    success: bool = Field(..., description="Whether creation succeeded")
    neural_atom_id: str | None = Field(
        default=None, description="Created Neural Atom ID"
    )
    stages_completed: list[CREATORStage] = Field(
        default_factory=list, description="Completed stages"
    )
    error: str | None = Field(default=None, description="Error message if failed")
    validation_results: dict[str, Any] = Field(
        default_factory=dict, description="Validation outcomes"
    )
    performance_metrics: dict[str, float] = Field(
        default_factory=dict, description="Creation metrics"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )

    class Config:
        use_enum_values = True


class MemoryQuery(BaseModel):
    """Schema for memory retrieval queries."""

    query_text: str = Field(..., description="Natural language query")
    query_type: str = Field(default="semantic", description="Type of memory query")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Similarity threshold"
    )
    filters: dict[str, Any] = Field(
        default_factory=dict, description="Additional filters"
    )
    include_metadata: bool = Field(
        default=True, description="Include metadata in results"
    )


class MemoryResult(BaseModel):
    """Schema for memory retrieval results."""

    memory_id: str = Field(..., description="Memory identifier")
    content: Any = Field(..., description="Memory content")
    similarity_score: float = Field(..., description="Similarity to query")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Memory metadata"
    )
    hierarchy_path: list[str] = Field(
        default_factory=list, description="Memory hierarchy"
    )
    timestamp: datetime = Field(..., description="When memory was stored")


class PredictionRequest(BaseModel):
    """Schema for world model prediction requests."""

    context: dict[str, Any] = Field(..., description="Current context state")
    action: str = Field(..., description="Proposed action")
    horizon: int = Field(default=3, ge=1, le=10, description="Prediction horizon steps")
    confidence_threshold: float = Field(
        default=0.6, description="Minimum confidence required"
    )


class PredictionResult(BaseModel):
    """Schema for world model prediction results."""

    predicted_outcome: dict[str, Any] = Field(
        ..., description="Predicted state outcome"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    reasoning: str = Field(..., description="Explanation of prediction")
    alternative_actions: list[dict[str, Any]] = Field(
        default_factory=list, description="Alternative suggestions"
    )
    risk_assessment: dict[str, float] = Field(
        default_factory=dict, description="Risk factors"
    )


class SafetyValidation(BaseModel):
    """Schema for safety validation results."""

    validation_id: str = Field(..., description="Validation identifier")
    item_type: str = Field(..., description="Type of item being validated")
    safety_score: float = Field(..., ge=0.0, le=1.0, description="Overall safety score")
    checks_passed: list[str] = Field(
        default_factory=list, description="Passed safety checks"
    )
    checks_failed: list[str] = Field(
        default_factory=list, description="Failed safety checks"
    )
    risk_factors: dict[str, float] = Field(
        default_factory=dict, description="Identified risks"
    )
    mitigation_suggestions: list[str] = Field(
        default_factory=list, description="Risk mitigation suggestions"
    )
    approved: bool = Field(..., description="Whether item is approved for use")


class LearningEvent(BaseModel):
    """Schema for learning and adaptation events."""

    event_id: str = Field(..., description="Learning event identifier")
    event_type: str = Field(..., description="Type of learning event")
    subject: str = Field(..., description="What was learned about")
    outcome: dict[str, Any] = Field(..., description="Learning outcome")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Learning confidence")
    performance_impact: float = Field(
        default=0.0, description="Expected performance impact"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="When learning occurred"
    )


class SystemState(BaseModel):
    """Schema for overall system state representation."""

    state_id: str = Field(..., description="State identifier")
    cognitive_load: float = Field(
        ..., ge=0.0, le=1.0, description="Current cognitive load"
    )
    active_tasks: list[str] = Field(
        default_factory=list, description="Currently active tasks"
    )
    memory_usage: dict[str, float] = Field(
        default_factory=dict, description="Memory usage statistics"
    )
    neural_atoms_active: int = Field(..., description="Number of active Neural Atoms")
    attention_focus: list[str] = Field(
        default_factory=list, description="Current attention focus"
    )
    performance_metrics: dict[str, float] = Field(
        default_factory=dict, description="System performance"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="State timestamp"
    )


# Unified event schemas for backwards compatibility
class ConversationEvent(BaseModel):
    """Schema for conversation events."""

    session_id: str = Field(..., description="Conversation session ID")
    user_message: str = Field(..., description="User's message")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Conversation context"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Message timestamp"
    )


class ToolCallEvent(BaseModel):
    """Schema for tool execution events."""

    tool_name: str = Field(..., description="Name of tool to execute")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Tool parameters"
    )
    session_id: str = Field(..., description="Session identifier")
    request_id: str = Field(..., description="Request identifier")


class ToolResultEvent(BaseModel):
    """Schema for tool execution results."""

    tool_name: str = Field(..., description="Name of executed tool")
    success: bool = Field(..., description="Whether execution succeeded")
    result: Any = Field(default=None, description="Tool execution result")
    error: str | None = Field(default=None, description="Error message if failed")
    session_id: str = Field(..., description="Session identifier")
    request_id: str = Field(..., description="Request identifier")
    execution_time: float = Field(..., description="Execution time in seconds")


class MemoryRequest(BaseModel):
    """Schema for memory operation requests."""

    operation: str = Field(
        ..., description="Memory operation type (save, recall, list)"
    )
    content: Any | None = Field(default=None, description="Content to save")
    query: str | None = Field(default=None, description="Search query for recall")
    session_id: str = Field(..., description="Session identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ToolExecutionRequest(BaseModel):
    """Schema for tool execution requests."""

    tool_name: str = Field(..., description="Name of tool to execute")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Tool parameters"
    )
    session_id: str = Field(..., description="Session identifier")
    request_id: str = Field(..., description="Request identifier")
    timeout: float | None = Field(default=30.0, description="Execution timeout")


class ToolExecutionResult(BaseModel):
    """Schema for tool execution results."""

    request_id: str = Field(..., description="Original request identifier")
    tool_name: str = Field(..., description="Name of executed tool")
    success: bool = Field(..., description="Whether execution succeeded")
    result: Any = Field(default=None, description="Tool execution result")
    error: str | None = Field(default=None, description="Error message if failed")
    execution_time: float = Field(..., description="Execution time in seconds")
    session_id: str = Field(..., description="Session identifier")


class MemoryType(str, Enum):
    """Types of memory storage."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    LONG_TERM = "long_term"


class SemanticQuery(BaseModel):
    """Schema for semantic search queries."""

    query_text: str = Field(..., description="Query text")
    limit: int = Field(default=10, description="Maximum results")
    threshold: float = Field(default=0.7, description="Similarity threshold")
    memory_type: MemoryType | None = Field(
        default=None, description="Memory type filter"
    )


class ExecutionStatus(str, Enum):
    """Status of tool execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
