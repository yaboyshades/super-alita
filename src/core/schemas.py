# Version: 3.0.0
# Description: Centralized Pydantic schemas for type safety and validation.

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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

    model_config = ConfigDict(use_enum_values=True)


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

    model_config = ConfigDict(use_enum_values=True)


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

    model_config = ConfigDict(use_enum_values=True)


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

    model_config = ConfigDict(use_enum_values=True)


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


class WorkingMemoryUpdate(BaseModel):
    """Schema for working memory updates."""

    memory_id: str = Field(..., description="Working memory identifier")
    content: Any = Field(..., description="Memory content to update")
    operation: str = Field(
        default="update", description="Operation type (update, delete, add)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Update timestamp"
    )
    priority: int = Field(default=5, ge=1, le=10, description="Update priority")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ExecutionStatus(str, Enum):
    """Status of tool execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


# GitHub Integration Event Schemas
class GitHubEventType(str, Enum):
    """Types of GitHub events for cognitive agent integration."""

    ISSUE_CREATED = "issue_created"
    ISSUE_UPDATED = "issue_updated"
    ISSUE_CLOSED = "issue_closed"
    PR_OPENED = "pr_opened"
    PR_UPDATED = "pr_updated"
    PR_MERGED = "pr_merged"
    PR_CLOSED = "pr_closed"
    COMMIT_PUSHED = "commit_pushed"
    WORKFLOW_RUN = "workflow_run"
    SECURITY_ALERT = "security_alert"
    REVIEW_SUBMITTED = "review_submitted"
    RELEASE_PUBLISHED = "release_published"


class GitHubEventSchema(BaseModel):
    """Schema for GitHub events captured by the cognitive agent."""

    event_type: GitHubEventType = Field(..., description="Type of GitHub event")
    repository: str = Field(..., description="Repository name (owner/repo)")
    actor: str = Field(..., description="GitHub username of the actor")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Event timestamp"
    )
    payload: dict[str, Any] = Field(..., description="GitHub event payload")
    event_id: str = Field(..., description="Unique event identifier")
    session_id: str | None = Field(default=None, description="Associated session ID")

    # Cognitive processing metadata
    attention_level: AttentionLevel = Field(
        default=AttentionLevel.MEDIUM, description="Cognitive attention priority"
    )
    processing_status: str = Field(
        default="pending", description="Cognitive processing status"
    )
    insights_extracted: list[str] = Field(
        default_factory=list, description="Extracted insights from event"
    )


class GitHubPriorityMetrics(BaseModel):
    """Schema for GitHub-specific priority calculation metrics."""

    has_security_alert: bool = Field(default=False, description="Has security implications")
    blocks_other_prs: bool = Field(default=False, description="Blocks other pull requests")
    has_stakeholder_mention: bool = Field(default=False, description="Mentions key stakeholders")
    ci_status: str = Field(default="unknown", description="CI/CD status")
    review_count: int = Field(default=0, ge=0, description="Number of reviews")
    comment_count: int = Field(default=0, ge=0, description="Number of comments")
    file_changes_count: int = Field(default=0, ge=0, description="Number of changed files")
    lines_changed: int = Field(default=0, ge=0, description="Lines of code changed")
    issue_labels: list[str] = Field(default_factory=list, description="GitHub issue labels")

    # Relationship metrics
    related_issues: list[str] = Field(default_factory=list, description="Related issue numbers")
    dependent_prs: list[str] = Field(default_factory=list, description="Dependent PR numbers")
    merge_conflicts: bool = Field(default=False, description="Has merge conflicts")


class GitHubApiRequest(BaseModel):
    """Schema for GitHub API requests."""

    endpoint: str = Field(..., description="GitHub API endpoint")
    method: str = Field(default="GET", description="HTTP method")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="API request parameters"
    )
    headers: dict[str, str] = Field(
        default_factory=dict, description="Additional headers"
    )
    repository: str | None = Field(default=None, description="Target repository")
    rate_limit_aware: bool = Field(
        default=True, description="Whether to respect rate limits"
    )


class GitHubApiResponse(BaseModel):
    """Schema for GitHub API responses."""

    success: bool = Field(..., description="Whether request succeeded")
    status_code: int = Field(..., description="HTTP status code")
    data: Any = Field(default=None, description="Response data")
    error: str | None = Field(default=None, description="Error message if failed")
    rate_limit_remaining: int | None = Field(
        default=None, description="Remaining API rate limit"
    )
    rate_limit_reset: datetime | None = Field(
        default=None, description="Rate limit reset time"
    )
    execution_time: float = Field(..., description="Request execution time")


class GitHubWorkflowConfig(BaseModel):
    """Schema for GitHub Actions workflow configuration."""

    workflow_name: str = Field(..., description="Workflow name")
    trigger_events: list[str] = Field(..., description="Events that trigger workflow")
    cognitive_mode: str = Field(
        default="shadow", description="Cognitive agent mode (shadow/act/batch)"
    )
    analysis_scope: list[str] = Field(
        default_factory=list, description="Scope of analysis to perform"
    )
    output_format: str = Field(
        default="comment", description="How to output results (comment/check/artifact)"
    )
    security_scanning: bool = Field(
        default=True, description="Enable security scanning"
    )
    performance_monitoring: bool = Field(
        default=True, description="Enable performance monitoring"
    )
