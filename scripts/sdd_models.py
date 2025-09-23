# scripts/sdd_models.py
"""
Pydantic models for Spec Kit SDD (Spec-Driven Development) artifacts.
Provides type-safe data structures for features, scenarios, plans, and tasks.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class UserScenario(BaseModel):
    """Represents a user scenario or requirement."""

    description: str = Field(
        ..., description="Natural language description of the scenario"
    )
    priority: str = Field("medium", description="Priority level: low, medium, high")
    acceptance_criteria: list[str] = Field(
        default_factory=list, description="List of acceptance criteria"
    )
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")


class FeatureSpec(BaseModel):
    """Core specification for a feature."""

    name: str = Field(..., description="Feature name")
    description: str = Field(..., description="Detailed description")
    scenarios: list[UserScenario] = Field(
        default_factory=list, description="User scenarios"
    )
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )


class Task(BaseModel):
    """Represents a single task in the SDD pipeline."""

    id: str = Field(..., description="Unique task identifier")
    description: str = Field(..., description="Task description")
    status: str = Field(
        "pending", description="Task status: pending, in_progress, completed, blocked"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="IDs of dependent tasks"
    )
    assignee: str | None = Field(None, description="Assigned person or role")
    estimated_effort: str | None = Field(
        None, description="Estimated effort (e.g., '2h', '1d')"
    )
    priority: str = Field("medium", description="Priority: low, medium, high, critical")


class Plan(BaseModel):
    """Plan containing tasks for implementing a feature."""

    feature_id: str = Field(..., description="Associated feature identifier")
    tasks: list[Task] = Field(default_factory=list, description="List of tasks")
    rationale: str = Field(..., description="Explanation of the planning approach")
    assumptions: list[str] = Field(
        default_factory=list, description="Key assumptions made"
    )
    risks: list[str] = Field(default_factory=list, description="Identified risks")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )


class ConstitutionCheck(BaseModel):
    """Result of a constitutional compliance check."""

    rule_name: str = Field(..., description="Name of the constitution rule")
    passed: bool = Field(..., description="Whether the check passed")
    severity: str = Field("medium", description="Severity: low, medium, high, critical")
    message: str = Field(..., description="Human-readable message")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional details"
    )


class ConstitutionResult(BaseModel):
    """Overall result of constitutional validation."""

    feature_id: str = Field(..., description="Feature being validated")
    checks: list[ConstitutionCheck] = Field(
        default_factory=list, description="Individual check results"
    )
    overall_score: float = Field(..., description="Overall compliance score (0.0-1.0)")
    passed: bool = Field(..., description="Whether overall validation passed")
    recommendations: list[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Validation timestamp"
    )


class SDDArtifact(BaseModel):
    """Container for all SDD artifacts related to a feature."""

    feature_spec: FeatureSpec
    plan: Plan | None = None
    constitution_result: ConstitutionResult | None = None
    implementation_status: str = Field(
        "spec", description="Current phase: spec, plan, tasks, implement"
    )


class AIRequest(BaseModel):
    """Request payload for AI generation services."""

    prompt: str = Field(..., description="The prompt to send to the AI")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )
    model: str = Field("gpt-4", description="AI model to use")
    temperature: float = Field(0.7, description="Creativity parameter")


class AIResponse(BaseModel):
    """Response from AI generation services."""

    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model used")
    usage: dict[str, Any] = Field(
        default_factory=dict, description="Token usage information"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )


# API Request/Response Models


class ConstitutionRequest(BaseModel):
    """Request to create or validate a constitution."""

    principles: str = Field(..., description="Constitutional principles as text")
    force: bool = Field(False, description="Force recreation if exists")


class ConstitutionResponse(BaseModel):
    """Response from constitution operations."""

    path: str = Field(..., description="Path to the constitution file")
    created: bool = Field(..., description="Whether a new constitution was created")
    message: str = Field(..., description="Status message")


class SpecificationRequest(BaseModel):
    """Request to create a feature specification."""

    feature_name: str = Field(..., description="Name of the feature")
    requirements: str = Field(..., description="Requirements as text")
    context: str = Field("", description="Additional context as JSON string")


class SpecificationResponse(BaseModel):
    """Response from specification creation."""

    path: str = Field(..., description="Path to the specification file")
    feature_id: str = Field(..., description="Generated feature ID")
    message: str = Field(..., description="Status message")


class PlanRequest(BaseModel):
    """Request to create an implementation plan."""

    feature_id: str = Field(..., description="Feature identifier")


class PlanResponse(BaseModel):
    """Response from plan creation."""

    path: str = Field(..., description="Path to the plan file")
    task_count: int = Field(..., description="Number of tasks created")
    message: str = Field(..., description="Status message")


class TasksRequest(BaseModel):
    """Request to create executable tasks."""

    feature_id: str = Field(..., description="Feature identifier")


class TasksResponse(BaseModel):
    """Response from tasks creation."""

    path: str = Field(..., description="Path to the tasks file")
    task_count: int = Field(..., description="Number of tasks created")
    message: str = Field(..., description="Status message")


class TaskUpdateRequest(BaseModel):
    """Request to update a task status."""

    feature_id: str = Field(..., description="Feature identifier")
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(
        ..., description="New status: pending, in_progress, completed, blocked"
    )


class TaskUpdateResponse(BaseModel):
    """Response from task update."""

    success: bool = Field(..., description="Whether the update succeeded")
    message: str = Field(..., description="Status message")


class ValidationRequest(BaseModel):
    """Request to validate a feature."""

    feature_id: str = Field(..., description="Feature identifier")
    context: str = Field("", description="Validation context")


class ValidationResponse(BaseModel):
    """Response from validation."""

    passed: bool = Field(..., description="Whether validation passed")
    score: float = Field(..., description="Compliance score")
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations"
    )
    message: str = Field(..., description="Status message")


class FeatureListResponse(BaseModel):
    """Response listing available features."""

    features: list[dict[str, Any]] = Field(
        default_factory=list, description="List of features"
    )
    count: int = Field(..., description="Number of features")


# Knowledge Base Models for Neural Indexing


class CodeEntityType(Enum):
    """Types of code entities that can be indexed."""

    FILE = "file"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"


class CodeEntity(BaseModel):
    """Base model for all code entities."""

    id: str = Field(..., description="Unique identifier for the entity")
    name: str = Field(..., description="Name of the entity")
    file_path: str = Field(..., description="Path to the file containing this entity")
    start_line: int = Field(..., description="Starting line number")
    end_line: int = Field(..., description="Ending line number")
    node_type: CodeEntityType = Field(..., description="Type of code entity")
    docstring: str | None = Field(None, description="Docstring if available")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class FunctionEntity(CodeEntity):
    """Represents a function or method."""

    is_async: bool = Field(False, description="Whether this is an async function")
    signature: str = Field(..., description="Function signature")
    decorators: list[str] = Field(
        default_factory=list, description="List of decorators"
    )
    parameters: list[str] = Field(
        default_factory=list, description="Function parameters"
    )
    calls: list[str] = Field(
        default_factory=list, description="Functions/methods called by this function"
    )


class ClassEntity(CodeEntity):
    """Represents a class."""

    inheritance: list[str] = Field(
        default_factory=list, description="Classes this class inherits from"
    )
    methods: list[str] = Field(
        default_factory=list, description="Method IDs belonging to this class"
    )
    attributes: list[str] = Field(default_factory=list, description="Attribute names")


class ModuleEntity(CodeEntity):
    """Represents a module/file."""

    imports: list[str] = Field(default_factory=list, description="Import statements")
    exports: list[str] = Field(default_factory=list, description="Exported symbols")
    classes: list[str] = Field(
        default_factory=list, description="Class IDs in this module"
    )
    functions: list[str] = Field(
        default_factory=list, description="Function IDs in this module"
    )


class NeuralAtom(BaseModel):
    """Represents a neural atom - a unit of AI reasoning."""

    id: str = Field(..., description="Unique identifier")
    content: str = Field(..., description="The actual content/thought")
    confidence: float = Field(..., description="Confidence score 0.0-1.0")
    source: str = Field(
        ..., description="Source of this atom (e.g., 'spec_analysis', 'code_context')"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    connections: list[str] = Field(
        default_factory=list, description="IDs of related atoms"
    )


class EvolutionaryPlan(BaseModel):
    """Represents an evolved implementation plan."""

    title: str = Field(..., description="Plan title")
    description: str = Field(..., description="Detailed description")
    components: list[dict[str, Any]] = Field(
        default_factory=list, description="Plan components"
    )
    tasks: list[dict[str, Any]] = Field(
        default_factory=list, description="Implementation tasks"
    )
    confidence_score: float = Field(..., description="AI confidence in this plan")
    evolution_iterations: int = Field(..., description="Number of evolution cycles")
    risk_assessment: dict[str, Any] = Field(
        default_factory=dict, description="Risk analysis"
    )


class WorldModelState(BaseModel):
    """Represents the current state of the AI's world model."""

    reasoning_chain: list[dict[str, Any]] = Field(
        default_factory=list, description="Chain of reasoning steps"
    )
    context_atoms: list[str] = Field(
        default_factory=list, description="Active neural atom IDs"
    )
    current_focus: str | None = Field(None, description="Current focus area")
    confidence_threshold: float = Field(
        0.7, description="Minimum confidence for decisions"
    )
