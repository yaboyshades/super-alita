"""Pydantic models for SDD API endpoints."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class SpecifyRequest(BaseModel):
    """Request model for /specify endpoint."""

    user_input: str = Field(
        ...,
        description="Natural language description of the feature to specify",
        min_length=10,
        max_length=2000,
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for specification generation",
    )
    constitutional_gates: bool = Field(
        default=True, description="Whether to apply constitutional validation gates"
    )
    spec_file: str | None = Field(
        default=None, description="Optional pre-created specification file to populate"
    )
    branch_name: str | None = Field(
        default=None, description="Git branch created for the feature"
    )
    feature_dir: str | None = Field(
        default=None, description="Directory containing feature artifacts"
    )

class ConstitutionalValidation(BaseModel):
    """Constitutional validation result."""

    article: str = Field(..., description="Constitutional article name")
    compliant: bool = Field(..., description="Whether the artifact is compliant")
    score: float = Field(..., description="Compliance score (0.0-1.0)")
    violations: list[str] = Field(
        default_factory=list, description="List of violations if not compliant"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Suggested improvements"
    )

class NextStepItem(BaseModel):
    """Structured representation of an actionable next step."""

    action: str = Field(..., description="Imperative statement for the follow-up item")
    owner: str = Field(
        default="unassigned",
        description="Owner responsible for resolving the item",
    )
    linked_artifact: str = Field(
        ..., description="Path or identifier of supporting evidence"
    )
    gate: Literal[
        "library_first",
        "test_first",
        "simplicity",
        "integration_first",
        "clarity",
        "counterfactual",
    ] = Field(
        ..., description="Constitutional gate the item helps to satisfy"
    )
    status: Literal["pending", "in_progress", "complete"] = Field(
        default="pending",
        description="Workflow status for the item",
    )
    rationale: str | None = Field(
        default=None, description="Reason the item exists"
    )
    source: Literal["clarification", "artefact", "command", "reminder"] = Field(
        ..., description="Category of the next step"
    )


class ConstitutionalAlignment(BaseModel):
    """Narrative summary of gate alignment for next steps."""

    gate: Literal[
        "library_first",
        "test_first",
        "simplicity",
        "integration_first",
        "clarity",
        "counterfactual",
    ] = Field(..., description="Constitutional gate name")
    summary: str = Field(..., description="How the listed steps satisfy the gate")
    evidence: str | None = Field(
        default=None,
        description="Pointer to supporting artefact or decision log",
    )


class NextStepGuidance(BaseModel):
    """Machine-readable bundle of next-step metadata."""

    feature_id: str = Field(..., description="Feature identifier for the guidance")
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp guidance was produced",
    )
    clarifications: list[NextStepItem] = Field(
        default_factory=list, description="Outstanding clarification items"
    )
    artefacts: list[NextStepItem] = Field(
        default_factory=list, description="Required artefact creation items"
    )
    commands: list[NextStepItem] = Field(
        default_factory=list, description="Recommended command checklist"
    )
    constitutional_alignment: list[ConstitutionalAlignment] = Field(
        default_factory=list,
        description="Gate summaries describing compliance coverage",
    )



class SpecifyResponse(BaseModel):
    """Response model for /specify endpoint."""

    success: bool = Field(..., description="Whether specification generation succeeded")
    specification: str = Field(..., description="Generated specification content")
    feature_id: str = Field(..., description="Unique identifier for the feature")
    feature_path: str = Field(..., description="Path to the generated specification")
    branch_name: str | None = Field(
        default=None, description="Git branch associated with the feature"
    )
    feature_name: str | None = Field(
        default=None, description="Human-friendly feature name or slug"
    )
    spec_file_path: str | None = Field(
        default=None, description="Alias for feature_path (kept for compatibility)"
    )
    feature_dir: str | None = Field(
        default=None, description="Directory containing specification artifacts"
    )
    analysis_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional analysis and Mangle-enhanced reasoning artifacts",
    )
    constitutional_compliance: dict[str, ConstitutionalValidation] = Field(
        default_factory=dict, description="Constitutional validation results by article"
    )
    overall_compliance_score: float = Field(
        ..., description="Overall constitutional compliance score"
    )
    compliance_threshold_met: bool = Field(
        ..., description="Whether the compliance threshold (0.75) was met"
    )
    next_steps: list[str] = Field(
        default_factory=list, description="Recommended next steps"
    )
    next_step_guidance: NextStepGuidance | None = Field(
        default=None,
        description="Structured guidance for downstream tooling",
    )
    next_step_metadata_path: str | None = Field(
        default=None,
        description="Location of the persisted next-step guidance file",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the specification was generated"
    )


class PlanRequest(BaseModel):
    """Request model for /plan endpoint.

    Accepts either a path to an existing specification file (specification_path)
    OR a raw specification document (specification). If a raw specification is
    provided, the service will materialize it under the workspace `specs/` tree
    and proceed as usual. An optional `feature_id` can be supplied to control
    the output directory naming.
    """

    specification_path: str | None = Field(
        default=None, description="Path to the specification file"
    )
    specification: str | None = Field(
        default=None, description="Raw specification content (alternative to path)"
    )
    feature_id: str | None = Field(
        default=None, description="Optional feature identifier to use for outputs"
    )
    technology_stack: str = Field(
        default="",
        description="Preferred technology stack (e.g., 'FastAPI + SQLAlchemy')",
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict, description="Additional constraints and preferences"
    )
    constitutional_gates: bool = Field(
        default=True, description="Whether to apply constitutional validation gates"
    )


class PlanResponse(BaseModel):
    """Response model for /plan endpoint."""

    success: bool = Field(..., description="Whether plan generation succeeded")
    feature_id: str | None = Field(
        default=None, description="Feature identifier associated with the plan"
    )
    implementation_plan: str = Field(..., description="Generated implementation plan")
    # Back-compat alias expected by some callers/tests
    plan: str = Field(..., description="Alias of implementation_plan for compatibility")
    plan_path: str = Field(..., description="Path to the generated plan file")
    supporting_documents: list[str] = Field(
        default_factory=list,
        description="Paths to generated supporting documents",
    )
    analysis_results: dict[str, Any] = Field(
        default_factory=dict,
        description=("Additional analysis and Mangle-enhanced reasoning artifacts"),
    )
    constitutional_compliance: dict[str, ConstitutionalValidation] = Field(
        default_factory=dict,
        description="Constitutional validation results by article",
    )
    overall_compliance_score: float = Field(
        ..., description="Overall constitutional compliance score"
    )
    compliance_threshold_met: bool = Field(
        ..., description="Whether the compliance threshold (0.75) was met"
    )
    technology_recommendations: list[str] = Field(
        default_factory=list, description="Technology stack recommendations"
    )
    architecture_decisions: list[str] = Field(
        default_factory=list, description="Key architectural decisions made"
    )
    next_steps: list[str] = Field(
        default_factory=list, description="Recommended next steps"
    )
    next_step_guidance: NextStepGuidance | None = Field(
        default=None, description="Structured next steps carried into the plan phase"
    )
    next_step_metadata_path: str | None = Field(
        default=None, description="Path to the persisted next-step guidance file"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the plan was generated"
    )


class TasksRequest(BaseModel):
    """Request model for /tasks endpoint.

    Accepts either a path to an existing plan file (plan_path) OR a raw plan
    document (plan). If a raw plan is provided, the service will materialize it
    under the workspace `specs/` tree. An optional `feature_id` can be supplied
    to control the output directory naming.
    """

    plan_path: str | None = Field(
        default=None, description="Path to the implementation plan file"
    )
    plan: str | None = Field(
        default=None, description="Raw plan content (alternative to path)"
    )
    feature_id: str | None = Field(
        default=None, description="Optional feature identifier to use for outputs"
    )
    priority_focus: str = Field(
        default="test-first",
        description=("Priority focus (test-first, library-first, integration-first)"),
    )
    team_size: int = Field(
        default=1, description="Team size for task estimation", ge=1, le=10
    )
    constitutional_gates: bool = Field(
        default=True, description="Whether to apply constitutional validation gates"
    )


class TaskBreakdown(BaseModel):
    """Individual task in the breakdown."""

    id: str = Field(..., description="Unique task identifier")
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Detailed task description")
    priority: str = Field(
        ..., description="Priority level (critical, high, medium, low)"
    )
    estimated_hours: int = Field(..., description="Estimated effort in hours")
    dependencies: list[str] = Field(
        default_factory=list, description="List of task IDs this task depends on"
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list, description="Acceptance criteria for task completion"
    )
    constitutional_requirements: list[str] = Field(
        default_factory=list,
        description="Constitutional requirements this task must satisfy",
    )


class TasksResponse(BaseModel):
    """Response model for /tasks endpoint."""

    success: bool = Field(..., description="Whether task generation succeeded")
    feature_id: str | None = Field(
        default=None, description="Feature identifier associated with the tasks"
    )
    tasks_breakdown: str = Field(..., description="Generated tasks breakdown content")
    tasks_path: str = Field(..., description="Path to the generated tasks file")
    tasks: list[TaskBreakdown] = Field(
        default_factory=list, description="Structured list of tasks"
    )
    analysis_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional analysis and Mangle-enhanced reasoning artifacts",
    )
    constitutional_compliance: dict[str, ConstitutionalValidation] = Field(
        default_factory=dict, description="Constitutional validation results by article"
    )
    overall_compliance_score: float = Field(
        ..., description="Overall constitutional compliance score"
    )
    compliance_threshold_met: bool = Field(
        ..., description="Whether the compliance threshold (0.75) was met"
    )
    estimated_total_hours: int = Field(
        ..., description="Total estimated hours for all tasks"
    )
    critical_path: list[str] = Field(
        default_factory=list, description="Critical path task IDs"
    )
    next_steps: list[str] = Field(
        default_factory=list, description="Recommended next steps"
    )
    next_step_guidance: NextStepGuidance | None = Field(
        default=None, description="Structured next steps reused during tasks phase"
    )
    next_step_metadata_path: str | None = Field(
        default=None, description="Path to the persisted next-step guidance file"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the tasks were generated"
    )


class ConstitutionalGateResult(BaseModel):
    """Result of a constitutional gate evaluation."""

    gate_name: str = Field(..., description="Name of the constitutional gate")
    passed: bool = Field(..., description="Whether the gate passed")
    score: float = Field(..., description="Gate score (0.0-1.0)")
    violations: list[str] = Field(
        default_factory=list, description="List of violations"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations for improvement"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the gate was evaluated"
    )

