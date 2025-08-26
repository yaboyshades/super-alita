"""
Event schemas for the Super Alita agent communication bus.
All events are versioned, typed, and can carry semantic embeddings.
Enhanced with DTA 2.0 cognitive processing events.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Import cognitive turn types if available
try:
    from src.dta.types import CognitiveTurnRecord

    COGNITIVE_TURN_AVAILABLE = True
except ImportError:
    COGNITIVE_TURN_AVAILABLE = False


class TelemetryInfo(BaseModel):
    """Shared telemetry fields present on all events."""

    source_plugin: str
    conversation_id: str | None = Field(
        default=None, description="Conversation/session identifier"
    )
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Correlation identifier across events",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Event creation timestamp",
    )


class BaseEvent(TelemetryInfo):
    """Base event class with common fields for all events."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    version: str = "1.0"
    trace_id: str | None = Field(
        default=None, description="Trace identifier for debugging"
    )
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", populate_by_name=True)  # pydantic v2


# Event type aliases for back-compatibility and publisher variance
EVENT_ALIASES: dict[str, str] = {
    "conversation_message": "conversation",
    "message": "conversation",  # some clients publish 'message'
}


class CognitiveEvent(BaseEvent):
    """Events related to cognitive processes."""

    model_config = ConfigDict(extra="allow")


class CognitiveTurnInitiatedEvent(CognitiveEvent):
    """
    Event fired when ConversationPlugin detects a task request and initiates
    a cognitive turn in the DTA 2.0 pipeline.

    This event serves as the unambiguous trigger for advanced cognitive processing,
    ensuring the DTA 2.0 preprocessor is the designated next step.
    """

    event_type: str = "cognitive_turn_initiated"
    user_message: str
    session_id: str
    conversation_id: str
    original_event_id: str  # For traceability back to the original conversation event
    intent_confidence: float = 0.9  # Confidence that this is truly a task request
    cognitive_context: dict[str, Any] = Field(
        default_factory=dict
    )  # Additional context for processing

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class SkillProposalEvent(CognitiveEvent):
    """Event for proposing new skills to the agent."""

    event_type: str = "skill_proposal"
    skill_name: str
    skill_description: str
    skill_code: str
    parent_skills: list[str] = Field(default_factory=list)
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
    query: str | None = None
    similarity_threshold: float | None = None


class MemoryUpsertEvent(BaseEvent):
    """Event for memory upsert operations."""

    event_type: str = "memory_upsert"
    memory_id: str
    content: dict[str, Any]
    hierarchy_path: list[str]
    embedding: list[float] | None = None
    metadata: dict[str, Any] | None = None


class PlanningEvent(BaseEvent):
    """Events related to planning and reasoning."""

    goal: str
    current_state: dict[str, Any]
    action_space: list[str]
    plan: list[dict[str, Any]] | None = None
    confidence: float | None = None


class GoalReceivedEvent(BaseEvent):
    """Event triggered when a goal is received from conversation or user input."""

    event_type: str = "goal_received"
    goal: str
    session_id: str
    tools_needed: list[str] = Field(
        default_factory=list
    )  # NEW: Tools selected by router
    goal_description: str | None = None  # Alias for goal
    context: dict[str, Any] = Field(default_factory=dict)
    priority: str = "normal"  # normal, high, urgent

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure goal_description is set for backward compatibility
        if not self.goal_description:
            self.goal_description = self.goal


class PlanningDecisionEvent(BaseEvent):
    """Event for LADDER-AOG planning decisions."""

    event_type: str = "planning_decision"
    plan_id: str
    decision: str


class StateTransitionEvent(BaseEvent):
    """Event for FSM state transitions."""

    event_type: str = "state_transition"
    previous_state: str
    new_state: str
    reason: str
    confidence: float
    state_embedding: list[float] | None = None
    confidence_score: float
    causal_factors: list[str] = Field(default_factory=list)


class FSMStateEvent(BaseEvent):
    """Events for semantic FSM state transitions."""

    event_type: str = "fsm_state_change"
    from_state: str
    to_state: str
    trigger: str
    semantic_similarity: float | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class EvolutionEvent(BaseEvent):
    """Events for evolutionary processes."""

    event_type: str = "evolution"
    generation: int
    population_size: int
    best_fitness: float
    mutation_rate: float
    selection_method: str
    evolved_entities: list[dict[str, Any]] = Field(default_factory=list)


class MCTSEvent(BaseEvent):
    """Events for Monte Carlo Tree Search operations."""

    event_type: str = "mcts_operation"
    node_id: str
    operation: str  # "select", "expand", "simulate", "backpropagate"
    value: float
    visit_count: int
    depth: int
    state_representation: dict[str, Any] = Field(default_factory=dict)


class PAECycleEvent(BaseEvent):
    """Events for Perceive-Act-Evolve cycle."""

    event_type: str = "pae_cycle"
    cycle_id: str
    phase: str  # "perceive", "act", "evolve"
    perception_data: dict[str, Any] = Field(default_factory=dict)
    action_taken: str | None = None
    evolution_result: dict[str, Any] | None = None
    fitness_score: float = 0.0


class AtomBirthEvent(BaseEvent):
    """Event for tracking atom creation and genealogy."""

    event_type: str = "atom_birth"
    atom_key: str
    parent_keys: list[str] = Field(default_factory=list)
    birth_context: str
    lineage_metadata: dict[str, Any] = Field(default_factory=dict)
    genealogy_depth: int = 0
    darwin_godel_signature: str = ""


class AtomDeathEvent(BaseEvent):
    """Event for tracking atom dissolution/death."""

    event_type: str = "atom_death"
    atom_key: str
    death_reason: str
    final_state: dict[str, Any] = Field(default_factory=dict)
    contribution_score: float = 0.0


class NeuralActivityEvent(BaseEvent):
    """Event for tracking neural atom activity patterns."""

    event_type: str = "neural_activity"
    activity_pattern: list[float]
    activation_threshold: float
    synaptic_weights: dict[str, float] = Field(default_factory=dict)
    hebbian_update: bool = False
    attention_focus: list[str] | None = None


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
    metrics: dict[str, float] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)


class ConversationEvent(BaseEvent):
    """
    Lightweight telemetry/message envelope for conversation streams.
    Prevents NameError in conversation flow and enables downstream analytics.
    Supports both 'text' and 'user_message' for compatibility.
    """

    event_type: str = "conversation"
    text: str | None = None
    user_message: str | None = Field(None, alias="message")  # Accept 'message' as alias
    role: str = "user"  # user, assistant, system
    session_id: str | None = None
    conversation_id: str | None = Field(
        None, alias="conv_id"
    )  # Accept 'conv_id' as alias

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    def _normalize(cls, data):
        """Normalize conversation event data before validation."""
        if isinstance(data, dict):
            # Back-compat alias
            if data.get("event_type") == "conversation_message":
                data["event_type"] = "conversation"
            # Prefer 'text', but accept 'user_message'
            if not data.get("text") and data.get("user_message"):
                data["text"] = data["user_message"]
            elif not data.get("user_message") and data.get("text"):
                data["user_message"] = data["text"]
        return data

    @model_validator(mode="after")
    def validate_message_fields(self):
        """Ensure at least one message field is populated and sync them."""
        # If only one field is provided, mirror it to the other
        if self.text and not self.user_message:
            self.user_message = self.text
        elif self.user_message and not self.text:
            self.text = self.user_message
        elif not self.text and not self.user_message:
            raise ValueError("Either 'text' or 'user_message' must be provided")

        return self

    @property
    def message(self) -> str:
        """Get the message content regardless of which field was used."""
        return self.text or self.user_message or ""


class AtomizeTextRequest(BaseEvent):
    """Request to atomize text into semantic units."""

    event_type: str = "atomize_text_request"
    text: str
    anchors: list[str] | None = None  # existing atom_ids or label hints to bond to
    max_notes: int = 5  # how many NOTE atoms to propose
    context: dict[str, Any] | None = None  # For traceability (e.g., task_id)


class BatchAtomsCreated(BaseEvent):
    """Event indicating multiple atoms were created."""

    event_type: str = "batch_atoms_created"
    atoms: list[dict[str, Any]]
    source_request: str | None = None


class BatchBondsAdded(BaseEvent):
    """Event indicating multiple bonds were added."""

    event_type: str = "batch_bonds_added"
    bonds: list[dict[str, Any]]
    source_request: str | None = None


class PreprocessedActionEvent(BaseEvent):
    """
    Event containing Pythonic preprocessed action with guaranteed valid code skeleton.
    This serves as the communication contract between PythonicPreprocessorPlugin
    and the simplified LLMPlannerPlugin.
    """

    event_type: str = "preprocessed_action"
    session_id: str
    conversation_id: str
    original_message: str
    python_intent: str  # The original Python function call
    action: dict[str, Any]  # Serialized Action object
    code_skeleton: str  # Guaranteed-valid executable code
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResponseEvent(BaseEvent):
    """Events for agent responses to conversations."""

    event_type: str = "agent_response"
    session_id: str
    response_text: str
    reasoning: str = ""
    context: dict[str, Any] = Field(default_factory=dict)
    response_id: str


class AgentThinkingEvent(BaseEvent):
    """Events for indicating agent thinking/processing."""

    event_type: str = "agent_thinking"
    session_id: str
    stage: str  # "processing", "analyzing", "generating"
    message_id: str = ""


class SystemStatusRequestEvent(BaseEvent):
    """Event requesting comprehensive system status."""

    event_type: str = "system_status_request"
    session_id: str | None = None
    detail_level: str = "comprehensive"  # basic, detailed, comprehensive


class SystemStatusResponseEvent(BaseEvent):
    """Event containing system status response."""

    event_type: str = "system_status_response"
    session_id: str | None = None
    status_report: dict[str, Any] = Field(default_factory=dict)


class ToolCallRequestEvent(BaseEvent):
    """Event requesting tool execution."""

    event_type: str = "tool_call_request"
    action: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    source: str = "unknown"


class ToolExecutionEvent(BaseEvent):
    """Event for atom-based tool execution requests."""

    event_type: str = "tool_execution"
    tool_key: str
    parameters: dict[str, Any]
    session_id: str
    request_id: str


class ToolResultEvent(BaseEvent):
    """Event signaling the result of a tool invocation."""

    event_type: str = Field(default="tool_result")
    tool_call_id: str = Field(..., description="ID matching the original ToolCallEvent")
    conversation_id: str = Field(..., description="session id for routing")
    session_id: str = Field(..., description="session id for the conversation")
    success: bool = Field(..., description="whether the tool execution succeeded")
    result: dict[str, Any] = Field(
        default_factory=dict, description="tool execution results"
    )
    error: str | None = Field(None, description="error message if execution failed")

    model_config = ConfigDict(frozen=True)  # makes the model hashable and immutable


class AtomCreatedEvent(BaseEvent):
    """Event for successful atom creation."""

    event_type: str = "atom_created"
    name: str
    atom_id: str
    session_id: str
    code: str | None = None


class WebSearchEvent(BaseEvent):
    """Event for web search requests."""

    event_type: str = "web_search"
    query: str
    web_k: int = 5
    github_k: int = 5
    auto_wrap: bool = False
    session_id: str = "default"


class WebSearchResultEvent(BaseEvent):
    """Event for web search results."""

    event_type: str = "web_search_result"
    query: str
    result: dict[str, Any]
    session_id: str = "default"


class PlanExecutionEvent(BaseEvent):
    """Event for executing a plan."""

    event_type: str = "plan_execution"
    plan: dict[str, Any]
    conversation_id: str


class PlanStepResult(BaseModel):
    """Result of executing a single plan step."""

    step_id: str
    success: bool
    result: Any = None
    error: str | None = None
    duration: float = 0.0


class AtomGapRequestEvent(BaseEvent):
    """Event requesting brainstorming for missing atoms"""

    event_type: str = "atom_gap_request"
    task: str
    context: dict[str, Any] = Field(default_factory=dict)


class AtomGapEvent(BaseEvent):
    """Event indicating a missing tool/capability gap detected during execution"""

    event_type: str = "atom_gap"
    missing_tool: str
    description: str
    session_id: str
    conversation_id: str
    gap_id: str


class AtomReadyEvent(BaseEvent):
    """Event indicating a new atom is ready"""

    event_type: str = "atom_ready"
    atom: dict[str, Any]


class ComposeRequestEvent(BaseEvent):
    """Event requesting composition of atoms for a goal"""

    event_type: str = "compose_request"
    goal: str
    params: dict[str, Any] = Field(default_factory=dict)


# Legacy events (keeping for compatibility)
class ToolCallEvent(BaseEvent):
    """Event signaling a request to invoke a tool."""

    event_type: str = "tool_call"
    tool_name: str
    parameters: dict[str, Any]
    conversation_id: str
    session_id: str
    tool_call_id: str


# NEW: Structured event models for clean serialization
class AgentReplyEvent(BaseModel):
    """Agent reply event with proper structure."""

    event_type: str = "agent_reply"
    text: str
    channel: str = "chat"
    message: str | None = None  # For backward compatibility
    session_id: str | None = None
    conversation_id: str | None = None
    correlation_id: str | None = None
    # Optional: classify nature of reply (info|gap|error)
    kind: str = "info"

    model_config = ConfigDict(
        populate_by_name=True,  # Fixed: was populate_by_name
        extra="ignore",
    )


class ToolCreatedEvent(BaseEvent):
    """Event emitted when a tool is successfully created."""

    event_type: str = "tool_created"
    tool_id: str
    name: str
    code_path: str
    contract: dict[str, Any] = Field(default_factory=dict)
    registry_key: str


class ShowCreatedToolRequest(BaseEvent):
    """Request to show details of a created tool."""

    event_type: str = "show_created_tool_request"
    tool_name: str  # Name of the tool to show
    # if omitted, plugins can use "last created"
    tool_id: str | None = None


class CognitiveAtomEvent(BaseEvent):
    """Cognitive atom event for distributed cognition."""

    event_type: str = "cognitive_atom"
    atom: dict[str, Any] | None = (
        None  # Full atom payload (id, type, meta, etc.) - optional for backward compatibility
    )
    atom_id: str | None = None
    atom_type: str | None = None
    correlation_id: str | None = None


class AtomReadyEvent(BaseEvent):
    """Atom ready event for tool registration notifications."""

    event_type: str = "atom_ready"
    atom: dict[str, Any]
    tool_name: str | None = None
    file_path: str | None = None
    activity_id: str | None = None
    gap_id: str | None = None
    session_id: str | None = None
    conversation_id: str | None = None
    correlation_id: str | None = None


class ConversationMessageEvent(BaseEvent):
    """Conversation message event."""

    event_type: str = "conversation_message"
    user_message: str
    session_id: str
    conversation_id: str


class AtomCreateEvent(BaseEvent):
    """Event for creating and storing a new atom."""

    event_type: str = "atom_create"
    tool_name: str
    description: str
    code: str
    created_by: str = "system"


class AtomRunEvent(BaseEvent):
    """Event for executing an existing atom."""

    event_type: str = "atom_run"
    tool_name: str
    args: str = ""
    requested_by: str = "system"


class AtomResultEvent(BaseEvent):
    """Event containing atom execution results."""

    event_type: str = "atom_result"
    tool_name: str
    args: str
    stdout: str = ""
    stderr: str = ""
    success: bool = True
    execution_time: float = 0.0


class UserMessageEvent(BaseEvent):
    """Event for user messages from chat interfaces."""

    event_type: str = "user_message"
    text: str
    session_id: str | None = None


# Event type registry for factory function


# Event type registry for dynamic event creation (no duplicates)
EVENT_TYPES = {
    "skill_proposal": SkillProposalEvent,
    "skill_evaluation": SkillEvaluationEvent,
    "memory": MemoryEvent,
    "memory_upsert": MemoryUpsertEvent,
    "planning": PlanningEvent,
    "goal_received": GoalReceivedEvent,
    "planning_decision": PlanningDecisionEvent,
    "state_transition": StateTransitionEvent,
    "fsm_state_change": FSMStateEvent,
    "evolution": EvolutionEvent,
    "mcts_operation": MCTSEvent,
    "pae_cycle": PAECycleEvent,
    "atom_birth": AtomBirthEvent,
    "atom_death": AtomDeathEvent,
    "neural_activity": NeuralActivityEvent,
    "system": SystemEvent,
    "health_check": HealthCheckEvent,
    "conversation": ConversationEvent,
    "preprocessed_action": PreprocessedActionEvent,
    "agent_response": AgentResponseEvent,
    "agent_thinking": AgentThinkingEvent,
    "system_status_request": SystemStatusRequestEvent,
    "system_status_response": SystemStatusResponseEvent,
    "cognitive_turn_initiated": CognitiveTurnInitiatedEvent,
    "tool_call_request": ToolCallRequestEvent,
    "tool_execution": ToolExecutionEvent,
    "tool_result": ToolResultEvent,
    "atom_created": AtomCreatedEvent,
    "web_search": WebSearchEvent,
    "web_search_result": WebSearchResultEvent,
    "plan_execution": PlanExecutionEvent,
    "atom_gap_request": AtomGapRequestEvent,
    "atom_gap": AtomGapEvent,
    "atom_ready": AtomReadyEvent,
    "compose_request": ComposeRequestEvent,
    "tool_call": ToolCallEvent,
    "atom_create": AtomCreateEvent,
    "atom_run": AtomRunEvent,
    "atom_result": AtomResultEvent,
    "user_message": UserMessageEvent,
    "atomize_text_request": AtomizeTextRequest,
    "batch_atoms_created": BatchAtomsCreated,
    "batch_bonds_added": BatchBondsAdded,
    # NEW: Structured event models for clean serialization
    "agent_reply": AgentReplyEvent,
    "cognitive_atom": CognitiveAtomEvent,
    "conversation_message": ConversationMessageEvent,
    "tool_created": ToolCreatedEvent,
    "show_created_tool_request": ShowCreatedToolRequest,
}

# Map event_type → model for deserialization (optimized lookup)
EVENT_TYPE_TO_MODEL = EVENT_TYPES


# DTA 2.0 Cognitive Processing Events
if COGNITIVE_TURN_AVAILABLE:

    class CognitiveTurnCompletedEvent(BaseEvent):
        """Event published after the preprocessor generates a CognitiveTurnRecord."""

        event_type: str = "cognitive_turn_completed"
        turn_record: CognitiveTurnRecord
        session_id: str
        conversation_id: str

    # Add to event types registry
    EVENT_TYPES["cognitive_turn_completed"] = CognitiveTurnCompletedEvent


def create_event(event_type: str, **kwargs) -> BaseEvent:
    """Factory function to create events by type."""

    event_class = EVENT_TYPES.get(event_type, BaseEvent)
    return event_class(event_type=event_type, **kwargs)


def serialize_event(event: BaseEvent) -> dict[str, Any]:
    """Serialize event to dictionary for transmission."""

    return event.model_dump()


def deserialize_event(data: dict[str, Any]) -> BaseEvent:
    """Deserialize event from dictionary."""

    event_type = data.get("event_type", "base")
    event_class = EVENT_TYPES.get(event_type, BaseEvent)
    return event_class(**data)
