# Version: 3.0.0
# Description: The agent's brain. Handles the PLANNING and SELECTION cognitive stages.

import json
import logging
import os
from datetime import datetime
from typing import Any

from src.core.global_workspace import AttentionLevel, GlobalWorkspace, WorkspaceEvent
from src.core.neural_atom import NeuralStore
from src.core.plugin_interface import PluginInterface
from src.core.schemas import (
    ConversationEvent,
    CREATORRequest,
    TaskRequest,
    TaskResult,
    TaskType,
    ToolCallEvent,
)
from src.core.telemetry_broker import (
    build_context_envelope,
)
from src.core.telemetry_broker import (
    ingest_event as broker_ingest,
)

# Try to import Google Generative AI, but don't fail if not available
try:  # pragma: no cover - optional dependency
    import google.generativeai as genai  # type: ignore

    HAS_GEMINI = True  # noqa: N816 (external constant style)
except ImportError:  # pragma: no cover
    HAS_GEMINI = False  # noqa: N816

logger = logging.getLogger(__name__)


class LLMPlannerPlugin(PluginInterface):
    """
    Uses LLM reasoning to formulate plans and select Neural Atoms.

    This plugin implements the PLANNING and SELECTION stages of the 8-stage
    cognitive cycle, using advanced LLM capabilities to make intelligent
    routing decisions and detect capability gaps.
    """

    def __init__(self):
        super().__init__()
        self.workspace: GlobalWorkspace | None = None
        self.store: NeuralStore | None = None
        self.llm_client: Any | None = None
        self.available_neural_atoms: dict[str, dict[str, Any]] = {}
        self.conversation_context: dict[str, list[dict[str, str]]] = {}

        # Performance tracking
        self.planning_stats = {
            "total_requests": 0,
            "successful_plans": 0,
            "capability_gaps_detected": 0,
            "neural_atoms_utilized": 0,
            "average_planning_time": 0.0,
        }

    @property
    def name(self) -> str:
        """Return the unique name identifier for this plugin."""
        return "llm_planner"

    async def setup(
        self, workspace: GlobalWorkspace, store: NeuralStore, config: dict[str, Any]
    ):
        """Initialize the LLM planner with workspace and store."""
        await super().setup(workspace, store, config)

        self.workspace = workspace
        self.store = store

        # Initialize LLM client if available
        if HAS_GEMINI and config.get("model"):
            await self._initialize_llm_client(config)
        else:
            logger.warning(
                "Gemini not available or no model configured - using fallback logic"
            )

        # Load available Neural Atoms from store
        await self._load_available_neural_atoms()

        logger.info("LLM Planner Plugin initialized for PLANNING and SELECTION stages")

    async def _initialize_llm_client(self, config: dict[str, Any]):
        """Initialize the Google Gemini client."""
        try:  # pragma: no cover - initialization side effects
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key and HAS_GEMINI:
                genai.configure(api_key=api_key)  # type: ignore[attr-defined]
                model_name = config.get("model", "gemini-2.0-flash-exp")
                # GenerativeModel attr not typed locally
                self.llm_client = genai.GenerativeModel(model_name)  # type: ignore[attr-defined]
                logger.info(f"LLM client initialized with model: {model_name}")
            else:
                logger.warning("GEMINI_API_KEY not found or gemini lib missing")
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to initialize LLM client: {e}")

    async def _load_available_neural_atoms(self):
        """Load available Neural Atoms from the store."""
        try:
            # This would query the neural store for available atoms
            # For now, we'll use a basic set
            self.available_neural_atoms = {
                "memory_manager": {
                    "name": "Memory Manager",
                    "description": "Manages long-term memory and knowledge storage",
                    "capabilities": ["save", "recall", "search"],
                    "parameters": {
                        "action": "string",
                        "content": "string",
                        "query": "string",
                    },
                },
                "web_agent": {
                    "name": "Web Agent",
                    "description": "Searches web and GitHub for information",
                    "capabilities": ["web_search", "github_search"],
                    "parameters": {"query": "string", "search_type": "string"},
                },
            }

            logger.info(f"Loaded {len(self.available_neural_atoms)} Neural Atoms")

        except Exception as e:
            logger.error(f"Error loading Neural Atoms: {e}")

    async def start(self):
        """Start the LLM planner and subscribe to workspace events."""
        await super().start()

        if self.workspace:
            # Subscribe to task requests and conversation events
            self.workspace.subscribe("llm_planner", self._handle_workspace_event)

            logger.info("LLM Planner subscribed to Global Workspace events")

    async def shutdown(self):
        """Gracefully shutdown the LLM planner."""
        await super().shutdown()
        logger.info("LLM Planner Plugin shutdown complete")

    async def _handle_workspace_event(self, event: WorkspaceEvent):
        """Handle events from the Global Workspace."""
        try:
            if isinstance(event.data, dict):
                event_type = event.data.get("type")

                if event_type == "task_request":
                    await self._handle_task_request(TaskRequest(**event.data))
                elif event_type == "conversation_message":
                    await self._handle_conversation(ConversationEvent(**event.data))
                else:
                    logger.debug(f"Unhandled event type: {event_type}")

        except Exception as e:
            logger.error(f"Error handling workspace event: {e}")

    async def _handle_task_request(self, task: TaskRequest):
        """Handle incoming task requests through the planning stage."""
        start_time = datetime.now()
        self.planning_stats["total_requests"] += 1

        try:
            logger.info(f"🧠 Planning stage: Processing task {task.task_id}")
            # Ingest planner start event (feature-flag aware inside broker_ingest)
            broker_ingest(
                "planner",
                f"Planning task {task.task_id}",
                importance=min(2.0, 1.0 + (task.priority or 0) * 0.1),
                meta={"task_type": task.task_type.value},
            )

            # Analyze task and determine approach
            plan = await self._analyze_and_plan(task)

            if plan["action"] == "NEURAL_ATOM":
                # Route to appropriate Neural Atom
                await self._route_to_neural_atom(task, plan)
                self.planning_stats["neural_atoms_utilized"] += 1

            elif plan["action"] == "CAPABILITY_GAP":
                # Detect capability gap and trigger CREATOR
                await self._handle_capability_gap(task, plan)
                self.planning_stats["capability_gaps_detected"] += 1

            elif plan["action"] == "DIRECT_RESPONSE":
                # Handle directly without tools
                await self._provide_direct_response(task, plan)

            self.planning_stats["successful_plans"] += 1

            # Update performance metrics
            planning_time = (datetime.now() - start_time).total_seconds()
            self._update_planning_stats(planning_time)

            # Record outcome telemetry
            broker_ingest(
                "planner",
                f"Plan decided for task {task.task_id}: {plan['action']}",
                importance=1.2 if plan["action"] == "CAPABILITY_GAP" else 1.0,
                meta={
                    k: v for k, v in plan.items() if k not in {"parameters", "response"}
                },
            )

        except Exception as e:
            logger.error(f"Error in planning stage for task {task.task_id}: {e}")
            await self._handle_planning_error(task, str(e))

    async def _analyze_and_plan(self, task: TaskRequest) -> dict[str, Any]:
        """Analyze the task and create an execution plan."""
        if self.llm_client:
            return await self._llm_based_planning(task)
        return await self._fallback_planning(task)

    async def _llm_based_planning(self, task: TaskRequest) -> dict[str, Any]:
        """Use LLM for intelligent planning and Neural Atom selection.

        Integrates TelemetryBroker context envelope (if enabled) to reduce
        prompt bloat and supply the most relevant recent system telemetry.
        """
        try:
            envelope: dict[str, Any] | None = None
            # Only attempt build if feature flag enabled (checked inside helper)
            envelope = build_context_envelope()
            if envelope:
                broker_ingest(
                    "planner",
                    "Context envelope attached",
                    importance=0.8,
                    meta={
                        "hash": envelope.get("hash"),
                        "total_tokens": envelope.get("total_tokens"),
                        "categories": list(envelope.get("categories", {}).keys()),
                    },
                )

            # Create comprehensive prompt for planning
            prompt = self._create_planning_prompt(task, envelope=envelope)

            # Generate plan using LLM
            response = await self.llm_client.generate_content_async(prompt)

            # Parse LLM response into structured plan
            plan = self._parse_llm_response(response.text)

            logger.info(f"LLM generated plan: {plan['action']}")
            return plan

        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            return await self._fallback_planning(task)

    def _create_planning_prompt(
        self, task: TaskRequest, *, envelope: dict[str, Any] | None = None
    ) -> str:
        """Create a comprehensive prompt for LLM-based planning.

        If a TelemetryBroker envelope is supplied, it is summarized and
        appended so the model receives only curated recent telemetry instead of
        raw unbounded logs.
        """
        atoms_description = self._format_neural_atoms_for_prompt()

        envelope_section = ""
        if envelope and envelope.get("categories"):
            # Summarize each category by listing scored messages newest-first as provided
            cat_parts: list[str] = []
            for cat, payload in envelope["categories"].items():
                events = payload.get("events", [])
                # Concise list: score|message
                lines = [
                    f"  - ({e['score']}) {e['message']}"[:220]
                    for e in events  # trunc to guard prompt size
                ]
                cat_parts.append(f"* {cat}:\n" + "\n".join(lines))
            envelope_section = (
                "\nCurated Recent Telemetry (scored):\n" + "\n".join(cat_parts) + "\n"
            )

        prompt = (
            "You are the central planner for an advanced AI agent with Neural Atom capabilities.\n\n"
            "Task Analysis:\n"
            f"- Task ID: {task.task_id}\n"
            f"- Type: {task.task_type.value}\n"
            f"- Description: {task.description}\n"
            f"- Priority: {task.priority}\n"
            f"- Context: {task.context}\n\n"
            "Available Neural Atoms:\n"
            f"{atoms_description}\n"
            f"{envelope_section}"
            "Planning Instructions:\n"
            "1. Analyze the task requirements carefully\n"
            "2. Determine if an existing Neural Atom can handle this task\n"
            "3. If no suitable atom exists, identify what capability is needed\n"
            "4. Choose the most appropriate action\n\n"
            "Response Format (choose one):\n\n"
            "NEURAL_ATOM: <atom_name>\n"
            'PARAMETERS: {"param": "value"}\n'
            "REASONING: <why this atom was selected>\n\n"
            "OR\n\n"
            "CAPABILITY_GAP: <description of missing capability>\n"
            "REASONING: <why no existing atom can handle this>\n\n"
            "OR\n\n"
            "DIRECT_RESPONSE: <direct answer to user>\n"
            "REASONING: <why no Neural Atom is needed>\n\n"
            "Your Response:"
        )
        return prompt

    def _format_neural_atoms_for_prompt(self) -> str:
        """Format available Neural Atoms for the LLM prompt."""
        descriptions = []
        for atom_name, atom_info in self.available_neural_atoms.items():
            descriptions.append(
                f"- {atom_name}: {atom_info['description']}\n"
                f"  Capabilities: {', '.join(atom_info['capabilities'])}\n"
                f"  Parameters: {atom_info['parameters']}"
            )
        return "\n\n".join(descriptions)

    def _parse_llm_response(self, response_text: str) -> dict[str, Any]:
        """Parse LLM response into a structured plan."""
        lines = response_text.strip().split("\n")

        for line in lines:
            line = line.strip()

            if line.startswith("NEURAL_ATOM:"):
                atom_name = line.split(":", 1)[1].strip()
                parameters = {}
                reasoning = ""

                # Extract parameters and reasoning
                for next_line in lines[lines.index(line) + 1 :]:
                    if next_line.strip().startswith("PARAMETERS:"):
                        param_text = next_line.split(":", 1)[1].strip()
                        try:
                            parameters = json.loads(param_text)
                        except json.JSONDecodeError:
                            pass
                    elif next_line.strip().startswith("REASONING:"):
                        reasoning = next_line.split(":", 1)[1].strip()

                return {
                    "action": "NEURAL_ATOM",
                    "atom_name": atom_name,
                    "parameters": parameters,
                    "reasoning": reasoning,
                }

            if line.startswith("CAPABILITY_GAP:"):
                gap_description = line.split(":", 1)[1].strip()
                reasoning = ""

                for next_line in lines[lines.index(line) + 1 :]:
                    if next_line.strip().startswith("REASONING:"):
                        reasoning = next_line.split(":", 1)[1].strip()
                        break

                return {
                    "action": "CAPABILITY_GAP",
                    "gap_description": gap_description,
                    "reasoning": reasoning,
                }

            if line.startswith("DIRECT_RESPONSE:"):
                response = line.split(":", 1)[1].strip()
                reasoning = ""

                for next_line in lines[lines.index(line) + 1 :]:
                    if next_line.strip().startswith("REASONING:"):
                        reasoning = next_line.split(":", 1)[1].strip()
                        break

                return {
                    "action": "DIRECT_RESPONSE",
                    "response": response,
                    "reasoning": reasoning,
                }

        # Fallback if parsing fails
        return {
            "action": "DIRECT_RESPONSE",
            "response": "I need to think about that more carefully.",
            "reasoning": "Unable to parse LLM response",
        }

    async def _fallback_planning(self, task: TaskRequest) -> dict[str, Any]:
        """Fallback planning when LLM is not available."""
        description_lower = task.description.lower()

        # Simple keyword-based routing
        if any(word in description_lower for word in ["remember", "save", "store"]):
            return {
                "action": "NEURAL_ATOM",
                "atom_name": "memory_manager",
                "parameters": {"action": "save", "content": task.description},
                "reasoning": "Keyword-based routing to memory manager",
            }

        if any(word in description_lower for word in ["search", "find", "look"]):
            return {
                "action": "NEURAL_ATOM",
                "atom_name": "web_agent",
                "parameters": {"query": task.description},
                "reasoning": "Keyword-based routing to web agent",
            }

        if any(word in description_lower for word in ["create", "build", "make"]):
            return {
                "action": "CAPABILITY_GAP",
                "gap_description": f"Tool creation requested: {task.description}",
                "reasoning": "Keyword-based gap detection for tool creation",
            }

        return {
            "action": "DIRECT_RESPONSE",
            "response": "I understand your request. Let me help you with that.",
            "reasoning": "General conversational response",
        }

    async def _route_to_neural_atom(self, task: TaskRequest, plan: dict[str, Any]):
        """Route task to the appropriate Neural Atom."""
        atom_name = plan["atom_name"]
        parameters = plan["parameters"]

        # Create tool call event
        tool_call = ToolCallEvent(
            tool_name=atom_name,
            parameters=parameters,
            session_id=task.task_id,
            request_id=f"plan_{task.task_id}",
        )

        # Broadcast to workspace
        await self.workspace.update(
            data={"type": "tool_call", **tool_call.model_dump()},
            source="llm_planner",
            attention_level=AttentionLevel.HIGH,
        )

        logger.info(f"Routed task {task.task_id} to Neural Atom: {atom_name}")

    async def _handle_capability_gap(self, task: TaskRequest, plan: dict[str, Any]):
        """Handle detected capability gaps by triggering CREATOR."""
        gap_description = plan["gap_description"]

        # Create CREATOR request
        creator_request = CREATORRequest(
            request_id=f"gap_{task.task_id}",
            capability_description=gap_description,
            context={"original_task": task.model_dump()},
            priority=task.priority,
            requester="llm_planner",
        )

        # Broadcast capability gap event
        await self.workspace.update(
            data={"type": "creator_request", **creator_request.model_dump()},
            source="llm_planner",
            attention_level=AttentionLevel.HIGH,
        )

        logger.info(
            f"Capability gap detected for task {task.task_id}: {gap_description}"
        )

    async def _provide_direct_response(self, task: TaskRequest, plan: dict[str, Any]):
        """Provide direct response without using Neural Atoms."""
        response = plan["response"]

        # Create task result
        result = TaskResult(
            task_id=task.task_id,
            success=True,
            result={"response": response},
            execution_time=0.1,
            stage_completed=TaskType.PLANNING,
        )

        # Broadcast result
        await self.workspace.update(
            data={"type": "task_result", **result.model_dump()},
            source="llm_planner",
            attention_level=AttentionLevel.MEDIUM,
        )

        logger.info(f"Direct response provided for task {task.task_id}")

    async def _handle_planning_error(self, task: TaskRequest, error_message: str):
        """Handle errors during planning stage."""
        error_result = TaskResult(
            task_id=task.task_id,
            success=False,
            error=error_message,
            execution_time=0.0,
            stage_completed=TaskType.PLANNING,
        )

        await self.workspace.update(
            data={"type": "task_result", **error_result.model_dump()},
            source="llm_planner",
            attention_level=AttentionLevel.HIGH,
        )

    def _update_planning_stats(self, planning_time: float):
        """Update planning performance statistics."""
        # Update average planning time using exponential moving average
        alpha = 0.1
        if self.planning_stats["average_planning_time"] == 0.0:
            self.planning_stats["average_planning_time"] = planning_time
        else:
            self.planning_stats["average_planning_time"] = (
                alpha * planning_time
                + (1 - alpha) * self.planning_stats["average_planning_time"]
            )

    async def _handle_conversation(self, event: ConversationEvent):
        """Handle conversation events (legacy compatibility)."""
        # Convert conversation to task request
        task = TaskRequest(
            task_id=f"conv_{event.session_id}",
            task_type=TaskType.PLANNING,
            description=event.user_message,
            context=event.context,
            requester="conversation",
        )

        await self._handle_task_request(task)

    def get_planning_stats(self) -> dict[str, Any]:
        """Get current planning statistics."""
        return {
            **self.planning_stats,
            "available_neural_atoms": len(self.available_neural_atoms),
            "llm_client_available": self.llm_client is not None,
        }
