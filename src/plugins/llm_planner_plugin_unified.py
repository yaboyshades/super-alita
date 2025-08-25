# Version: 3.0.0
# Description: The agent's brain. Handles the PLANNING and SELECTION cognitive stages.

import json
import logging
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

# Try to import Google Generative AI, but don't fail if not available
try:
    import google.generativeai as genai

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

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
        try:
            import os

            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)

                model_name = config.get("model", "gemini-2.0-flash-exp")
                self.llm_client = genai.GenerativeModel(model_name)

                logger.info(f"LLM client initialized with model: {model_name}")
            else:
                logger.warning("GEMINI_API_KEY not found in environment")
        except Exception as e:
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

        except Exception as e:
            logger.error(f"Error in planning stage for task {task.task_id}: {e}")
            await self._handle_planning_error(task, str(e))

    async def _analyze_and_plan(self, task: TaskRequest) -> dict[str, Any]:
        """Analyze the task and create an execution plan."""
        if self.llm_client:
            return await self._llm_based_planning(task)
        return await self._fallback_planning(task)

    async def _llm_based_planning(self, task: TaskRequest) -> dict[str, Any]:
        """Use LLM for intelligent planning and Neural Atom selection."""
        try:
            # Create comprehensive prompt for planning
            prompt = self._create_planning_prompt(task)

            # Generate plan using LLM
            response = await self.llm_client.generate_content_async(prompt)

            # Parse LLM response into structured plan
            plan = self._parse_llm_response(response.text)

            logger.info(f"LLM generated plan: {plan['action']}")
            return plan

        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            return await self._fallback_planning(task)

    def _create_planning_prompt(self, task: TaskRequest) -> str:
        """Create a comprehensive prompt for LLM-based planning."""
        atoms_description = self._format_neural_atoms_for_prompt()

        prompt = f"""You are the central planner for an advanced AI agent with Neural Atom capabilities.

Task Analysis:
- Task ID: {task.task_id}
- Type: {task.task_type.value}
- Description: {task.description}
- Priority: {task.priority}
- Context: {task.context}

Available Neural Atoms:
{atoms_description}

Planning Instructions:
1. Analyze the task requirements carefully
2. Determine if an existing Neural Atom can handle this task
3. If no suitable atom exists, identify what capability is needed
4. Choose the most appropriate action

Response Format (choose one):

NEURAL_ATOM: <atom_name>
PARAMETERS: {{"param": "value"}}
REASONING: <why this atom was selected>

OR

CAPABILITY_GAP: <description of missing capability>
REASONING: <why no existing atom can handle this>

OR

DIRECT_RESPONSE: <direct answer to user>
REASONING: <why no Neural Atom is needed>

Your Response:"""

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
