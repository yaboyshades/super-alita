# src/plugins/ladder_aog_plugin.py

# ruff: noqa: I001  # Allow long lines & skip import sort for legacy compatibility
# mypy: ignore-errors

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Optional

import numpy as np

from src.core.aog import AOGNode, AOGPlan
from src.core.event_bus import EventBus
from src.core.events import BaseEvent
from src.core.neural_atom import NeuralAtom, NeuralStore, create_memory_atom
from src.core.plugin_interface import PluginInterface

"""
A neuro-symbolic reasoning plugin that uses an And-Or Graph (AOG)
to generate and diagnose plans.

This plugin implements the LADDER (Language-driven Algorithmic Decision-making
for Dynamic Environments) approach with And-Or Graph reasoning for:
- Forward planning: Goal -> Actions (LADDER)
- Backward diagnosis: Effect -> Causes (Reverse-LADDER)
- Neuro-symbolic decisions: Semantic attention at OR-nodes
"""

logger = logging.getLogger(__name__)


# Define fallback aemit_safe first, then try overriding via relative import
async def aemit_safe(plugin: PluginInterface, event):  # type: ignore
    try:
        event_type = getattr(event, "event_type", None)
        if event_type:
            if hasattr(event, "model_dump"):
                payload = {
                    k: v for k, v in event.model_dump().items() if k != "event_type"
                }
            else:
                payload = {
                    k: v
                    for k, v in getattr(event, "__dict__", {}).items()
                    if k != "event_type"
                }
            await plugin.emit_event(event_type, **payload)
        else:
            await plugin.emit_event(str(event))
    except Exception:
        logging.getLogger(__name__).warning("Fallback aemit_safe failed", exc_info=True)


try:  # pragma: no cover
    from ._emit import aemit_safe as _real_aemit_safe  # type: ignore

    aemit_safe = _real_aemit_safe
except Exception:
    pass


class MCTSNode:
    """Monte Carlo Tree Search node for planning."""

    def __init__(self, aog_node_id: str = None, parent: "MCTSNode" = None):
        self.aog_node_id = aog_node_id
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self, available_children: list[str]) -> bool:
        """Check if all available children have been expanded."""
        expanded_ids = {child.aog_node_id for child in self.children}
        return set(available_children).issubset(expanded_ids)

    def ucb1_score(self, exploration_weight: float = 1.414) -> float:
        """Upper Confidence Bound score for selection."""
        if self.visits == 0:
            return float("inf")

        if self.parent is None or self.parent.visits == 0:
            return self.value / self.visits

        import math

        exploration = exploration_weight * (
            (2 * math.log(self.parent.visits) / self.visits) ** 0.5
        )
        return (self.value / self.visits) + exploration

    def select_best_child(self) -> Optional["MCTSNode"]:
        """Select child with highest UCB1 score."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.ucb1_score())

    def add_child(self, child: "MCTSNode"):
        """Add child node and set parent reference."""
        child.parent = self
        self.children.append(child)

    def update(self, reward: float):
        """Update node statistics with reward."""
        self.visits += 1
        self.value += reward


class LADDERAOGPlugin(PluginInterface):
    """
    Implements bidirectional planning and diagnosis using an AOG.

    - LADDER (Forward Planning): Traverses the AOG from a high-level goal
      down to a sequence of executable tool calls (LEAF nodes).
    - Reverse-LADDER (Diagnosis): Traverses the AOG backwards from an observed
      effect up to its potential root causes.
    - Neuro-Symbolic Decisions: At OR-nodes, it uses semantic similarity (attention)
      to choose the most relevant path, rather than relying on static rules.
    """

    AOG_TAG = "aog:"  # A prefix to distinguish AOG atoms in the NeuralStore

    def __init__(self):
        super().__init__()
        self._name = "ladder_aog_plugin"
        self._config: dict[str, Any] = {}
        self._parent_map: dict[str, list[str]] = defaultdict(list)
        self._active_plans: dict[str, AOGPlan] = {}
        self._execution_history: list[dict[str, Any]] = []

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    @name.setter
    def name(self, value: str) -> None:  # allow tests to assign
        self._name = value

    # Backwards compatibility alias expected by some tests
    async def setup(
        self, event_bus: EventBus, store: NeuralStore, config: dict[str, Any]
    ):  # type: ignore[override]
        """Initialize the LADDER-AOG plugin."""
        await super().setup(event_bus, store, config)
        self._config = config.get(self.name, {})
        logger.info("LADDER-AOG plugin initialized")

    async def start(self) -> None:
        """Start the plugin and seed initial AOG."""
        await super().start()

        # Subscribe to planning events
        await self.subscribe("planning_request", self._handle_planning_request)
        await self.subscribe("diagnosis_request", self._handle_diagnosis_request)
        await self.subscribe("plan_execution_request", self._handle_execution_request)
        await self.subscribe("goal_received", self._on_goal_received)

        # Seed the NeuralStore with initial AOG
        await self._seed_initial_aog()

        logger.info("LADDER-AOG plugin started with seeded reasoning graph")

    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        await super().shutdown()
        logger.info("LADDER-AOG plugin shut down")

    async def _handle_planning_request(self, event: BaseEvent):
        """Handle planning requests."""
        goal = event.metadata.get("goal")
        context = event.metadata.get("context", {})

        if goal:
            plan = await self.forward_plan(goal, context)

            if plan:
                await self.emit_event(
                    "planning_decision",
                    plan_id=plan.plan_id,
                    decision=f"Generated plan with {len(plan.steps)} steps",
                    plugin_name=self.name,
                    timestamp=datetime.now(UTC).isoformat(),
                )

    async def _handle_diagnosis_request(self, event: BaseEvent):
        """Handle diagnosis requests."""
        effect = event.metadata.get("effect")
        context = event.metadata.get("context", {})

        if effect:
            causes = await self.backward_diagnose(effect, context)

            if causes:
                await self.emit_event(
                    "planning_decision",
                    plan_id=f"diag_{uuid.uuid4().hex[:8]}",
                    decision=f"Found {len(causes)} potential causes",
                    plugin_name=self.name,
                    timestamp=datetime.now(UTC).isoformat(),
                )

    async def _handle_execution_request(self, event: BaseEvent):
        """Handle plan execution requests."""
        plan_id = event.metadata.get("plan_id")

        if plan_id and plan_id in self._active_plans:
            success = await self.execute_plan(plan_id)
            logger.info(
                f"Plan execution {'succeeded' if success else 'failed'}: {plan_id}"
            )

    async def _on_goal_received(self, event):
        """Handle goal received events from conversation plugin."""
        try:
            goal_description = getattr(event, "goal", "") or getattr(
                event, "goal_description", ""
            )
            session_id = getattr(event, "session_id", "unknown")

            if not goal_description:
                logger.warning("Received goal event with no goal description")
                return

            logger.info(f"Planning for goal: {goal_description}")

            # Check if this requires dynamic tool generation
            plan = None
            if self._requires_dynamic_tool(goal_description):
                plan = await self._plan_with_dynamic_tools(goal_description, session_id)
            else:
                # Generate plan using traditional LADDER-AOG
                plan = await self.forward_plan(
                    goal_description, context={"session_id": session_id}
                )
            if plan:
                # Store the plan
                plan_id = f"plan_{uuid.uuid4().hex[:8]}"
                self._active_plans[plan_id] = plan

                # Emit plan ready event
                await self.emit_event(
                    "plan_ready",
                    plan_id=plan_id,
                    plan=plan.to_dict(),
                    goal_description=goal_description,
                    session_id=session_id,
                    metadata={"source": "goal_received"},
                )

                logger.info(f"Plan generated and ready: {plan_id}")
            else:
                logger.warning(f"Failed to generate plan for goal: {goal_description}")

                # Emit plan failure event
                await self.emit_event(
                    "plan_failed",
                    goal_description=goal_description,
                    session_id=session_id,
                    reason="No suitable goal node found or planning failed",
                )

        except Exception as e:
            logger.error(f"Error handling goal received event: {e}")
            import traceback

            traceback.print_exc()

    async def forward_plan(
        self, goal_description: str, context: dict[str, Any] = None
    ) -> AOGPlan | None:
        """
        Generate a forward plan from goal to executable actions.

        This is the core LADDER algorithm with neuro-symbolic decision making.
        """
        context = context or {}

        # Find the goal node using semantic search
        goal_atom = await self._find_goal_node(goal_description)
        if not goal_atom:
            logger.warning(f"No suitable goal node found for: {goal_description}")
            return None

        # Create plan structure
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        plan = AOGPlan(
            plan_id=plan_id,
            goal_node_id=goal_atom.key,
            steps=[],
            path_taken=[],
            created_at=datetime.now(UTC).isoformat(),
        )

        # Perform forward traversal with neuro-symbolic decisions
        success = await self._traverse_forward(goal_atom, plan, context)

        if success and plan.steps:
            # Calculate plan metrics
            plan.estimated_cost = await self._calculate_plan_cost(plan)
            plan.estimated_probability = await self._calculate_plan_probability(plan)

            # Store active plan
            self._active_plans[plan_id] = plan

            # Strengthen neural pathways (Hebbian learning)
            if plan.path_taken:
                await self._strengthen_pathways(plan.path_taken)

            logger.info(f"Generated plan {plan_id} with {len(plan.steps)} steps")
            return plan

        logger.warning(f"Failed to generate valid plan for goal: {goal_description}")
        return None

    async def _traverse_forward(
        self, current_atom: NeuralAtom, plan: AOGPlan, context: dict[str, Any]
    ) -> bool:
        """
        Traverse the AOG forward from current node to LEAF nodes.

        This implements the neuro-symbolic decision making at OR-nodes.
        """
        if not isinstance(current_atom.value, AOGNode):
            return False

        node: AOGNode = current_atom.value
        plan.path_taken.append(current_atom.key)

        # Base case: LEAF node
        if node.node_type == "LEAF":
            plan.steps.append(current_atom.key)
            return True

        # Recursive case: AND/OR nodes
        if node.node_type == "AND":
            # AND node: all children must succeed
            for child_key in node.children:
                child_atom = self.store.get(child_key)
                if not child_atom:
                    logger.warning(f"Missing child node: {child_key}")
                    return False

                if not await self._traverse_forward(child_atom, plan, context):
                    return False

            return True

        if node.node_type == "OR":
            # OR node: use neuro-symbolic attention to choose best child
            best_child = await self._choose_or_child(current_atom, node, context)

            if best_child:
                return await self._traverse_forward(best_child, plan, context)
            logger.warning(f"No suitable child found for OR node: {node.node_id}")
            return False

        return False

    async def _choose_or_child(
        self, parent_atom: NeuralAtom, node: AOGNode, context: dict[str, Any]
    ) -> NeuralAtom | None:
        """
        Choose the best child at an OR node using the Pilot-Mech model.

        CRITICAL REFACTOR: This now delegates strategic decisions to the LLM (Pilot)
        instead of using simple semantic similarity. The framework (Mech) gathers
        context and presents options; the Pilot makes the strategic choice.
        """
        if not node.children:
            return None

        # Get child atoms
        child_atoms = []
        for child_key in node.children:
            child_atom = self.store.get(child_key)
            if child_atom:
                child_atoms.append(child_atom)

        if not child_atoms:
            return None

        # PILOT-MECH MODEL: Ask the LLM to make the strategic choice
        best_child = await self._ask_pilot_for_or_choice(
            parent_atom, node, child_atoms, context
        )

        if best_child:
            return best_child

        # Fallback: use attention mechanism if Pilot fails
        logger.warning("Pilot decision failed, falling back to attention mechanism")
        if hasattr(self.store, "attention") and parent_atom.vector is not None:
            child_keys = [atom.key for atom in child_atoms]
            attention_results = await self.store.attention(
                query_vector=parent_atom.vector,
                filter_keys=child_keys,
                top_k=len(child_keys),
            )

            if attention_results:
                best_key, best_score = attention_results[0]
                best_child = self.store.get(best_key)

                if best_child:
                    logger.info(
                        f"Fallback attention selected OR-path '{best_key}' with similarity {best_score:.4f}"
                    )
                    return best_child

        # Final fallback: choose first available child
        logger.warning(
            "All decision mechanisms failed, using first child as final fallback"
        )
        return child_atoms[0]

    async def _ask_pilot_for_or_choice(
        self,
        parent_atom: NeuralAtom,
        node: AOGNode,
        child_atoms: list[NeuralAtom],
        context: dict[str, Any],
    ) -> NeuralAtom | None:
        """
        Consult the LLM (Pilot) for strategic OR-node decisions.

        This is the core of the Pilot-Mech model: the framework gathers context
        and presents options clearly, while the LLM makes the strategic choice.
        """
        try:
            # 1. Gather rich context through neural attention
            goal_description = context.get("goal_description", "Unknown goal")
            context_query = f"Planning for goal: {goal_description}. At decision point: {node.description}"

            # Get relevant memories for this decision
            if hasattr(self, "_memory") and self._memory:
                try:
                    query_embedding = await asyncio.wait_for(
                        self._memory.embed_text([context_query]), timeout=15.0
                    )
                    relevant_memories = await self.store.attention(
                        query_embedding[0], top_k=3
                    )
                except TimeoutError:
                    logger.warning(
                        "Memory query timed out, proceeding without relevant memories"
                    )
                    relevant_memories = []
                except Exception as e:
                    logger.warning(f"Memory query failed: {e}")
                    relevant_memories = []
            else:
                relevant_memories = []

            # 2. Build structured options for the Pilot
            options_data = []
            for i, child_atom in enumerate(child_atoms):
                if isinstance(child_atom.value, AOGNode):
                    child_node = child_atom.value
                    option_info = {
                        "index": i + 1,
                        "key": child_atom.key,
                        "description": child_node.description,
                        "node_type": child_node.node_type,
                        "fitness": child_atom.fitness,
                        "metadata": getattr(child_node, "metadata", {}),
                    }
                    options_data.append(option_info)

            # 3. Construct the rich prompt for strategic decision-making
            prompt = self._build_pilot_decision_prompt(
                goal_description=goal_description,
                decision_point=node.description,
                options=options_data,
                relevant_memories=relevant_memories,
                context=context,
            )

            # 4. Consult the Pilot with timeout protection (Protocol 2)
            try:
                # Use lower temperature for strategic decisions (precision over creativity)
                llm_response = await asyncio.wait_for(
                    self._generate_llm_response(
                        prompt, temperature=0.3, max_tokens=500
                    ),
                    timeout=45.0,
                )

                # 5. Parse the Pilot's strategic choice
                choice_data = await self._parse_pilot_choice(llm_response, child_atoms)

                if choice_data:
                    chosen_atom, reasoning = choice_data

                    # Record the strategic decision for genealogy
                    # Decision metadata implicitly captured via planning event

                    # Emit planning decision event
                    await self.emit_event(
                        "planning_decision",
                        node_id=node.node_id,
                        node_type="OR",
                        decision_type="pilot_choice",
                        options=[atom.key for atom in child_atoms],
                        chosen_option=chosen_atom.key,
                        confidence=0.8,
                        reasoning=reasoning,
                        plugin_name=self.name,
                        timestamp=datetime.now(UTC).isoformat(),
                    )

                    logger.info(
                        f"Pilot chose '{chosen_atom.key}' for OR-node '{node.node_id}': {reasoning[:100]}"
                    )
                    return chosen_atom

            except TimeoutError:
                logger.warning(
                    f"Pilot consultation timed out for OR-node '{node.node_id}'"
                )
                return None
            except Exception as e:
                logger.error(f"Error during Pilot consultation: {e}")
                return None

        except Exception as e:
            logger.error(f"Critical error in _ask_pilot_for_or_choice: {e}")
            return None

        return None

    def _build_pilot_decision_prompt(
        self,
        goal_description: str,
        decision_point: str,
        options: list[dict],
        relevant_memories: list,
        context: dict[str, Any],
    ) -> str:
        """
        Construct the Cognitive Contract prompt for AOG Strategic Path Selection.

        This is the master template that ensures the Pilot (LLM) receives
        structured, comprehensive context for strategic planning decisions.
        """
        import time

        # Generate correlation ID for this decision
        correlation_id = f"aog-{int(time.time())}"

        # Format memories with proper truncation
        memory_section = ""
        if relevant_memories:
            memory_section = "\nrelevant_memories:"
            for key, score in relevant_memories[:5]:
                atom = self.store.get(key)
                if atom:
                    content_summary = (
                        str(atom.value)[:150] + "..."
                        if len(str(atom.value)) > 150
                        else str(atom.value)
                    )
                    memory_section += f'\n  - key: "{key}"'
                    memory_section += f"\n    similarity_score: {score:.4f}"
                    memory_section += f'\n    content_summary: "{content_summary}"'
        else:
            memory_section = '\nrelevant_memories:\n  - "No relevant memories found."'

        # Format available options
        options_section = "\navailable_options:"
        for i, option in enumerate(options):
            options_section += f"\n  - index: {i + 1}"
            options_section += f'\n    description: "{option["description"]}"'
            options_section += f'\n    node_type: "{option["node_type"]}"'
            options_section += f"\n    fitness: {option['fitness']:.3f}"

        # Build the Cognitive Contract prompt
        prompt = f"""# COGNITIVE CONTRACT: AOG Strategic Path Selection
# AGENT: Super Alita
# CORRELATION ID: {correlation_id}
# TIMESTAMP: {datetime.now(UTC).isoformat()}

# --- DIRECTIVES AND BINDING PROTOCOLS ---
# ROLE: You are the Planning Core of the Super Alita agent. Your function is to make optimal strategic choices during plan generation.
# BINDING DIRECTIVE 1: You MUST select the single best option from the provided list to achieve the overall goal.
# BINDING DIRECTIVE 2: Your response MUST be a single, valid JSON object containing only the index of your chosen option. No other text is permitted.
# BINDING DIRECTIVE 3: Your reasoning must demonstrate strategic analysis of the options in context of the goal.

# --- SECTION 1: MISSION CONTEXT ---
mission_context:
  overall_goal: "{goal_description}"
  current_decision_point: "{decision_point}"

# --- SECTION 2: RELEVANT LONG-TERM MEMORY (Neural Context) ---
# Results from NeuralStore.attention() query.{memory_section}

# --- SECTION 3: AVAILABLE STRATEGIC OPTIONS ---
# Each option represents a potential path forward in the plan.{options_section}

# --- SECTION 4: TASK AND REQUIRED OUTPUT SCHEMA ---
task:
  description: "Based on the overall goal and your relevant memories, analyze the available options. Determine which single option represents the most logical and effective path forward. Your response must indicate the index of that option."
  required_output_schema: {{
    "title": "PlannerChoice",
    "type": "object",
    "properties": {{
      "best_option_index": {{"type": "integer", "minimum": 1, "maximum": {len(options)}}},
      "reasoning": {{"type": "string"}}
    }},
    "required": ["best_option_index", "reasoning"]
  }}

# --- END OF CONTRACT ---"""

        return prompt

    def _build_aog_cognitive_contract(
        self,
        goal_description: str,
        decision_point: str,
        options: list[dict],
        relevant_memories: list,
        context: dict[str, Any],
    ) -> str:
        """
        Alias for _build_pilot_decision_prompt to match Cognitive Contract naming convention.
        """
        return self._build_pilot_decision_prompt(
            goal_description, decision_point, options, relevant_memories, context
        )

    async def _generate_llm_response(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 500
    ) -> str:
        """
        Generate LLM response with proper error handling.
        In production, this would interface with the actual LLM service.
        """
        # TODO: Replace with actual LLM client integration
        # For now, return a mock response for testing
        import json

        mock_response = {
            "chosen_option": 1,
            "reasoning": "Selected first option based on strategic analysis",
            "confidence": 0.8,
        }
        return json.dumps(mock_response)

    async def _parse_pilot_choice(
        self, llm_response: str, child_atoms: list[NeuralAtom]
    ) -> tuple[NeuralAtom, str] | None:
        """
        Parse the Pilot's response from the Cognitive Contract.
        Expects the new PlannerChoice schema with best_option_index.
        """
        try:
            import json

            choice_data = json.loads(llm_response)

            # Handle both old and new response formats for backward compatibility
            chosen_option = choice_data.get("best_option_index") or choice_data.get(
                "chosen_option"
            )
            reasoning = choice_data.get("reasoning", "No reasoning provided")

            if isinstance(chosen_option, int) and 1 <= chosen_option <= len(
                child_atoms
            ):
                chosen_atom = child_atoms[chosen_option - 1]
                return chosen_atom, reasoning
            logger.error(
                f"Invalid option choice: {chosen_option} (valid range: 1-{len(child_atoms)})"
            )
            return None

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse Pilot response: {e}")
            return None

    async def backward_diagnose(
        self, effect_description: str, context: dict[str, Any] = None
    ) -> list[str] | None:
        """
        Perform backward diagnosis from effect to potential causes.

        This is the Reverse-LADDER algorithm.
        """
        context = context or {}

        # Find the effect node
        effect_atom = await self._find_effect_node(effect_description)
        if not effect_atom:
            logger.warning(f"No suitable effect node found for: {effect_description}")
            return None

        # Traverse backward to find potential causes
        causes = []
        visited = set()
        queue = [effect_atom.key]

        while queue:
            current_key = queue.pop(0)
            if current_key in visited:
                continue

            visited.add(current_key)

            # Find parent nodes
            parent_keys = self._parent_map.get(current_key, [])
            for parent_key in parent_keys:
                parent_atom = self.store.get(parent_key)
                if parent_atom and isinstance(parent_atom.value, AOGNode):
                    parent_node: AOGNode = parent_atom.value

                    # If parent is a LEAF node, it's a potential cause
                    if parent_node.node_type == "LEAF":
                        causes.append(parent_key)
                    else:
                        queue.append(parent_key)

        if causes:
            logger.info(
                f"Found {len(causes)} potential causes for: {effect_description}"
            )

        return causes if causes else None

    async def _calculate_plan_cost(self, plan: AOGPlan) -> float:
        """Calculate estimated cost for a plan."""
        total_cost = 0.0

        for step_key in plan.steps:
            atom = self.store.get(step_key)
            if atom and isinstance(atom.value, AOGNode):
                node = atom.value
                total_cost += node.cost_estimate

        return total_cost

    async def _calculate_plan_probability(self, plan: AOGPlan) -> float:
        """Calculate estimated success probability for a plan."""
        if not plan.steps:
            return 0.0

        # Multiply individual probabilities (assuming independence)
        total_prob = 1.0

        for step_key in plan.steps:
            atom = self.store.get(step_key)
            if atom and isinstance(atom.value, AOGNode):
                node = atom.value
                total_prob *= node.get_success_rate()
        return total_prob

    def _extract_causal_factors(self, plan: AOGPlan | None) -> list[str]:
        """Extract causal factors from a plan."""
        if not plan:
            return []

        factors = []
        for step_key in plan.steps:
            atom = self.store.get(step_key)
            if atom and hasattr(atom, "value") and hasattr(atom.value, "description"):
                factors.append(atom.value.description)

        return factors

    async def _handle_diagnosis_request(self, event: BaseEvent):
        """Handle causal diagnosis request."""
        # Implement abductive reasoning for causal diagnosis
        logger.info("Diagnosis request received - implementing abductive reasoning")
        logger.info("Generating causal explanation for event: %s", event)
        result = {"factors": []}
        await self.emit_event("diagnosis_complete", **result)

    async def _handle_aog_update(self, event: BaseEvent):
        """Handle AOG structure updates."""
        logger.info("AOG update request received")
        payload = getattr(event, "payload", {})
        node_id = payload.get("node_id")
        description = payload.get("description", "")
        if node_id and node_id in self.aog_graphs:
            self.aog_graphs[node_id].description = description

    def _save_aog_graphs(self):
        """Save AOG graphs to persistent storage."""
        if not self.config.get("save_aog_graphs"):
            return
        path = self.config.get("graph_path", "data/aog_graphs.json")
        try:
            data = {
                k: v.model_dump() if hasattr(v, "model_dump") else v.__dict__
                for k, v in self.aog_graphs.items()
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("Saved AOG graphs to %s", path)
        except Exception as e:
            logger.error("Failed to save AOG graphs: %s", e)

    async def _strengthen_pathways(self, path_taken: list[str]):
        """Strengthen neural pathways using Hebbian learning."""
        if len(path_taken) < 2:
            return

        # Apply Hebbian update to strengthen connections
        if hasattr(self.store, "hebbian_update"):
            await self.store.hebbian_update(path_taken)

        logger.debug(f"Strengthened pathways for {len(path_taken)} nodes")

    async def execute_plan(self, plan_id: str) -> bool:
        """Execute a plan step by step."""
        if plan_id not in self._active_plans:
            logger.error(f"Plan not found: {plan_id}")
            return False

        plan = self._active_plans[plan_id]
        plan.status = "executing"

        success = True
        start_time = datetime.now(UTC)

        for i, step_key in enumerate(plan.steps):
            step_atom = self.store.get(step_key)
            if not step_atom or not isinstance(step_atom.value, AOGNode):
                logger.error(f"Invalid step atom: {step_key}")
                success = False
                break

            node = step_atom.value
            plan.current_step = i

            # Execute the step (placeholder - would integrate with tool execution)
            step_success = await self._execute_step(node)

            # Update execution statistics
            timestamp = datetime.now(UTC).isoformat()
            duration = (datetime.now(UTC) - start_time).total_seconds()

            plan.add_execution_log(i, step_key, step_success, duration)
            node.update_execution_stats(step_success, timestamp)

            if not step_success:
                success = False
                logger.warning(f"Step failed: {node.node_id}")
                break

            logger.info(f"Step completed: {node.node_id}")

        # Update final plan status
        plan.status = "completed" if success else "failed"
        plan.current_step = len(plan.steps)

        # Record execution history
        self._execution_history.append(
            {
                "plan_id": plan_id,
                "success": success,
                "duration": (datetime.now(UTC) - start_time).total_seconds(),
                "steps_completed": plan.current_step,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        logger.info(f"Plan execution {'completed' if success else 'failed'}: {plan_id}")
        return success

    async def _execute_step(self, node: AOGNode) -> bool:
        """Execute a single step (LEAF node)."""
        if not node.is_terminal() or not node.tool_template:
            logger.error(f"Cannot execute non-terminal node: {node.node_id}")
            return False

        # Placeholder for actual tool execution
        # In production, this would integrate with the tool execution system
        logger.info(f"Executing tool: {node.tool_template.get('tool', 'unknown')}")

        # Simulate execution with success rate
        import hashlib

        # Use node ID to create deterministic "random" success
        hash_val = int(hashlib.md5(node.node_id.encode()).hexdigest()[:8], 16)
        success_threshold = node.get_success_rate()
        simulated_success = (hash_val % 100) / 100.0 < success_threshold

        return simulated_success

    async def _seed_initial_aog(self):
        """Creates and embeds a basic AOG for code bounty hunting."""
        nodes_data = [
            {
                "id": "solve_bounty",
                "desc": "Solve a software code bounty",
                "type": "OR",
                "children": ["analyze_bounty", "exploit_bounty"],
            },
            {
                "id": "analyze_bounty",
                "desc": "Analyze the bounty to understand requirements",
                "type": "AND",
                "children": [
                    "find_bounty_url",
                    "read_bounty_page",
                    "summarize_requirements",
                ],
            },
            {
                "id": "exploit_bounty",
                "desc": "Develop and submit an exploit for the bounty",
                "type": "AND",
                "children": ["generate_code", "test_code", "submit_solution"],
            },
            {
                "id": "find_bounty_url",
                "desc": "Use web search to find the bounty's primary URL",
                "type": "LEAF",
                "template": {"tool": "web_search"},
            },
            {
                "id": "read_bounty_page",
                "desc": "Browse the bounty webpage to extract text",
                "type": "LEAF",
                "template": {"tool": "browse_website"},
            },
            {
                "id": "summarize_requirements",
                "desc": "Summarize the extracted text to define clear requirements",
                "type": "LEAF",
                "template": {"tool": "summarize_text"},
            },
            {
                "id": "generate_code",
                "desc": "Generate code that addresses the bounty requirements",
                "type": "LEAF",
                "template": {"tool": "code_generation"},
            },
            {
                "id": "test_code",
                "desc": "Execute tests to validate the generated code",
                "type": "LEAF",
                "template": {"tool": "run_tests"},
            },
            {
                "id": "submit_solution",
                "desc": "Submit the solution to the bounty platform",
                "type": "LEAF",
                "template": {"tool": "submit_pr"},
            },
        ]

        # Create embeddings for descriptions using semantic memory if available
        descriptions = [n["desc"] for n in nodes_data]

        # Try to use semantic memory for batch embedding, fallback to simple method
        if hasattr(self, "_semantic_memory_plugin") and self._semantic_memory_plugin:
            try:
                vectors = await self._semantic_memory_plugin.embed_text(descriptions)
            except Exception as e:
                logger.warning(f"Failed to use semantic memory for embedding: {e}")
                vectors = [await self._create_embedding(desc) for desc in descriptions]
        else:
            vectors = [await self._create_embedding(desc) for desc in descriptions]

        for i, data in enumerate(nodes_data):
            key = f"{self.AOG_TAG}{data['id']}"
            node = AOGNode(
                node_id=data["id"],
                description=data["desc"],
                type=data["type"],
                children=[f"{self.AOG_TAG}{c}" for c in data.get("children", [])],
                tool_template=data.get("template"),
                priority=1.0,
                cost_estimate=1.0,
                success_probability=0.8,
            )
            atom = NeuralAtom(key=key, default_value=node, vector=vectors[i])
            self.store.register(atom)

            # Build the parent map for reverse diagnosis
            for child_key in node.children:
                self._parent_map[child_key].append(key)

        logger.info(
            f"Seeded and embedded {len(nodes_data)} AOG nodes for bounty hunting into the NeuralStore."
        )

    async def plan(self, goal_atom_key: str) -> list[str] | None:
        """
        Generates a plan by performing a forward pass over the AOG.

        This is the core LADDER (forward planning) algorithm that traverses
        the And-Or Graph from a high-level goal down to executable actions.
        """
        goal_atom = self.store.get(goal_atom_key)
        if not goal_atom:
            logger.error(f"Goal atom '{goal_atom_key}' not found for planning.")
            return None

        plan_steps: list[str] = []
        queue: list[str] = [goal_atom_key]
        visited: set[str] = set()
        path_taken: set[str] = {goal_atom_key}

        while queue:
            current_key = queue.pop(0)
            if current_key in visited:
                continue
            visited.add(current_key)

            node_atom = self.store.get(current_key)
            if not node_atom or not isinstance(node_atom.value, AOGNode):
                logger.warning(f"AOG traversal found a non-AOGNode atom: {current_key}")
                continue

            node: AOGNode = node_atom.value

            if node.node_type == "LEAF":
                plan_steps.append(current_key)
                continue

            if not node.children:
                continue

            if node.node_type == "AND":
                # For AND nodes, add all children to the queue
                queue.extend(node.children)
                path_taken.update(node.children)
            elif node.node_type == "OR":
                # For OR nodes, use attention to pick the best child
                child_atoms = [self.store.get(k) for k in node.children]
                valid_children = [a for a in child_atoms if a is not None]

                if not valid_children:
                    continue

                # Use the goal's vector to find the most relevant path
                attention_results = await self.store.attention(
                    goal_atom.vector, top_k=len(valid_children)
                )

                # Find the best child among the valid children that is also in the top attention results
                best_child_key = None
                for key, score in attention_results:
                    if key in node.children:
                        best_child_key = key
                        logger.info(
                            f"Attention selected OR-path '{key}' with similarity {score:.4f}"
                        )
                        break

                if best_child_key:
                    queue.append(best_child_key)
                    path_taken.add(best_child_key)

        if plan_steps:
            # Strengthen the synaptic connections in the successful path
            self.store.hebbian_update(list(path_taken))

            # Emit planning decision event with proper schema
            decision_event = {
                "plan_id": f"plan_{uuid.uuid4().hex[:8]}",
                "decision": f"Generated plan with {len(plan_steps)} steps.",
                "confidence_score": 0.9,  # Could be calculated based on attention scores
                "causal_factors": list(path_taken),
                "timestamp": datetime.now(UTC).isoformat(),
                "plugin_name": self.name,
            }
            await self.emit_event("planning_decision", **decision_event)

        return plan_steps

    async def _handle_planning_request(self, event):
        """Handle planning request events."""
        goal = event.metadata.get("goal")
        if goal:
            await self.plan(goal_atom_key=goal)

    async def get_statistics(self) -> dict[str, Any]:
        """Get plugin statistics."""
        return {
            "active_plans": len(self._active_plans),
            "total_executions": len(self._execution_history),
            "successful_executions": sum(
                1 for h in self._execution_history if h["success"]
            ),
            "aog_nodes": len(
                [k for k in self.store.get_all_keys() if k.startswith(self.AOG_TAG)]
            ),
            "recent_executions": (
                self._execution_history[-5:] if self._execution_history else []
            ),
        }

    async def health_check(self) -> dict[str, Any]:
        """Check plugin health."""
        aog_nodes = [k for k in self.store.get_all_keys() if k.startswith(self.AOG_TAG)]

        issues = []
        if not aog_nodes:
            issues.append("No AOG nodes found in neural store")

        if len(self._active_plans) > 10:
            issues.append(f"Many active plans: {len(self._active_plans)}")

        return {
            "status": "healthy" if not issues else "warning",
            "aog_nodes_count": len(aog_nodes),
            "active_plans_count": len(self._active_plans),
            "issues": issues,
        }

    async def _find_goal_node(self, goal_description: str) -> NeuralAtom | None:
        """Find the most suitable goal node using semantic search."""
        # Create embedding for goal description
        goal_embedding = await self._create_embedding(goal_description)

        if goal_embedding is not None:
            # Use attention to find most similar AOG node
            aog_keys = [
                key for key in self.store.get_all_keys() if key.startswith(self.AOG_TAG)
            ]

            if hasattr(self.store, "attention"):
                attention_results = await self.store.attention(
                    query_vector=goal_embedding, filter_keys=aog_keys, top_k=1
                )

                if attention_results:
                    best_key, score = attention_results[0]
                    if score > 0.7:  # Threshold for similarity
                        return self.store.get(best_key)

        # Fallback: look for root nodes
        for key in self.store.get_all_keys():
            if key.startswith(self.AOG_TAG):
                atom = self.store.get(key)
                if atom and isinstance(atom.value, AOGNode):
                    node = atom.value
                    # Check if this looks like a root goal node
                    if not self._parent_map.get(key) and node.node_type in [
                        "AND",
                        "OR",
                    ]:
                        return atom

        return None

    async def _find_effect_node(self, effect_description: str) -> NeuralAtom | None:
        """Find the most suitable effect node using semantic search."""
        # Similar to _find_goal_node but for effects
        effect_embedding = await self._create_embedding(effect_description)

        if effect_embedding is not None and hasattr(self.store, "attention"):
            aog_keys = [
                key for key in self.store.get_all_keys() if key.startswith(self.AOG_TAG)
            ]

            attention_results = await self.store.attention(
                query_vector=effect_embedding, filter_keys=aog_keys, top_k=1
            )

            if attention_results:
                best_key, score = attention_results[0]
                if score > 0.6:  # Lower threshold for effects
                    return self.store.get(best_key)

        return None

    def _requires_dynamic_tool(self, goal_description: str) -> bool:
        """Check if the goal requires dynamic tool generation."""
        keywords = [
            "build",
            "create",
            "generate",
            "quantum",
            "circuit",
            "analyze function",
            "plot",
            "mathematical",
            "process data",
            "atom-challenge",
            "dynamic",
        ]
        return any(keyword in goal_description.lower() for keyword in keywords)

    async def _plan_with_dynamic_tools(
        self, goal_description: str, session_id: str
    ) -> AOGPlan | None:
        """Generate a secure plan using dynamically created tools with full audit trail."""
        try:
            logger.info(f"🔧 Secure dynamic tool planning for: {goal_description}")

            # Import secure dynamic tool system
            from src.core.dynamic_tool_generator import get_dynamic_generator
            from src.core.secure_executor import get_secure_executor
            from src.core.tool_memory import get_memory_manager

            generator = get_dynamic_generator()
            _ = get_secure_executor()  # side-effect initialization
            _ = get_memory_manager(self.store)  # side-effect initialization

            # Extract parameters from goal with security validation
            params = generator._extract_parameters(goal_description)

            # Determine tool type
            tool_type = self._determine_secure_tool_type(goal_description)

            logger.info(
                f"Generating secure dynamic tool: {tool_type} with params: {params}"
            )

            # Generate secure dynamic tool with audit trail
            code, tool_info = generator.generate_dynamic_tool(
                tool_type=tool_type,
                params=params,
                user_id=session_id,
                context_id=f"planning_{session_id}",
            )

            # Create secure AOG plan
            plan = AOGPlan(
                plan_id=f"secure_dynamic_plan_{uuid.uuid4().hex[:8]}",
                goal_description=goal_description,
                execution_steps=[],
                estimated_cost=1.0,
                estimated_probability=0.9,
                metadata={
                    "planning_method": "secure_dynamic_tool_generation",
                    "tool_name": tool_info["tool_name"],
                    "memory_id": tool_info["memory_id"],
                    "session_id": session_id,
                    "security_validated": True,
                    "audit_trail": True,
                    "provenance_tracked": True,
                    "tool_type": tool_type,
                },
            )

            # Add secure execution step
            plan.execution_steps.append(
                {
                    "step_id": f"secure_dynamic_step_{uuid.uuid4().hex[:8]}",
                    "action": "execute_secure_dynamic_tool",
                    "tool_name": tool_info["tool_name"],
                    "tool_code": code,
                    "parameters": params,
                    "memory_id": tool_info["memory_id"],
                    "description": f"Securely execute {tool_info['tool_name']}",
                    "estimated_duration": 5.0,
                    "security_audit_id": tool_info.get("test_result", {}).get(
                        "audit_id"
                    ),
                    "validation_passed": tool_info.get("test_result", {}).get("status")
                    == "success",
                }
            )

            # Store additional metadata in neural store
            await self._store_secure_dynamic_tool_metadata(
                tool_info, goal_description, session_id
            )

            logger.info(
                f"✅ Secure dynamic plan created with tool: {tool_info['tool_name']}"
            )
            return plan

        except Exception as e:
            logger.error(f"Secure dynamic tool planning failed: {e}")
            # Return error plan instead of None for better error handling
            return AOGPlan(
                plan_id=f"error_plan_{uuid.uuid4().hex[:8]}",
                goal_description=goal_description,
                execution_steps=[
                    {
                        "step_id": "error_step",
                        "action": "respond_with_error",
                        "message": f"Could not generate secure dynamic tool: {e!s}",
                        "description": "Error response",
                    }
                ],
                estimated_cost=0.1,
                estimated_probability=1.0,
                metadata={
                    "planning_method": "error_fallback",
                    "error": str(e),
                    "session_id": session_id,
                },
            )

    def _determine_secure_tool_type(self, goal_description: str) -> str:
        """Determine tool type from goal description with security considerations"""
        goal_lower = goal_description.lower()

        # Quantum computing
        if any(
            word in goal_lower
            for word in ["quantum", "circuit", "qubit", "gate", "superposition"]
        ):
            return "quantum_circuit"

        # Data analysis
        if any(
            word in goal_lower
            for word in ["data", "analyze", "statistics", "chart", "graph", "dataset"]
        ):
            return "data_analysis"

        # Mathematical computation
        if any(
            word in goal_lower
            for word in [
                "math",
                "calculate",
                "compute",
                "algorithm",
                "formula",
                "equation",
            ]
        ):
            return "math_computation"

        # Text processing
        if any(
            word in goal_lower
            for word in ["text", "language", "process", "nlp", "parse", "extract"]
        ):
            return "text_processing"

        # Simulation
        if any(
            word in goal_lower
            for word in ["simulate", "model", "random", "walk", "monte", "carlo"]
        ):
            return "simulation"

        # Default to generic for security
        return "generic"

    async def _store_secure_dynamic_tool_metadata(
        self, tool_info: dict[str, Any], goal_description: str, session_id: str
    ) -> None:
        """Store secure dynamic tool metadata in neural store for audit trail"""
        try:
            # Create embedding for the tool metadata
            tool_text = (
                f"Dynamic tool: {tool_info['tool_name']} for goal: {goal_description}"
            )
            embedding = await self._create_embedding(tool_text)

            # Create metadata atom
            metadata_atom = create_memory_atom(
                memory_key=f"secure_tool_metadata_{tool_info['memory_id']}",
                content={
                    "type": "secure_dynamic_tool_metadata",
                    "tool_name": tool_info["tool_name"],
                    "memory_id": tool_info["memory_id"],
                    "goal_description": goal_description,
                    "session_id": session_id,
                    "test_result": tool_info.get("test_result"),
                    "params_schema": tool_info.get("params_schema"),
                    "created_at": datetime.now(UTC).isoformat(),
                    "security_validated": True,
                    "audit_trail_available": True,
                },
                memory_type="secure_tool_metadata",
                embedding_vector=embedding,
                confidence=1.0,
            )

            # Add security-specific lineage metadata
            metadata_atom.lineage_metadata.update(
                {
                    "security_level": "validated",
                    "audit_trail": True,
                    "provenance_tracked": True,
                    "planning_context": f"secure_dynamic_{session_id}",
                    "generated_at": datetime.now(UTC).isoformat(),
                }
            )

            # Register with store
            self.store.register(metadata_atom)
            logger.info(
                f"Stored secure dynamic tool metadata: {tool_info['tool_name']}"
            )

        except Exception as e:
            logger.error(f"Failed to store secure tool metadata: {e}")

    async def _create_embedding(self, text: str) -> np.ndarray | None:
        """Create embedding for text (fallback implementation)."""
        from src.core.config import EMBEDDING_DIM

        # In production, this would use the semantic memory plugin
        # For now, create a simple hash-based embedding
        words = text.lower().split()
        embedding = np.zeros(EMBEDDING_DIM)  # Match NeuralAtom expected dimensions

        for i, word in enumerate(words):
            hash_val = hash(word) % EMBEDDING_DIM  # Use EMBEDDING_DIM modulo
            embedding[hash_val] += 1.0 / (i + 1)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm

        return embedding
