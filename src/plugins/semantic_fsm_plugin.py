"""
Semantic FSM Plugin for Super Alita.
Implements embedding-based state transitions for flexible workflow management.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
from enum import Enum

from sentence_transformers import SentenceTransformer

from ..core.plugin_interface import PluginInterface
from ..core.neural_atom import NeuralAtom


class TransitionType(Enum):
    """Types of state transitions."""
    SEMANTIC = "semantic"      # Based on embedding similarity
    EXPLICIT = "explicit"      # Explicit trigger match
    TEMPORAL = "temporal"      # Time-based transition
    CONDITIONAL = "conditional" # Condition-based transition


@dataclass
class FSMState:
    """State in the semantic finite state machine."""
    
    name: str
    description: str
    embedding: Optional[np.ndarray] = None
    entry_actions: List[str] = field(default_factory=list)
    exit_actions: List[str] = field(default_factory=list)
    timeout_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_final: bool = False
    
    def __post_init__(self):
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)


@dataclass
class FSMTransition:
    """Transition between FSM states."""
    
    from_state: str
    to_state: str
    trigger: str
    transition_type: TransitionType
    embedding: Optional[np.ndarray] = None
    similarity_threshold: float = 0.7
    condition_func: Optional[Callable] = None
    priority: int = 0  # Higher priority transitions checked first
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)


class SemanticFSM:
    """
    Semantic Finite State Machine with embedding-based transitions.
    
    Allows flexible state transitions based on semantic similarity
    rather than just exact trigger matching.
    """
    
    def __init__(self, name: str, encoder_model: SentenceTransformer):
        self.name = name
        self.encoder = encoder_model
        self.states: Dict[str, FSMState] = {}
        self.transitions: List[FSMTransition] = []
        self.current_state: Optional[str] = None
        self.state_history: List[Tuple[str, datetime, str]] = []  # (state, timestamp, trigger)
        self.context: Dict[str, Any] = {}
        self.start_time: Optional[datetime] = None
        self.state_enter_time: Optional[datetime] = None
        self.is_running = False
    
    def add_state(
        self,
        name: str,
        description: str,
        entry_actions: List[str] = None,
        exit_actions: List[str] = None,
        timeout_seconds: Optional[float] = None,
        is_final: bool = False,
        metadata: Dict[str, Any] = None
    ) -> FSMState:
        """Add a state to the FSM."""
        
        # Generate embedding for the state description
        embedding = self.encoder.encode(description, normalize_embeddings=True)
        
        state = FSMState(
            name=name,
            description=description,
            embedding=embedding,
            entry_actions=entry_actions or [],
            exit_actions=exit_actions or [],
            timeout_seconds=timeout_seconds,
            is_final=is_final,
            metadata=metadata or {}
        )
        
        self.states[name] = state
        return state
    
    def add_transition(
        self,
        from_state: str,
        to_state: str,
        trigger: str,
        transition_type: TransitionType = TransitionType.SEMANTIC,
        similarity_threshold: float = 0.7,
        condition_func: Optional[Callable] = None,
        priority: int = 0,
        metadata: Dict[str, Any] = None
    ) -> FSMTransition:
        """Add a transition to the FSM."""
        
        # Generate embedding for semantic transitions
        embedding = None
        if transition_type == TransitionType.SEMANTIC:
            embedding = self.encoder.encode(trigger, normalize_embeddings=True)
        
        transition = FSMTransition(
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            transition_type=transition_type,
            embedding=embedding,
            similarity_threshold=similarity_threshold,
            condition_func=condition_func,
            priority=priority,
            metadata=metadata or {}
        )
        
        self.transitions.append(transition)
        return transition
    
    def set_initial_state(self, state_name: str) -> None:
        """Set the initial state."""
        if state_name not in self.states:
            raise ValueError(f"State '{state_name}' not found")
        self.current_state = state_name
        self.start_time = datetime.utcnow()
        self.state_enter_time = self.start_time
    
    async def process_trigger(self, trigger: str, context_update: Dict[str, Any] = None) -> Optional[str]:
        """
        Process a trigger and potentially transition states.
        
        Args:
            trigger: The trigger string
            context_update: Additional context to update
            
        Returns:
            New state name if transition occurred, None otherwise
        """
        
        if not self.current_state:
            return None
        
        # Update context
        if context_update:
            self.context.update(context_update)
        
        # Find matching transitions
        matching_transitions = await self._find_matching_transitions(trigger)
        
        if not matching_transitions:
            return None
        
        # Select best transition (highest priority, then highest similarity)
        best_transition = max(
            matching_transitions,
            key=lambda t: (t[1].priority, t[0])  # (similarity, transition)
        )[1]
        
        # Execute transition
        return await self._execute_transition(best_transition, trigger)
    
    async def _find_matching_transitions(self, trigger: str) -> List[Tuple[float, FSMTransition]]:
        """Find transitions that match the trigger."""
        
        current_time = datetime.utcnow()
        matching = []
        
        # Generate trigger embedding for semantic matching
        trigger_embedding = self.encoder.encode(trigger, normalize_embeddings=True)
        
        for transition in self.transitions:
            # Check if transition is from current state
            if transition.from_state != self.current_state:
                continue
            
            similarity = 0.0
            matches = False
            
            if transition.transition_type == TransitionType.EXPLICIT:
                # Exact trigger match
                if transition.trigger.lower() == trigger.lower():
                    similarity = 1.0
                    matches = True
            
            elif transition.transition_type == TransitionType.SEMANTIC:
                # Semantic similarity match
                if transition.embedding is not None:
                    similarity = np.dot(trigger_embedding, transition.embedding)
                    matches = similarity >= transition.similarity_threshold
            
            elif transition.transition_type == TransitionType.TEMPORAL:
                # Time-based transition
                if self.state_enter_time:
                    elapsed = (current_time - self.state_enter_time).total_seconds()
                    if elapsed >= float(transition.trigger):  # trigger contains seconds
                        similarity = 1.0
                        matches = True
            
            elif transition.transition_type == TransitionType.CONDITIONAL:
                # Condition-based transition
                if transition.condition_func:
                    try:
                        if await self._evaluate_condition(transition.condition_func):
                            similarity = 1.0
                            matches = True
                    except Exception as e:
                        print(f"Error evaluating condition: {e}")
            
            if matches:
                matching.append((similarity, transition))
        
        return matching
    
    async def _evaluate_condition(self, condition_func: Callable) -> bool:
        """Evaluate a condition function."""
        
        if asyncio.iscoroutinefunction(condition_func):
            return await condition_func(self.context, self.current_state)
        else:
            return condition_func(self.context, self.current_state)
    
    async def _execute_transition(self, transition: FSMTransition, trigger: str) -> str:
        """Execute a state transition."""
        
        old_state = self.current_state
        new_state = transition.to_state
        
        # Execute exit actions for current state
        if old_state in self.states:
            for action in self.states[old_state].exit_actions:
                await self._execute_action(action, "exit", old_state)
        
        # Record transition in history
        self.state_history.append((new_state, datetime.utcnow(), trigger))
        
        # Update current state
        self.current_state = new_state
        self.state_enter_time = datetime.utcnow()
        
        # Execute entry actions for new state
        if new_state in self.states:
            for action in self.states[new_state].entry_actions:
                await self._execute_action(action, "entry", new_state)
        
        return new_state
    
    async def _execute_action(self, action: str, action_type: str, state: str) -> None:
        """Execute a state action."""
        # This would be overridden by the plugin to emit events or call functions
        print(f"Executing {action_type} action '{action}' for state '{state}'")
    
    def get_current_state_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current state."""
        
        if not self.current_state or self.current_state not in self.states:
            return None
        
        state = self.states[self.current_state]
        current_time = datetime.utcnow()
        
        time_in_state = None
        if self.state_enter_time:
            time_in_state = (current_time - self.state_enter_time).total_seconds()
        
        return {
            "name": state.name,
            "description": state.description,
            "time_in_state_seconds": time_in_state,
            "is_final": state.is_final,
            "entry_actions": state.entry_actions,
            "exit_actions": state.exit_actions,
            "timeout_seconds": state.timeout_seconds,
            "metadata": state.metadata
        }
    
    def get_available_transitions(self) -> List[Dict[str, Any]]:
        """Get available transitions from current state."""
        
        if not self.current_state:
            return []
        
        available = []
        for transition in self.transitions:
            if transition.from_state == self.current_state:
                available.append({
                    "to_state": transition.to_state,
                    "trigger": transition.trigger,
                    "type": transition.transition_type.value,
                    "similarity_threshold": transition.similarity_threshold,
                    "priority": transition.priority,
                    "metadata": transition.metadata
                })
        
        return available
    
    def get_state_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent state history."""
        
        history = []
        for state, timestamp, trigger in self.state_history[-limit:]:
            history.append({
                "state": state,
                "timestamp": timestamp.isoformat(),
                "trigger": trigger
            })
        
        return history
    
    def export_definition(self) -> Dict[str, Any]:
        """Export FSM definition for serialization."""
        
        return {
            "name": self.name,
            "states": {
                name: {
                    "description": state.description,
                    "entry_actions": state.entry_actions,
                    "exit_actions": state.exit_actions,
                    "timeout_seconds": state.timeout_seconds,
                    "is_final": state.is_final,
                    "metadata": state.metadata
                }
                for name, state in self.states.items()
            },
            "transitions": [
                {
                    "from_state": t.from_state,
                    "to_state": t.to_state,
                    "trigger": t.trigger,
                    "type": t.transition_type.value,
                    "similarity_threshold": t.similarity_threshold,
                    "priority": t.priority,
                    "metadata": t.metadata
                }
                for t in self.transitions
            ],
            "current_state": self.current_state,
            "context": self.context
        }


class SemanticFSMPlugin(PluginInterface):
    """
    Plugin for managing semantic finite state machines.
    
    Provides flexible workflow management with embedding-based
    state transitions for natural language triggers.
    """
    
    def __init__(self):
        super().__init__()
        self.encoder_model: Optional[SentenceTransformer] = None
        self.fsms: Dict[str, SemanticFSM] = {}
        self.active_fsm: Optional[str] = None
        self.monitoring_task: Optional[asyncio.Task] = None
    
    @property
    def name(self) -> str:
        return "semantic_fsm"
    
    @property
    def description(self) -> str:
        return "Semantic finite state machine for flexible workflow management"
    
    async def setup(self, event_bus, store, config: Dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)
        
        # Initialize encoder
        model_name = self.get_config("encoder_model", "all-MiniLM-L6-v2")
        self.encoder_model = SentenceTransformer(model_name)
        
        # Load predefined FSMs
        await self._load_predefined_fsms()
    
    async def start(self) -> None:
        await super().start()
        
        # Subscribe to events
        await self.subscribe("fsm_state_change", self._handle_state_change_request)
        await self.subscribe("system", self._handle_system_event)
        await self.subscribe("planning", self._handle_planning_event)
        
        # Start monitoring task
        self.monitoring_task = self.add_task(self._monitor_fsms())
    
    async def shutdown(self) -> None:
        if self.monitoring_task:
            self.monitoring_task.cancel()
    
    async def _load_predefined_fsms(self) -> None:
        """Load predefined FSMs for common workflows."""
        
        # Agent lifecycle FSM
        await self._create_agent_lifecycle_fsm()
        
        # Problem solving FSM
        await self._create_problem_solving_fsm()
        
        # Learning FSM
        await self._create_learning_fsm()
    
    async def _create_agent_lifecycle_fsm(self) -> None:
        """Create FSM for agent lifecycle management."""
        
        fsm = SemanticFSM("agent_lifecycle", self.encoder_model)
        
        # States
        fsm.add_state(
            "idle",
            "Agent is idle and waiting for tasks",
            entry_actions=["emit_status_idle"],
            timeout_seconds=300
        )
        
        fsm.add_state(
            "planning",
            "Agent is planning how to approach a task",
            entry_actions=["emit_status_planning"],
            timeout_seconds=60
        )
        
        fsm.add_state(
            "executing",
            "Agent is actively executing a plan",
            entry_actions=["emit_status_executing"],
            timeout_seconds=600
        )
        
        fsm.add_state(
            "reflecting",
            "Agent is reflecting on completed work",
            entry_actions=["emit_status_reflecting"],
            timeout_seconds=30
        )
        
        fsm.add_state(
            "error",
            "Agent encountered an error and needs intervention",
            entry_actions=["emit_status_error"],
            is_final=True
        )
        
        # Transitions
        fsm.add_transition(
            "idle", "planning",
            "new task received",
            TransitionType.SEMANTIC,
            similarity_threshold=0.6
        )
        
        fsm.add_transition(
            "planning", "executing",
            "plan is ready",
            TransitionType.SEMANTIC,
            similarity_threshold=0.7
        )
        
        fsm.add_transition(
            "executing", "reflecting",
            "task completed successfully",
            TransitionType.SEMANTIC,
            similarity_threshold=0.7
        )
        
        fsm.add_transition(
            "reflecting", "idle",
            "reflection complete",
            TransitionType.SEMANTIC,
            similarity_threshold=0.7
        )
        
        # Error transitions from any state
        for state in ["idle", "planning", "executing", "reflecting"]:
            fsm.add_transition(
                state, "error",
                "error occurred",
                TransitionType.SEMANTIC,
                similarity_threshold=0.6,
                priority=10  # High priority
            )
        
        # Set initial state
        fsm.set_initial_state("idle")
        
        self.fsms["agent_lifecycle"] = fsm
        self.active_fsm = "agent_lifecycle"
    
    async def _create_problem_solving_fsm(self) -> None:
        """Create FSM for problem-solving workflow."""
        
        fsm = SemanticFSM("problem_solving", self.encoder_model)
        
        # States
        fsm.add_state(
            "problem_analysis",
            "Analyzing and understanding the problem",
            entry_actions=["emit_analysis_start"]
        )
        
        fsm.add_state(
            "solution_generation",
            "Generating potential solutions",
            entry_actions=["emit_generation_start"]
        )
        
        fsm.add_state(
            "solution_evaluation",
            "Evaluating and ranking solutions",
            entry_actions=["emit_evaluation_start"]
        )
        
        fsm.add_state(
            "implementation",
            "Implementing the chosen solution",
            entry_actions=["emit_implementation_start"]
        )
        
        fsm.add_state(
            "validation",
            "Validating that the solution works",
            entry_actions=["emit_validation_start"]
        )
        
        fsm.add_state(
            "complete",
            "Problem solved successfully",
            entry_actions=["emit_problem_solved"],
            is_final=True
        )
        
        # Transitions
        transitions = [
            ("problem_analysis", "solution_generation", "problem understood"),
            ("solution_generation", "solution_evaluation", "solutions generated"),
            ("solution_evaluation", "implementation", "best solution selected"),
            ("implementation", "validation", "solution implemented"),
            ("validation", "complete", "solution validated"),
            # Back-tracking transitions
            ("solution_evaluation", "solution_generation", "need more solutions"),
            ("validation", "implementation", "solution needs refinement"),
            ("implementation", "solution_evaluation", "solution failed")
        ]
        
        for from_state, to_state, trigger in transitions:
            fsm.add_transition(from_state, to_state, trigger, TransitionType.SEMANTIC)
        
        self.fsms["problem_solving"] = fsm
    
    async def _create_learning_fsm(self) -> None:
        """Create FSM for learning workflow."""
        
        fsm = SemanticFSM("learning", self.encoder_model)
        
        # States
        fsm.add_state(
            "exploration",
            "Exploring and gathering information",
            entry_actions=["emit_exploration_start"]
        )
        
        fsm.add_state(
            "comprehension",
            "Processing and understanding information",
            entry_actions=["emit_comprehension_start"]
        )
        
        fsm.add_state(
            "practice",
            "Practicing and applying knowledge",
            entry_actions=["emit_practice_start"]
        )
        
        fsm.add_state(
            "integration",
            "Integrating new knowledge with existing knowledge",
            entry_actions=["emit_integration_start"]
        )
        
        fsm.add_state(
            "mastery",
            "Knowledge has been mastered",
            entry_actions=["emit_mastery_achieved"],
            is_final=True
        )
        
        # Add transitions
        learning_transitions = [
            ("exploration", "comprehension", "information gathered"),
            ("comprehension", "practice", "concept understood"),
            ("practice", "integration", "skill practiced"),
            ("integration", "mastery", "knowledge integrated"),
            # Revision transitions
            ("comprehension", "exploration", "need more information"),
            ("practice", "comprehension", "concept unclear"),
            ("integration", "practice", "need more practice")
        ]
        
        for from_state, to_state, trigger in learning_transitions:
            fsm.add_transition(from_state, to_state, trigger, TransitionType.SEMANTIC)
        
        self.fsms["learning"] = fsm
    
    async def _handle_state_change_request(self, event) -> None:
        """Handle explicit state change requests."""
        
        fsm_name = getattr(event, 'fsm_name', self.active_fsm)
        trigger = getattr(event, 'trigger', event.to_state)
        context_update = getattr(event, 'context', {})
        
        if fsm_name and fsm_name in self.fsms:
            result = await self.fsms[fsm_name].process_trigger(trigger, context_update)
            
            if result:
                # Emit successful state change
                await self.emit_event(
                    "fsm_state_changed",
                    fsm_name=fsm_name,
                    from_state=event.from_state if hasattr(event, 'from_state') else None,
                    to_state=result,
                    trigger=trigger,
                    context=self.fsms[fsm_name].context
                )
    
    async def _handle_system_event(self, event) -> None:
        """Handle system events and trigger state changes."""
        
        if not self.active_fsm or self.active_fsm not in self.fsms:
            return
        
        # Map system events to FSM triggers
        trigger_mapping = {
            "task_received": "new task received",
            "task_completed": "task completed successfully",
            "error": "error occurred",
            "planning_complete": "plan is ready",
            "reflection_complete": "reflection complete"
        }
        
        trigger = trigger_mapping.get(event.message, event.message)
        
        result = await self.fsms[self.active_fsm].process_trigger(trigger)
        if result:
            await self.emit_event(
                "fsm_state_changed",
                fsm_name=self.active_fsm,
                to_state=result,
                trigger=trigger,
                source_event=event.event_id
            )
    
    async def _handle_planning_event(self, event) -> None:
        """Handle planning events."""
        
        if hasattr(event, 'plan') and event.plan:
            await self._trigger_state_change("plan is ready")
        elif hasattr(event, 'goal'):
            await self._trigger_state_change("new task received")
    
    async def _trigger_state_change(self, trigger: str) -> None:
        """Trigger state change in active FSM."""
        
        if self.active_fsm and self.active_fsm in self.fsms:
            result = await self.fsms[self.active_fsm].process_trigger(trigger)
            if result:
                await self.emit_event(
                    "fsm_state_changed",
                    fsm_name=self.active_fsm,
                    to_state=result,
                    trigger=trigger
                )
    
    async def _monitor_fsms(self) -> None:
        """Monitor FSMs for timeout transitions."""
        
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                for fsm_name, fsm in self.fsms.items():
                    if not fsm.current_state:
                        continue
                    
                    state = fsm.states.get(fsm.current_state)
                    if not state or not state.timeout_seconds:
                        continue
                    
                    if fsm.state_enter_time:
                        elapsed = (datetime.utcnow() - fsm.state_enter_time).total_seconds()
                        if elapsed >= state.timeout_seconds:
                            # Trigger timeout
                            await fsm.process_trigger(f"timeout_{state.timeout_seconds}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in FSM monitoring: {e}")
    
    async def get_fsm_status(self, fsm_name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of FSM(s)."""
        
        if fsm_name:
            if fsm_name in self.fsms:
                fsm = self.fsms[fsm_name]
                return {
                    "name": fsm_name,
                    "current_state": fsm.get_current_state_info(),
                    "available_transitions": fsm.get_available_transitions(),
                    "recent_history": fsm.get_state_history(),
                    "context": fsm.context
                }
            return {"error": f"FSM '{fsm_name}' not found"}
        
        # Return status of all FSMs
        status = {}
        for name, fsm in self.fsms.items():
            status[name] = {
                "current_state": fsm.current_state,
                "is_active": name == self.active_fsm,
                "state_count": len(fsm.states),
                "transition_count": len(fsm.transitions)
            }
        
        return {
            "active_fsm": self.active_fsm,
            "fsms": status
        }
    
    async def switch_active_fsm(self, fsm_name: str) -> bool:
        """Switch the active FSM."""
        
        if fsm_name in self.fsms:
            self.active_fsm = fsm_name
            await self.emit_event(
                "fsm_switched",
                new_active_fsm=fsm_name
            )
            return True
        
        return False
