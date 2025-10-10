"""
EOS State Machine Implementation

Implements the state machine lifecycle for E-UPUSF orchestration with
support for non-linear cycles, guards, triggers, and telemetry.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class StateType(Enum):
    """EOS state types"""
    OBSERVE = "Observe"
    ANALYZE = "Analyze" 
    SYNTHESIZE = "Synthesize"
    IMPLEMENT = "Implement"
    EVALUATE = "Evaluate"


class TransitionType(Enum):
    """Types of state transitions"""
    FORWARD = "forward"      # Normal progression
    CYCLE = "cycle"          # Permitted non-linear cycle
    EMERGENCY = "emergency"  # Emergency jump to Observe
    HIL = "hil"             # Human-in-loop checkpoint


@dataclass
class StateContext:
    """Context passed between states"""
    run_id: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    evidence_score: float = 0.0
    contradiction_debt: float = 0.0
    breadth: int = 0
    quality: float = 0.0
    chaotic_mode: bool = False
    budget_remaining: dict[str, float] = field(default_factory=dict)
    time_remaining: dict[str, float] = field(default_factory=dict)


@dataclass
class TransitionDecision:
    """Decision about state transition"""
    target_state: StateType
    transition_type: TransitionType
    reason: str
    confidence: float = 1.0


class Guard(ABC):
    """Abstract base class for state guards"""
    
    @abstractmethod
    def evaluate(self, context: StateContext, 
                 spec: dict[str, Any]) -> bool:
        """Evaluate guard condition"""
        pass


class ContextShiftGuard(Guard):
    """Guard for detecting context shifts to chaotic"""
    
    def evaluate(self, context: StateContext, 
                 spec: dict[str, Any]) -> bool:
        # Check if chaotic probability exceeded threshold
        tau_emergency = spec.get("context", {}).get(
            "uncertainty_thresholds", {}
        ).get("chaotic_emergency", 0.4)
        
        # Would need actual context classifier here
        # For now, use chaotic_mode flag from context
        return context.chaotic_mode and context.metrics.get(
            "p_chaotic", 0.0
        ) > tau_emergency


class SufficiencyGuard(Guard):
    """Guard for checking evidence sufficiency"""
    
    def evaluate(self, context: StateContext, 
                 spec: dict[str, Any]) -> bool:
        tau_evidence = 0.5  # Could be configurable
        tau_contra = 0.3    # Could be configurable
        
        return (context.evidence_score >= tau_evidence and 
                context.contradiction_debt <= tau_contra)


class BudgetGuard(Guard):
    """Guard for checking resource budgets"""
    
    def evaluate(self, context: StateContext, 
                 spec: dict[str, Any]) -> bool:
        # Check if any budget fell below reserve
        reserve_fraction = spec.get("routing", {}).get(
            "budgets", {}
        ).get("reserve_fraction", 0.2)
        
        for resource, remaining in context.budget_remaining.items():
            if remaining < reserve_fraction:
                return False
        return True


class LambdaGuard(Guard):
    """Guard that wraps a lambda function"""
    
    def __init__(self, func: Callable[[StateContext, dict[str, Any]], bool]):
        self.func = func
    
    def evaluate(self, context: StateContext, 
                 spec: dict[str, Any]) -> bool:
        return self.func(context, spec)


class TimeGuard(Guard):
    """Guard for checking time limits"""
    
    def __init__(self, state_name: str):
        self.state_name = state_name
    
    def evaluate(self, context: StateContext, 
                 spec: dict[str, Any]) -> bool:
        # Check if time limit exceeded for this state
        time_limits = spec.get("resources", {}).get(
            "time_limits", {}
        ).get("per_state_minutes", {})
        
        limit = time_limits.get(self.state_name, float('inf'))
        remaining = context.time_remaining.get(self.state_name, limit)
        
        return remaining is not None and remaining > 0


class State(ABC):
    """Abstract base class for EOS states"""
    
    def __init__(self, state_type: StateType, spec: dict[str, Any]):
        self.state_type = state_type
        self.spec = spec
        self.entry_guards: list[Guard] = []
        self.exit_guards: list[Guard] = []
        self.start_time: float | None = None
    
    @abstractmethod
    async def execute(self, context: StateContext) -> StateContext:
        """Execute state logic"""
        pass
    
    def can_enter(self, context: StateContext) -> bool:
        """Check if state can be entered"""
        return all(guard.evaluate(context, self.spec)
                   for guard in self.entry_guards)
    
    def can_exit(self, context: StateContext) -> bool:
        """Check if state can be exited"""
        return all(guard.evaluate(context, self.spec)
                   for guard in self.exit_guards)
    
    def enter(self, context: StateContext) -> None:
        """Called when entering state"""
        self.start_time = time.time()
        logger.info(f"Entering state: {self.state_type.value}")
    
    def exit(self, context: StateContext) -> None:
        """Called when exiting state"""
        if self.start_time:
            duration = time.time() - self.start_time
            context.metrics[f"{self.state_type.value}_duration_ms"] = (
                duration * 1000
            )
        logger.info(f"Exiting state: {self.state_type.value}")


class ObserveState(State):
    """Observe state implementation"""
    
    def __init__(self, spec: dict[str, Any]):
        super().__init__(StateType.OBSERVE, spec)
        # Entry guard always true (can always observe)
        self.exit_guards = [SufficiencyGuard(), TimeGuard("Observe")]
    
    async def execute(self, context: StateContext) -> StateContext:
        """Execute observation logic"""
        logger.info("Executing Observe state")
        
        # Simulate observation work
        await asyncio.sleep(0.1)
        
        # Update context with observations
        context.evidence_score += 0.3
        context.artifacts["observations"] = {
            "timestamp": time.time(),
            "data": "Mock observation data"
        }
        
        return context


class AnalyzeState(State):
    """Analyze state implementation"""
    
    def __init__(self, spec: dict[str, Any]):
        super().__init__(StateType.ANALYZE, spec)
        # Cannot enter if in chaotic mode
        self.entry_guards = [LambdaGuard(
            lambda ctx, spec: not ctx.chaotic_mode
        )]
        self.exit_guards = [SufficiencyGuard(), TimeGuard("Analyze")]
    
    async def execute(self, context: StateContext) -> StateContext:
        """Execute analysis logic"""
        logger.info("Executing Analyze state")
        
        # Simulate analysis work
        await asyncio.sleep(0.1)
        
        # Update context with analysis
        context.evidence_score += 0.2
        context.contradiction_debt = max(0, context.contradiction_debt - 0.1)
        context.artifacts["analysis"] = {
            "timestamp": time.time(),
            "contradictions": [],
            "patterns": []
        }
        
        return context


class SynthesizeState(State):
    """Synthesize state implementation"""
    
    def __init__(self, spec: dict[str, Any]):
        super().__init__(StateType.SYNTHESIZE, spec)
        self.exit_guards = [
            LambdaGuard(
                lambda ctx, spec: ctx.breadth >= 5 and ctx.quality >= 0.7
            ),
            TimeGuard("Synthesize")
        ]
    
    async def execute(self, context: StateContext) -> StateContext:
        """Execute synthesis logic"""
        logger.info("Executing Synthesize state")
        
        # Simulate synthesis work
        await asyncio.sleep(0.1)
        
        # Update context with synthesis results
        context.breadth += 2
        context.quality = min(1.0, context.quality + 0.2)
        context.artifacts["synthesis"] = {
            "timestamp": time.time(),
            "alternatives": ["option1", "option2", "option3"],
            "quality_score": context.quality
        }
        
        return context


class ImplementState(State):
    """Implement state implementation"""
    
    def __init__(self, spec: dict[str, Any]):
        super().__init__(StateType.IMPLEMENT, spec)
        self.exit_guards = [
            LambdaGuard(
                lambda ctx, spec: ctx.artifacts.get(
                    "rollout_plan_ready", False
                )
            ),
            TimeGuard("Implement")
        ]
    
    async def execute(self, context: StateContext) -> StateContext:
        """Execute implementation logic"""
        logger.info("Executing Implement state")
        
        # Simulate implementation work
        await asyncio.sleep(0.1)
        
        # Update context with implementation plan
        context.artifacts["implementation"] = {
            "timestamp": time.time(),
            "rollout_plan": {"phase1": "setup", "phase2": "deploy"}
        }
        context.artifacts["rollout_plan_ready"] = True
        
        return context


class EvaluateState(State):
    """Evaluate state implementation"""
    
    def __init__(self, spec: dict[str, Any]):
        super().__init__(StateType.EVALUATE, spec)
        self.exit_guards = [
            LambdaGuard(
                lambda ctx, spec: (
                    ctx.metrics.get("success_kpis", 0) >=
                    spec.get("evaluation", {}).get("target_kpis", 0.8)
                )
            ),
            TimeGuard("Evaluate")
        ]
    
    async def execute(self, context: StateContext) -> StateContext:
        """Execute evaluation logic"""
        logger.info("Executing Evaluate state")
        
        # Simulate evaluation work
        await asyncio.sleep(0.1)
        
        # Update context with evaluation results
        context.metrics["success_kpis"] = 0.85
        context.artifacts["evaluation"] = {
            "timestamp": time.time(),
            "kpis": {"metric1": 0.9, "metric2": 0.8},
            "success": True
        }
        
        return context


class Transition:
    """State transition definition"""
    
    def __init__(self, from_state: StateType, to_state: StateType,
                 transition_type: TransitionType,
                 condition: Callable[[StateContext, dict[str, Any]], bool]):
        self.from_state = from_state
        self.to_state = to_state
        self.transition_type = transition_type
        self.condition = condition
    
    def can_transition(self, context: StateContext,
                       spec: dict[str, Any]) -> bool:
        """Check if transition is allowed"""
        return self.condition(context, spec)


class EOSStateMachine:
    """E-UPUSF Orchestration State Machine"""
    
    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.current_state: State | None = None
        self.context: StateContext | None = None
        
        # Initialize states
        self.states = {
            StateType.OBSERVE: ObserveState(spec),
            StateType.ANALYZE: AnalyzeState(spec),
            StateType.SYNTHESIZE: SynthesizeState(spec),
            StateType.IMPLEMENT: ImplementState(spec),
            StateType.EVALUATE: EvaluateState(spec)
        }
        
        # Define permitted transitions
        self.transitions = self._create_transitions()
        
        # Guards
        self.context_shift_guard = ContextShiftGuard()
        self.sufficiency_guard = SufficiencyGuard()
        self.budget_guard = BudgetGuard()
    
    def _create_transitions(self) -> list[Transition]:
        """Create transition rules"""
        transitions = []
        
        # Normal forward progression
        flow = [StateType.OBSERVE, StateType.ANALYZE, StateType.SYNTHESIZE,
                StateType.IMPLEMENT, StateType.EVALUATE]
        
        for i in range(len(flow) - 1):
            transitions.append(Transition(
                from_state=flow[i],
                to_state=flow[i + 1],
                transition_type=TransitionType.FORWARD,
                condition=lambda ctx, spec: True  # Always allowed
            ))
        
        # Permitted non-linear cycles
        
        # Implement → Observe (new signals discovered)
        transitions.append(Transition(
            from_state=StateType.IMPLEMENT,
            to_state=StateType.OBSERVE,
            transition_type=TransitionType.CYCLE,
            condition=lambda ctx, spec: bool(ctx.metrics.get(
                "new_signals_discovered", False
            ))
        ))
        
        # Evaluate → Analyze (unmet targets)
        transitions.append(Transition(
            from_state=StateType.EVALUATE,
            to_state=StateType.ANALYZE,
            transition_type=TransitionType.CYCLE,
            condition=lambda ctx, spec: ctx.metrics.get(
                "success_kpis", 0
            ) < spec.get("evaluation", {}).get("target_kpis", 0.8)
        ))
        
        # Synthesize ↔ Analyze (exploration/refinement)
        transitions.append(Transition(
            from_state=StateType.SYNTHESIZE,
            to_state=StateType.ANALYZE,
            transition_type=TransitionType.CYCLE,
            condition=lambda ctx, spec: ctx.quality < 0.5
        ))
        
        transitions.append(Transition(
            from_state=StateType.ANALYZE,
            to_state=StateType.SYNTHESIZE,
            transition_type=TransitionType.CYCLE,
            condition=lambda ctx, spec: ctx.contradiction_debt <= 0.3
        ))
        
        # Emergency transitions (* → Observe if chaotic)
        for state in StateType:
            if state != StateType.OBSERVE:
                transitions.append(Transition(
                    from_state=state,
                    to_state=StateType.OBSERVE,
                    transition_type=TransitionType.EMERGENCY,
                    condition=lambda ctx, spec: (
                        ctx.chaotic_mode and 
                        ctx.metrics.get("p_chaotic", 0) > 
                        spec.get("context", {}).get(
                            "uncertainty_thresholds", {}
                        ).get("chaotic_emergency", 0.4)
                    )
                ))
        
        return transitions
    
    def start(self, run_id: str) -> StateContext:
        """Start state machine execution"""
        self.context = StateContext(run_id=run_id)
        
        # Initialize budgets from spec
        budgets = self.spec.get("resources", {}).get("budgets", {})
        self.context.budget_remaining = budgets.copy()
        
        # Initialize time limits
        time_limits = self.spec.get("resources", {}).get(
            "time_limits", {}
        ).get("per_state_minutes", {})
        self.context.time_remaining = time_limits.copy()
        
        # Start with Observe state
        self.current_state = self.states[StateType.OBSERVE]
        self.current_state.enter(self.context)
        
        logger.info(f"Started EOS state machine with run_id: {run_id}")
        return self.context
    
    async def step(self) -> TransitionDecision | None:
        """Execute one state machine step"""
        if not self.current_state or not self.context:
            raise RuntimeError("State machine not started")
        
        # Execute current state
        self.context = await self.current_state.execute(self.context)
        
        # Check for state transition
        decision = self._decide_transition()
        
        if decision:
            await self._transition_to(decision)
        
        return decision
    
    def _decide_transition(self) -> TransitionDecision | None:
        """Decide on state transition"""
        if not self.current_state or not self.context:
            return None
        
        current_type = self.current_state.state_type
        
        # Check emergency transitions first
        for transition in self.transitions:
            if (transition.from_state == current_type and
                    transition.transition_type == TransitionType.EMERGENCY and
                    transition.can_transition(self.context, self.spec)):
                
                return TransitionDecision(
                    target_state=transition.to_state,
                    transition_type=transition.transition_type,
                    reason="Emergency transition due to chaotic context",
                    confidence=1.0
                )
        
        # Check if current state can exit
        if not self.current_state.can_exit(self.context):
            return None  # Stay in current state
        
        # Find best transition
        valid_transitions = []
        for transition in self.transitions:
            if (transition.from_state == current_type and
                    transition.can_transition(self.context, self.spec)):
                valid_transitions.append(transition)
        
        if not valid_transitions:
            return None
        
        # Prefer forward transitions, then cycles
        forward_transitions = [t for t in valid_transitions
                               if t.transition_type == TransitionType.FORWARD]
        if forward_transitions:
            transition = forward_transitions[0]
        else:
            transition = valid_transitions[0]
        
        return TransitionDecision(
            target_state=transition.to_state,
            transition_type=transition.transition_type,
            reason=f"Transition from {current_type.value} to "
                   f"{transition.to_state.value}",
            confidence=0.8
        )
    
    async def _transition_to(self, decision: TransitionDecision) -> None:
        """Execute state transition"""
        if not self.current_state or not self.context:
            return
        
        # Exit current state
        self.current_state.exit(self.context)
        
        # Enter new state
        new_state = self.states[decision.target_state]
        if new_state.can_enter(self.context):
            self.current_state = new_state
            self.current_state.enter(self.context)
            
            logger.info(
                f"Transitioned to {decision.target_state.value} "
                f"(type: {decision.transition_type.value}): "
                f"{decision.reason}"
            )
        else:
            logger.warning(
                f"Cannot enter state {decision.target_state.value}, "
                "staying in current state"
            )
    
    async def run_until_complete(self, max_steps: int = 100) -> StateContext:
        """Run state machine until completion or max steps"""
        if not self.context:
            raise RuntimeError("State machine not started")
        
        steps = 0
        while steps < max_steps:
            decision = await self.step()
            steps += 1
            
            # Check stopping conditions
            if self._should_stop():
                break
            
            # Prevent infinite loops
            if not decision:
                # No transition possible, check if we're stuck
                if not self.current_state or not self.current_state.can_exit(
                    self.context
                ):
                    # Stuck in current state, wait a bit then try again
                    await asyncio.sleep(0.1)
                    continue
                else:
                    # No valid transitions but can exit - might be complete
                    break
        
        logger.info(f"State machine completed after {steps} steps")
        return self.context
    
    def _should_stop(self) -> bool:
        """Check if state machine should stop"""
        if not self.context:
            return True
        
        # Check evaluation stopping conditions
        eval_config = self.spec.get("evaluation", {})
        stopping_conditions = eval_config.get("stopping", [])
        
        for condition in stopping_conditions:
            condition_str = condition.get("condition", "")
            
            if condition_str == "all_targets_met":
                # Check if all targets are met
                metrics = eval_config.get("metrics", [])
                all_met = True
                for metric in metrics:
                    metric_id = metric["id"]
                    target = metric["target"]
                    
                    if metric_id == "time_to_insight":
                        max_minutes = target.get("max_minutes", float('inf'))
                        actual = self.context.metrics.get(
                            "total_duration_ms", 0
                        ) / (1000 * 60)
                        if actual > max_minutes:
                            all_met = False
                            break
                    
                    elif metric_id == "breadth":
                        min_alternatives = target.get("min_alternatives", 0)
                        if self.context.breadth < min_alternatives:
                            all_met = False
                            break
                
                if all_met:
                    return True
            
            elif condition_str == "budget_exhausted and marginal_gain < 0.02":
                # Simplified budget check
                if (self.context.budget_remaining.get("compute_gh", 1.0) <= 0
                        or self.context.metrics.get(
                            "marginal_gain", 1.0
                        ) < 0.02):
                    return True
        
        return False